// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "dev_mem_map.h"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_fw_loader.hpp"
#include "tt_metal/tt_metal/dram/mimir_hw_state.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::fw_load {

constexpr ChipId kFwLoadChip = 0;
constexpr const char* kNullKernelPath = "tests/tt_metal/tt_metal/test_kernels/cce/cce_null.cpp";
constexpr uint32_t kDramDmProcessorClass = static_cast<uint32_t>(HalProcessorClassType::DM);
constexpr uint64_t kFirmwareTextMaxBytes = MEM_CCE_FIRMWARE_SIZE;
constexpr uint32_t kReinitCycles = 3;

uint32_t emu_cce_count() { return topology_from(configured_soc_desc()).num_cces; }

const HalJitBuildConfig& cce_jit_build_config() {
    const Hal& hal = MetalContext::instance().hal();
    return hal.get_jit_build_config(
        hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM), kDramDmProcessorClass, 0);
}

uint64_t mailbox_field(dev_msgs::mailboxes_t::Field field) {
    const Hal& hal = MetalContext::instance().hal();
    return hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX) +
           hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).offset_of<dev_msgs::mailboxes_t>(field);
}

uint8_t read_go_signal(const tt::Cluster& cluster, ChipId chip, uint32_t cce_index) {
    const std::vector<uint32_t> words = read_cce_tl1(
        cluster, chip, cce_index, go_message_tl1_offset(), static_cast<uint32_t>(sizeof(uint32_t)), Tl1Range::WholeTl1);
    return go_signal_of(words.front());
}

CoreCoord read_core_info_xy(IDevice* device, uint32_t cce_index) {
    const Hal& hal = MetalContext::instance().hal();
    dev_msgs::core_info_msg_t core_info =
        hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::core_info_msg_t>();
    TT_FATAL(
        core_info.size() % sizeof(uint32_t) == 0, "core_info_msg_t is {} bytes, not whole words", core_info.size());
    const std::vector<uint32_t> words = read_cce_tl1(
        device,
        cce_index,
        hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::CORE_INFO),
        static_cast<uint32_t>(core_info.size()),
        Tl1Range::WholeTl1);
    std::memcpy(core_info.data(), words.data(), core_info.size());
    return {core_info.view().absolute_logical_x(), core_info.view().absolute_logical_y()};
}

uint64_t read_subordinate_sync(IDevice* device, uint32_t cce_index) {
    const std::vector<uint32_t> words = read_cce_tl1(
        device,
        cce_index,
        mailbox_field(dev_msgs::mailboxes_t::Field::subordinate_sync),
        sizeof(uint64_t),
        Tl1Range::WholeTl1);
    return words[0] | (static_cast<uint64_t>(words[1]) << 32);
}

uint8_t read_globals_ready(IDevice* device, uint32_t cce_index) {
    const uint64_t offset = mailbox_field(dev_msgs::mailboxes_t::Field::fw_shared_globals_ready);
    const uint64_t aligned_offset = offset & ~uint64_t{sizeof(uint32_t) - 1};
    const std::vector<uint32_t> words =
        read_cce_tl1(device, cce_index, aligned_offset, sizeof(uint32_t), Tl1Range::WholeTl1);
    return static_cast<uint8_t>((words[0] >> ((offset & 3) * 8)) & 0xFF);
}

uint32_t read_launch_enables(IDevice* device, uint32_t cce_index) {
    const Hal& hal = MetalContext::instance().hal();
    dev_msgs::launch_msg_t launch_message =
        hal.get_dev_msgs_factory(HalProgrammableCoreType::DRAM).create<dev_msgs::launch_msg_t>();
    TT_FATAL(
        launch_message.size() % sizeof(uint32_t) == 0,
        "launch_msg_t is {} bytes, not whole words",
        launch_message.size());
    const std::vector<uint32_t> words = read_cce_tl1(
        device,
        cce_index,
        hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::LAUNCH),
        static_cast<uint32_t>(launch_message.size()),
        Tl1Range::WholeTl1);
    std::memcpy(launch_message.data(), words.data(), launch_message.size());
    return launch_message.view().kernel_config().enables();
}

std::vector<uint32_t> read_mailbox(IDevice* device, uint32_t cce_index) {
    const Hal& hal = MetalContext::instance().hal();
    return read_cce_tl1(
        device,
        cce_index,
        hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX),
        hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX),
        Tl1Range::WholeTl1);
}

void expect_cce_initialized(IDevice* device, uint32_t cce_index) {
    const tt::Cluster& cluster = MetalContext::instance().get_cluster();
    const ChipId chip = device->id();
    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t expected_reset_vector = cce_jit_build_config().fw_launch_addr_value;

    EXPECT_EQ(read_globals_ready(device, cce_index), dev_msgs::SHARED_GLOBALS_READY_GO) << "CCE " << cce_index;
    EXPECT_EQ(read_subordinate_sync(device, cce_index), dev_msgs::RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE)
        << "CCE " << cce_index;
    const std::vector<uint64_t> reset_vectors = read_cce_reset_vectors(cluster, soc, chip, cce_index);
    for (uint32_t hart = 0; hart < cce_hart_count(); ++hart) {
        EXPECT_EQ(static_cast<uint32_t>(reset_vectors[hart]), expected_reset_vector)
            << "CCE " << cce_index << " hart " << hart << " low word of 0x" << std::hex << reset_vectors[hart];
    }
    const CceResetState reset_state = read_cce_reset_states(cluster, soc, chip).at(cce_index);
    EXPECT_EQ(reset_state.core_reset, 0xFFu) << "CCE " << cce_index;
    EXPECT_EQ(reset_state.raw_lo, 0x1FFu) << "CCE " << cce_index;
}

void add_null_kernel(Program& program, IDevice* device, uint32_t cce_index) {
    CreateKernel(
        program,
        kNullKernelPath,
        logical_cce_core(soc_desc_of(device), cce_index),
        DramConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
}

void launch_null_kernel(IDevice* device, const std::vector<uint32_t>& cce_indices) {
    Program program = CreateProgram();
    for (uint32_t cce_index : cce_indices) {
        add_null_kernel(program, device, cce_index);
    }
    launch_program_and_wait(device, program);
}

// An earlier process's firmware stays in TL1, and the released harts would run it.
void zero_firmware_windows(const tt::Cluster& cluster) {
    for (const CceResetState& reset_state :
         read_cce_reset_states(cluster, cluster.get_soc_desc(kFwLoadChip), kFwLoadChip)) {
        ASSERT_TRUE(reset_state.uncore_released)
            << "CCE " << reset_state.cce_index << " uncore is in reset, so TL1 is unreachable";
    }
    const std::vector<uint32_t> zero_firmware(kFirmwareTextMaxBytes / sizeof(uint32_t), 0);
    for (uint32_t cce_index = 0; cce_index < emu_cce_count(); ++cce_index) {
        write_cce_tl1(
            cluster, kFwLoadChip, cce_index, cce_jit_build_config().fw_base_addr, zero_firmware, Tl1Range::WholeTl1);
    }
}

// Defined first so that it is the first device open of its invocation: a later device open in the same process does not
// reload firmware.
TEST(MimirFwLoad, LoadAndReachInitDone) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    const tt::Cluster& cluster = MetalContext::instance().get_cluster();
    ASSERT_NO_FATAL_FAILURE(zero_firmware_windows(cluster));
    for (uint32_t cce_index = 0; cce_index < emu_cce_count(); ++cce_index) {
        ASSERT_NO_FATAL_FAILURE(poison_cce_reset_vectors(cluster, kFwLoadChip, cce_index));
    }
    ScopedDevice device_guard;
    IDevice* device = device_guard.device();
    for (uint32_t cce_index = 0; cce_index < emu_cce_count(); ++cce_index) {
        expect_cce_initialized(device, cce_index);
    }

    const MetalContext& context = MetalContext::instance();
    const Hal& hal = context.hal();
    const uint32_t dram_core_type_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM);
    const std::string binary_path =
        BuildEnvManager::get_instance(context.get_context_id())
            .get_firmware_binary_path(device->id(), dram_core_type_index, kDramDmProcessorClass, 0);
    const std::vector<CceFwSpan> spans = cce_fw_spans(binary_path);
    const uint64_t firmware_window_begin = cce_jit_build_config().fw_base_addr;
    const std::vector<CceFwSpan>::const_iterator text_span_iterator =
        std::find_if(spans.begin(), spans.end(), [firmware_window_begin](const CceFwSpan& span) {
            return span.tl1_offset == firmware_window_begin;
        });
    ASSERT_NE(text_span_iterator, spans.end()) << "no span starts at the firmware base (fw_base_addr)";
    for (uint32_t cce_index = 0; cce_index < emu_cce_count(); ++cce_index) {
        const std::vector<uint32_t> readback = read_cce_tl1(
            device,
            cce_index,
            text_span_iterator->tl1_offset,
            static_cast<uint32_t>(text_span_iterator->words.size() * sizeof(uint32_t)),
            Tl1Range::WholeTl1);
        EXPECT_TRUE(compare_words(fmt::format("CCE {} firmware text", cce_index), text_span_iterator->words, readback));
    }
}

TEST(MimirFwLoad, ImageManifest) {
    skip_unless_programmable_dram_package();
    ScopedDevice device_guard;
    IDevice* device = device_guard.device();

    const MetalContext& context = MetalContext::instance();
    const Hal& hal = context.hal();
    const uint32_t dram_core_type_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::DRAM);
    const std::string binary_path =
        BuildEnvManager::get_instance(context.get_context_id())
            .get_firmware_binary_path(device->id(), dram_core_type_index, kDramDmProcessorClass, 0);
    const HalJitBuildConfig& jit_build_config = cce_jit_build_config();
    const std::vector<CceFwSpan> spans = cce_fw_spans(binary_path);
    const uint64_t firmware_window_begin = jit_build_config.fw_base_addr;
    const uint64_t firmware_window_end = firmware_window_begin + MEM_CCE_FIRMWARE_SIZE;
    const uint64_t init_scratch_begin = jit_build_config.local_init_addr;
    const uint64_t init_scratch_end = init_scratch_begin + MEM_CCE_INIT_LOCAL_L1_SCRATCH_SIZE;
    const uint64_t firmware_region_begin = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX) +
                                           hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::MAILBOX);
    const uint64_t firmware_region_end = init_scratch_begin;

    ASSERT_FALSE(spans.empty());
    log_info(tt::LogTest, "CCE firmware {}", binary_path);
    EXPECT_EQ(static_cast<uint64_t>(jit_build_config.fw_launch_addr_value), firmware_window_begin);

    // Text is the span that starts at the launch address. Its unclipped length must fit the firmware window.
    const std::vector<CceFwSpan>::const_iterator text_span_iterator =
        std::find_if(spans.begin(), spans.end(), [firmware_window_begin](const CceFwSpan& span) {
            return span.tl1_offset == firmware_window_begin;
        });
    ASSERT_NE(text_span_iterator, spans.end()) << "no span starts at the firmware base (fw_base_addr)";
    const uint64_t text_span_bytes = text_span_iterator->words.size() * sizeof(uint32_t);
    EXPECT_LE(text_span_bytes, kFirmwareTextMaxBytes) << "text span is " << text_span_bytes << " bytes";

    for (const CceFwSpan& span : spans) {
        const uint64_t span_bytes = span.words.size() * sizeof(uint32_t);
        const uint64_t span_end = span.tl1_offset + span_bytes;
        const bool in_firmware_region = span.tl1_offset >= firmware_region_begin && span_end <= firmware_region_end;
        const bool in_init_scratch = span.tl1_offset >= init_scratch_begin && span_end <= init_scratch_end;
        log_info(tt::LogTest, "CCE firmware span TL1 [0x{:x}, 0x{:x}) {} bytes", span.tl1_offset, span_end, span_bytes);
        EXPECT_TRUE(in_firmware_region || in_init_scratch) << fmt::format(
            "span TL1 [0x{:x}, 0x{:x}) is outside [MAILBOX_END, MAP_END) and the init scratch",
            span.tl1_offset,
            span_end);
        const bool overlaps_firmware_window = span.tl1_offset < firmware_window_end && span_end > firmware_window_begin;
        const bool inside_firmware_window = span.tl1_offset >= firmware_window_begin && span_end <= firmware_window_end;
        EXPECT_FALSE(overlaps_firmware_window && !inside_firmware_window) << fmt::format(
            "span TL1 [0x{:x}, 0x{:x}) straddles the firmware window [0x{:x}, 0x{:x})",
            span.tl1_offset,
            span_end,
            firmware_window_begin,
            firmware_window_end);
    }

    // One binary for all harts: a single image, and every hart's launch config matches hart 0's.
    EXPECT_EQ(hal.get_processor_class_num_fw_binaries(dram_core_type_index, kDramDmProcessorClass), 1u);
    for (uint32_t hart = 0; hart < hal.get_processor_types_count(dram_core_type_index, kDramDmProcessorClass); ++hart) {
        const HalJitBuildConfig& hart_config =
            hal.get_jit_build_config(dram_core_type_index, kDramDmProcessorClass, hart);
        EXPECT_EQ(hart_config.fw_launch_addr, jit_build_config.fw_launch_addr) << "hart " << hart;
        EXPECT_EQ(hart_config.fw_launch_addr_value, jit_build_config.fw_launch_addr_value) << "hart " << hart;
    }
}

TEST(MimirFwLoad, AllCcesIndependent) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_count = emu_cce_count();
    if (cce_count < 2) {
        GTEST_SKIP() << "Configured descriptor exposes " << cce_count << " CCEs.";
    }
    ScopedDevice device_guard;
    IDevice* device = device_guard.device();

    for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
        const CoreCoord expected_translated_core = translated_dram_core(soc_desc_of(device), cce_index);
        const CoreCoord core_info_xy = read_core_info_xy(device, cce_index);
        EXPECT_EQ(core_info_xy.x, static_cast<uint8_t>(expected_translated_core.x)) << "CCE " << cce_index;
        EXPECT_EQ(core_info_xy.y, static_cast<uint8_t>(expected_translated_core.y)) << "CCE " << cce_index;
    }

    for (uint32_t launched_cce = 0; launched_cce < cce_count; ++launched_cce) {
        std::vector<std::vector<uint32_t>> mailboxes_before(cce_count);
        for (uint32_t other_cce = 0; other_cce < cce_count; ++other_cce) {
            if (other_cce != launched_cce) {
                mailboxes_before[other_cce] = read_mailbox(device, other_cce);
            }
        }

        launch_null_kernel(device, {launched_cce});

        for (uint32_t other_cce = 0; other_cce < cce_count; ++other_cce) {
            if (other_cce != launched_cce) {
                EXPECT_TRUE(compare_words(
                    fmt::format("CCE {} mailbox while CCE {} ran", other_cce, launched_cce),
                    mailboxes_before[other_cce],
                    read_mailbox(device, other_cce)));
            }
        }
    }
}

class MimirFwLaunch : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirFwLaunch, NoOpLaunchCompletes) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const uint32_t cce_index = GetParam();
    ScopedDevice device_guard;
    IDevice* device = device_guard.device();

    Program program = CreateProgram();
    add_null_kernel(program, device, cce_index);
    constexpr uint32_t kLaunchRepeats = 2;
    for (uint32_t repeat = 0; repeat < kLaunchRepeats; ++repeat) {
        launch_program_and_wait(device, program);
        // The firmware skips a kernel whose enable bit is clear and still reports DONE.
        EXPECT_NE(read_launch_enables(device, cce_index), 0u) << "launch " << repeat;
    }
}

TEST(MimirFwReinit, ReloadsOnEveryOpen) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    if (!MetalContext::instance().rtoptions().get_force_context_reinit()) {
        GTEST_SKIP() << "Set TT_METAL_FORCE_REINIT=1.";
    }
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }
    const uint32_t cce_count = emu_cce_count();
    std::vector<uint32_t> all_cces(cce_count);
    for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
        all_cces[cce_index] = cce_index;
    }

    for (uint32_t cycle = 0; cycle < kReinitCycles; ++cycle) {
        ScopedDevice device_guard;
        IDevice* device = device_guard.device();

        for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
            expect_cce_initialized(device, cce_index);
            EXPECT_EQ(read_launch_enables(device, cce_index), 0u) << "cycle " << cycle << " CCE " << cce_index;
        }

        launch_null_kernel(device, all_cces);

        for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
            EXPECT_NE(read_launch_enables(device, cce_index), 0u) << "cycle " << cycle << " CCE " << cce_index;
        }
    }
}

TEST(MimirFwLoadNegative, SkipLoadNeverDone) {
    skip_unless_programmable_dram_package();
    skip_unless_mimir_local_package();
    const llrt::RunTimeOptions& rtoptions = MetalContext::instance().rtoptions();
    if (!rtoptions.get_skip_loading_fw()) {
        GTEST_SKIP() << "Set TT_METAL_SKIP_LOADING_FW=1.";
    }
    if (const std::string skip_reason = mimir_watcher_skip_reason(); !skip_reason.empty()) {
        GTEST_SKIP() << skip_reason;
    }

    const tt::Cluster& cluster = MetalContext::instance().get_cluster();
    ASSERT_NO_FATAL_FAILURE(zero_firmware_windows(cluster));

    try {
        ScopedDevice device_guard;
        ADD_FAILURE() << "Device open succeeded without loading CCE firmware";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("failed to initialize DRAM FW"), std::string::npos)
            << "unexpected error: " << error.what();
    }

    // INIT is the runtime's own write, so the released harts are what show the firmware got its chance to run.
    const std::vector<CceResetState> reset_states =
        read_cce_reset_states(cluster, cluster.get_soc_desc(kFwLoadChip), kFwLoadChip);
    for (uint32_t cce_index = 0; cce_index < emu_cce_count(); ++cce_index) {
        EXPECT_EQ(read_go_signal(cluster, kFwLoadChip, cce_index), dev_msgs::RUN_MSG_INIT) << "CCE " << cce_index;
        EXPECT_EQ(reset_states.at(cce_index).core_reset, 0xFFu) << "CCE " << cce_index;
    }
}

INSTANTIATE_TEST_SUITE_P(MM2, MimirFwLaunch, ::testing::ValuesIn(mimir_contract::cce_indices()), cce_param_name);

}  // namespace tt::tt_metal::mimir_test::fw_load
