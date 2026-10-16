// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/tt_metal/dram/mimir_access.hpp"
#include "tt_metal/tt_metal/dram/mimir_patterns.hpp"
#include "tt_metal/tt_metal/dram/mimir_test_common.hpp"

namespace tt::tt_metal::mimir_test::kernel_load {

constexpr const char* kNullKernelPath = "tests/tt_metal/tt_metal/test_kernels/cce/cce_null.cpp";
constexpr const char* kEchoKernelPath = "tests/tt_metal/tt_metal/test_kernels/cce/cce_args_echo.cpp";

constexpr uint32_t kEchoCtMarkerBase = 0xC0DE0000;
constexpr uint32_t kEchoSlotPoisonWord = 0xBADEC0DE;
constexpr size_t kEchoSlotWords = kTl1EchoSlotBytes / sizeof(uint32_t);
constexpr uint32_t kEchoRuntimeArgCount = 4;
constexpr uint32_t kEchoCommonArgCount = 4;
static_assert(3 + kEchoRuntimeArgCount + kEchoCommonArgCount <= kEchoSlotWords);

struct EchoArguments {
    uint32_t ct_marker = 0;
    std::array<uint32_t, kEchoRuntimeArgCount> runtime{};
    std::array<uint32_t, kEchoCommonArgCount> common{};
};

EchoArguments make_echo_arguments(uint32_t seed, uint32_t ct_marker) {
    EchoArguments arguments;
    arguments.ct_marker = ct_marker;
    for (uint32_t index = 0; index < kEchoRuntimeArgCount; index++) {
        arguments.runtime[index] = 0x52000000u + seed * 16 + index;
    }
    for (uint32_t index = 0; index < kEchoCommonArgCount; index++) {
        arguments.common[index] = 0xC4000000u + seed * 16 + index;
    }
    return arguments;
}

// Compile-time args are fixed per kernel, so an update keeps the ct_marker of the original.
EchoArguments make_updated_arguments(const EchoArguments& original, uint32_t seed) {
    return make_echo_arguments(seed + 0x100, original.ct_marker);
}

std::vector<uint32_t> expected_echo_words(uint32_t hart, const EchoArguments& arguments) {
    std::vector<uint32_t> words = {kArgsEchoMagic, hart, arguments.ct_marker};
    words.insert(words.end(), arguments.runtime.begin(), arguments.runtime.end());
    words.insert(words.end(), arguments.common.begin(), arguments.common.end());
    return words;
}

uint32_t echo_slot_base_address() {
    return MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
}

// The runtime places a kernel by logical core; this proves that core is CCE `cce_index`'s location.
::testing::AssertionResult check_logical_core_is_cce(
    IDevice* device, const CoreCoord& logical_core, uint32_t cce_index) {
    const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
    const CoreCoord expected_core = translated_dram_core(soc_desc_of(device), cce_index);
    if (virtual_core == expected_core) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << fmt::format(
               "CCE {} logical core ({}, {}) maps to ({}, {}), expected ({}, {})",
               cce_index,
               logical_core.x,
               logical_core.y,
               virtual_core.x,
               virtual_core.y,
               expected_core.x,
               expected_core.y);
}

void poison_echo_slots(IDevice* device, uint32_t cce_index) {
    const std::vector<uint32_t> poison(cce_hart_count() * kEchoSlotWords, kEchoSlotPoisonWord);
    write_cce_tl1(device, cce_index, echo_slot_base_address(), poison);
}

std::vector<uint32_t> read_echo_slots(IDevice* device, uint32_t cce_index) {
    const uint32_t echo_block_bytes = static_cast<uint32_t>(cce_hart_count() * kEchoSlotWords * sizeof(uint32_t));
    return read_cce_tl1(device, cce_index, echo_slot_base_address(), echo_block_bytes);
}

std::vector<uint32_t> echo_slot_words(const std::vector<uint32_t>& echo_block, uint32_t hart) {
    const size_t first_index = static_cast<size_t>(hart) * kEchoSlotWords;
    return std::vector<uint32_t>(
        echo_block.begin() + static_cast<std::ptrdiff_t>(first_index),
        echo_block.begin() + static_cast<std::ptrdiff_t>(first_index + kEchoSlotWords));
}

// Compares the whole 64 B slot: the echo words, then poison in the words the kernel does not write.
::testing::AssertionResult check_echo_slot(
    uint32_t hart, const std::vector<uint32_t>& echo_block, const EchoArguments& arguments) {
    std::vector<uint32_t> expected = expected_echo_words(hart, arguments);
    expected.resize(kEchoSlotWords, kEchoSlotPoisonWord);
    return compare_words(fmt::format("echo slot of hart {}", hart), expected, echo_slot_words(echo_block, hart));
}

::testing::AssertionResult check_echo_slot_untouched(uint32_t hart, const std::vector<uint32_t>& echo_block) {
    const std::vector<uint32_t> poison(kEchoSlotWords, kEchoSlotPoisonWord);
    return compare_words(
        fmt::format("echo slot of hart {} (not launched)", hart), poison, echo_slot_words(echo_block, hart));
}

KernelHandle create_echo_kernel(Program& program, const CoreCoord& logical_core, uint32_t hart, uint32_t ct_marker) {
    return CreateKernel(
        program,
        kEchoKernelPath,
        logical_core,
        DramConfig{
            .processor = static_cast<DataMovementProcessor>(hart),
            .noc = NOC::NOC_0,
            .compile_args = {
                echo_slot_base_address(),
                ct_marker,
                kArgsEchoMagic,
                static_cast<uint32_t>(kTl1EchoSlotBytes),
                kEchoRuntimeArgCount,
                kEchoCommonArgCount}});
}

// First-time set. Common runtime args may be set only once per kernel.
void set_echo_arguments(
    Program& program, KernelHandle kernel, const CoreCoord& logical_core, const EchoArguments& arguments) {
    const std::vector<uint32_t> runtime_args(arguments.runtime.begin(), arguments.runtime.end());
    const std::vector<uint32_t> common_args(arguments.common.begin(), arguments.common.end());
    SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    SetCommonRuntimeArgs(program, kernel, common_args);
}

// Update after launch. Unique args are re-set with the same count; common args are written in place.
void update_echo_arguments(
    Program& program, KernelHandle kernel, const CoreCoord& logical_core, const EchoArguments& arguments) {
    const std::vector<uint32_t> runtime_args(arguments.runtime.begin(), arguments.runtime.end());
    SetRuntimeArgs(program, kernel, logical_core, runtime_args);
    RuntimeArgsData& common_args = GetCommonRuntimeArgs(program, kernel);
    for (size_t index = 0; index < arguments.common.size(); index++) {
        common_args[index] = arguments.common[index];
    }
}

::testing::AssertionResult launch_single_hart_echo(
    IDevice* device, uint32_t cce_index, const CoreCoord& logical_core, uint32_t hart, const EchoArguments& arguments) {
    poison_echo_slots(device, cce_index);
    Program program = CreateProgram();
    const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, arguments.ct_marker);
    set_echo_arguments(program, kernel, logical_core, arguments);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
    return check_echo_slot(hart, read_echo_slots(device, cce_index), arguments);
}

class MimirKernelLoad : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MimirKernelLoad, MinimalKernelExecutes) {
    MIMIR_SKIP_UNLESS_EMU();
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(cluster_soc_desc());
    if (cce_index >= topology.num_cces) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_cces << " CCEs.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    constexpr uint32_t hart = 0;

    Program null_program = CreateProgram();
    CreateKernel(
        null_program,
        kNullKernelPath,
        logical_core,
        DramConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    detail::LaunchProgram(device, null_program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    const EchoArguments arguments = make_echo_arguments(cce_index, kEchoCtMarkerBase + cce_index);
    Program echo_program = CreateProgram();
    const KernelHandle echo_kernel = create_echo_kernel(echo_program, logical_core, hart, arguments.ct_marker);
    set_echo_arguments(echo_program, echo_kernel, logical_core, arguments);
    poison_echo_slots(device, cce_index);
    detail::LaunchProgram(device, echo_program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    EXPECT_TRUE(check_echo_slot(hart, read_echo_slots(device, cce_index), arguments));
}

TEST_P(MimirKernelLoad, EachHartAlone) {
    MIMIR_SKIP_UNLESS_EMU();
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(cluster_soc_desc());
    if (cce_index >= topology.num_cces) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_cces << " CCEs.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        const EchoArguments arguments = make_echo_arguments(hart, kEchoCtMarkerBase + hart);
        Program program = CreateProgram();
        const KernelHandle kernel = create_echo_kernel(program, logical_core, hart, arguments.ct_marker);
        set_echo_arguments(program, kernel, logical_core, arguments);
        for (uint32_t poisoned_cce = 0; poisoned_cce < topology.num_cces; poisoned_cce++) {
            poison_echo_slots(device, poisoned_cce);
        }
        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
        for (uint32_t read_cce = 0; read_cce < topology.num_cces; read_cce++) {
            const std::vector<uint32_t> echo_block = read_echo_slots(device, read_cce);
            for (uint32_t slot_hart = 0; slot_hart < cce_hart_count(); slot_hart++) {
                if (read_cce == cce_index && slot_hart == hart) {
                    EXPECT_TRUE(check_echo_slot(hart, echo_block, arguments)) << "CCE " << read_cce;
                } else {
                    EXPECT_TRUE(check_echo_slot_untouched(slot_hart, echo_block)) << "CCE " << read_cce;
                }
            }
        }
    }
}

TEST_P(MimirKernelLoad, ArgsReplaceStale) {
    MIMIR_SKIP_UNLESS_EMU();
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(cluster_soc_desc());
    if (cce_index >= topology.num_cces) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_cces << " CCEs.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    std::vector<EchoArguments> first_arguments;
    std::vector<KernelHandle> kernels;
    Program program = CreateProgram();
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        first_arguments.push_back(make_echo_arguments(hart, kEchoCtMarkerBase + hart));
        kernels.push_back(create_echo_kernel(program, logical_core, hart, first_arguments[hart].ct_marker));
        set_echo_arguments(program, kernels[hart], logical_core, first_arguments[hart]);
    }

    poison_echo_slots(device, cce_index);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
    const std::vector<uint32_t> first_block = read_echo_slots(device, cce_index);
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        EXPECT_TRUE(check_echo_slot(hart, first_block, first_arguments[hart]));
    }

    std::vector<EchoArguments> updated_arguments;
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        updated_arguments.push_back(make_updated_arguments(first_arguments[hart], hart));
        update_echo_arguments(program, kernels[hart], logical_core, updated_arguments[hart]);
    }

    poison_echo_slots(device, cce_index);
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
    const std::vector<uint32_t> updated_block = read_echo_slots(device, cce_index);
    for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
        EXPECT_TRUE(check_echo_slot(hart, updated_block, updated_arguments[hart]));
    }
}

// CloseDevice then CreateDevice in one process reloads and resets nothing: MetalContext::initialize returns early
// on unchanged parameters (metal_context.cpp:203-205) and CCE harts are reset only at MetalContext teardown. So the
// launch after the reopen runs on the firmware loaded by the first open. Reset and reload on every open is
// MimirFwReinit (TT_METAL_FORCE_REINIT).
TEST_P(MimirKernelLoad, FirmwareKeepsRunningAcrossReopen) {
    MIMIR_SKIP_UNLESS_EMU();
    const uint32_t cce_index = GetParam();
    const MimirTopology topology = topology_from(cluster_soc_desc());
    if (cce_index >= topology.num_cces) {
        GTEST_SKIP() << "Descriptor exposes " << topology.num_cces << " CCEs.";
    }

    constexpr uint32_t hart = 0;
    constexpr uint32_t launches_per_session = 4;
    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord logical_core = logical_cce_core(soc_desc_of(device), cce_index);
    ASSERT_TRUE(check_logical_core_is_cce(device, logical_core, cce_index));
    EchoArguments arguments = make_echo_arguments(cce_index, kEchoCtMarkerBase + cce_index);
    uint32_t sequence = 0;
    for (; sequence < launches_per_session; sequence++) {
        arguments.runtime[0] = sequence;
        EXPECT_TRUE(launch_single_hart_echo(device, cce_index, logical_core, hart, arguments));
    }

    device_guard.close();
    device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice reopened_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    arguments.runtime[0] = sequence;
    EXPECT_TRUE(launch_single_hart_echo(device, cce_index, logical_core, hart, arguments));
}

INSTANTIATE_TEST_SUITE_P(MM3, MimirKernelLoad, ::testing::Range(0u, kMaxCces), cce_param_name);

TEST(MimirKernelLoad, SameBinaryAllCcesUnicast) {
    MIMIR_SKIP_UNLESS_EMU();
    const MimirTopology topology = topology_from(cluster_soc_desc());
    if (topology.num_cces == 0) {
        GTEST_SKIP() << "Descriptor exposes no CCEs.";
    }

    IDevice* device = CreateDevice(0);
    ASSERT_NE(device, nullptr);
    ScopedDevice device_guard{device};
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const metal_SocDescriptor& soc = soc_desc_of(device);
    const uint32_t num_cces = num_cces_from_soc(soc);
    std::vector<std::vector<EchoArguments>> arguments_per_cce;
    std::vector<CoreCoord> logical_cores;
    Program program = CreateProgram();
    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        logical_cores.push_back(logical_cce_core(soc, cce_index));
        ASSERT_TRUE(check_logical_core_is_cce(device, logical_cores[cce_index], cce_index));
        arguments_per_cce.emplace_back();
        for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
            // Same compile-time args on every hart of every CCE; runtime and common args differ per hart.
            arguments_per_cce[cce_index].push_back(
                make_echo_arguments(cce_index * cce_hart_count() + hart, kEchoCtMarkerBase));
            const KernelHandle kernel = create_echo_kernel(program, logical_cores[cce_index], hart, kEchoCtMarkerBase);
            set_echo_arguments(program, kernel, logical_cores[cce_index], arguments_per_cce[cce_index][hart]);
        }
    }

    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        poison_echo_slots(device, cce_index);
    }
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);

    for (uint32_t cce_index = 0; cce_index < num_cces; cce_index++) {
        const std::vector<uint32_t> echo_block = read_echo_slots(device, cce_index);
        for (uint32_t hart = 0; hart < cce_hart_count(); hart++) {
            EXPECT_TRUE(check_echo_slot(hart, echo_block, arguments_per_cce[cce_index][hart])) << "CCE " << cce_index;
        }
    }
}

}  // namespace tt::tt_metal::mimir_test::kernel_load
