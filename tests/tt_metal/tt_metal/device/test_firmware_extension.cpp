// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runs a reload walk under Metal's own BRISC firmware extension (reload_extension/runtime_reload.h),
// which uses only internal::fw_ext. Three images, each a different kernel binary and runtime arg, are
// captured configure-only, staged in a holder core's L1, and walked from one launch.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/dispatch_context.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "reload_extension/reload_test_table.h"

namespace tt::tt_metal {

namespace {

constexpr const char* LOG_KERNEL = "tests/tt_metal/tt_metal/test_kernels/misc/reload_log.cpp";
constexpr const char* EXTENSION_HEADER = "tests/tt_metal/tt_metal/device/reload_extension/runtime_reload.h";
constexpr uint32_t NUM_IMAGES = 3;
constexpr uint32_t LOG_CAPACITY = 1024;
constexpr uint32_t L1_ALIGN = 16;

uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

std::vector<uint32_t> to_words(const void* data, size_t bytes) {
    std::vector<uint32_t> words(round_up(bytes, sizeof(uint32_t)) / sizeof(uint32_t), 0);
    std::memcpy(words.data(), data, bytes);
    return words;
}

uint32_t log_entry(uint32_t image, uint32_t core_index) { return (image << 16) | (0x10 * (image + 1) + core_index); }

class FirmwareExtensionFixture : public MeshDeviceFixture {
protected:
    void SetUp() override {
        auto& rtoptions = MetalContext::instance().rtoptions();
        if (MetalContext::instance().get_cluster().arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "the BRISC firmware extension is Blackhole only";
        }
        previous_variant_ = rtoptions.get_brisc_firmware_variant();
        previous_header_ = rtoptions.get_brisc_firmware_header();
        previous_disable_precompiled_ = rtoptions.get_disable_precompiled_fw();
        rtoptions.set_brisc_firmware_variant(
            llrt::BriscFirmwareVariant::Blaze, (std::filesystem::path(rtoptions.get_root_dir()) / EXTENSION_HEADER).string());
        restore_ = true;
        MeshDeviceFixture::SetUp();
    }

    void TearDown() override {
        MeshDeviceFixture::TearDown();
        if (restore_) {
            auto& rtoptions = MetalContext::instance().rtoptions();
            rtoptions.set_brisc_firmware_variant(previous_variant_, previous_header_);
            rtoptions.set_disable_precompiled_fw(previous_disable_precompiled_);
        }
    }

    struct WalkOptions {
        bool persistent = false;
        uint32_t num_rounds = 0;
        NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
        // Every kernel ends with a non-posted multicast it does not wait on, so acks can still be arriving
        // when the extension runs.
        bool late_ack = false;
        // Control for late_ack: the same multicast, acked before the kernel returns.
        bool ack_barrier = false;
        // Every kernel starts with a write and its barrier, which sees counters the previous image left.
        bool probe = false;
    };

    // Captures the images, stages them on `holder_`, writes every core's table and launches image 0
    // without waiting. Returns image 0's program, which the caller waits on.
    Program start_walk(const WalkOptions& options) {
        mesh_ = devices_.at(0);
        device_ = mesh_->impl().get_devices().at(0);
        const CoreRangeSet core_set(std::vector<CoreRange>{CoreRange(cores_.front(), cores_.back())});

        // No buffers are allocated, so L1 above the allocator base is free on every core.
        const auto base = static_cast<uint32_t>(device_->allocator()->get_base_allocator_addr(HalMemType::L1));
        log_addr_ = base;
        table_addr_ = base + 0x2000;
        const uint32_t scratch_addr = base + 0x2400;
        const uint32_t arena_addr = base + 0x3000;

        // The late-ack multicast covers every core from row 2 down, far from the walking cores in row 0.
        const CoreCoord grid = device_->compute_with_storage_grid_size();
        const CoreCoord mcast_start = device_->worker_core_from_logical_core({0, 2});
        const CoreCoord mcast_end = device_->worker_core_from_logical_core({grid.x - 1, grid.y - 1});
        const uint32_t mcast_dests = grid.x * (grid.y - 2);
        constexpr uint32_t mcast_bytes = 16 * 1024;
        const uint32_t mcast_addr = base + 0x10000;

        std::vector<Program> programs;
        for (uint32_t image = 0; image < NUM_IMAGES; ++image) {
            Program program = CreateProgram(*mesh_);
            std::map<std::string, std::string> defines{{"IMAGE_ID", std::to_string(image)}};
            if (options.late_ack) {
                defines["LATE_ACK_MCAST"] = "1";
            }
            if (options.ack_barrier) {
                defines["LATE_ACK_BARRIER"] = "1";
            }
            if (options.probe) {
                defines["LATE_ACK_PROBE"] = "1";
            }
            KernelHandle kernel = CreateKernel(
                program,
                LOG_KERNEL,
                core_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .noc_mode = options.noc_mode,
                    .defines = defines});
            for (uint32_t c = 0; c < cores_.size(); ++c) {
                SetRuntimeArgs(
                    program,
                    kernel,
                    cores_[c],
                    {log_addr_,
                     0x10 * (image + 1) + c,
                     LOG_CAPACITY,
                     static_cast<uint32_t>(mcast_start.x),
                     static_cast<uint32_t>(mcast_start.y),
                     static_cast<uint32_t>(mcast_end.x),
                     static_cast<uint32_t>(mcast_end.y),
                     mcast_addr,
                     mcast_bytes,
                     mcast_dests});
            }
            programs.push_back(std::move(program));
        }
        programs[0].impl().set_reload_table(table_addr_, core_set);

        // Capture each image configure-only, image 0 last so it is still resident for the real launch.
        struct Captured {
            std::vector<uint32_t> block;
            std::vector<uint8_t> config;
        };
        std::vector<std::vector<Captured>> captured(cores_.size(), std::vector<Captured>(NUM_IMAGES));
        for (uint32_t i = 0; i < NUM_IMAGES; ++i) {
            const uint32_t image = (i + 1) % NUM_IMAGES;
            experimental::ConfigureProgramWithoutLaunch(device_, programs[image]);
            for (uint32_t c = 0; c < cores_.size(); ++c) {
                const auto cfg = experimental::CaptureKernelConfig(device_, cores_[c]);
                EXPECT_EQ(cfg.kernel_config_base() % L1_ALIGN, 0u);
                detail::ReadFromDeviceL1(
                    device_,
                    cores_[c],
                    cfg.kernel_config_base(),
                    round_up(cfg.kernel_config_size(), L1_ALIGN),
                    captured[c][image].block);
                captured[c][image].config = cfg.launch_kernel_config();
            }
        }

        // Stage every (core, image) on the holder and write each core's table and an empty log.
        const CoreCoord holder_virtual = device_->worker_core_from_logical_core(holder_);
        const CoreCoord first_virtual = device_->worker_core_from_logical_core(cores_.front());
        const CoreCoord last_virtual = device_->worker_core_from_logical_core(cores_.back());
        uint32_t arena = arena_addr;
        for (uint32_t c = 0; c < cores_.size(); ++c) {
            reload_test_table_t table{};
            table.num_images = NUM_IMAGES;
            table.num_rounds = options.num_rounds;
            table.persistent = options.persistent;
            table.src_x = holder_virtual.x;
            table.src_y = holder_virtual.y;
            table.scratch_addr = scratch_addr;
            table.is_coord = c == 0;
            table.num_cores = cores_.size();
            table.coord_x = first_virtual.x;
            table.coord_y = first_virtual.y;
            table.mcast_start_x = first_virtual.x;
            table.mcast_start_y = first_virtual.y;
            table.mcast_end_x = last_virtual.x;
            table.mcast_end_y = last_virtual.y;
            table.mcast_dests = cores_.size() - 1;
            for (uint32_t image = 0; image < NUM_IMAGES; ++image) {
                auto& cap = captured[c][image];
                auto& entry = table.images[image];
                entry.block_src = arena;
                entry.block_bytes = cap.block.size() * sizeof(uint32_t);
                detail::WriteToDeviceL1(device_, holder_, arena, cap.block);
                arena += round_up(entry.block_bytes, L1_ALIGN);
                entry.config_src = arena;
                entry.config_bytes = cap.config.size();
                auto config_words = to_words(cap.config.data(), cap.config.size());
                detail::WriteToDeviceL1(device_, holder_, arena, config_words);
                arena += round_up(entry.config_bytes, L1_ALIGN);
            }
            auto table_words = to_words(&table, sizeof(table));
            detail::WriteToDeviceL1(device_, cores_[c], table_addr_, table_words);
            std::vector<uint32_t> empty_log(1 + LOG_CAPACITY, 0);
            detail::WriteToDeviceL1(device_, cores_[c], log_addr_, empty_log);
        }
        MetalContext::instance().get_cluster().l1_barrier(device_->id());

        slow_dispatch::LaunchProgramAsync(*device_, programs[0], /*force_slow_dispatch=*/false);
        return std::move(programs[0]);
    }

    // Polls until every core has passed its final barrier; on timeout prints where each core is and exits,
    // since a hung walk cannot be torn down.
    void wait_for_stop(uint32_t seconds) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        const uint32_t phase_addr = table_addr_ + offsetof(reload_test_table_t, phase);
        for (;;) {
            bool stopped = true;
            for (const auto& core : cores_) {
                stopped = stopped && read_word(core, phase_addr) == RELOAD_TEST_PHASE_STOPPED;
            }
            if (stopped) {
                return;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                for (const auto& core : cores_) {
                    std::cerr << "core " << core.str() << ": runs " << read_word(core, log_addr_) << ", installs "
                              << read_word(core, table_addr_ + offsetof(reload_test_table_t, installs)) << ", phase "
                              << read_word(core, phase_addr) << std::endl;
                }
                std::cerr << "walk hung; reset the device (tt-smi -r) before the next run" << std::endl;
                std::_Exit(2);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    uint32_t read_word(const CoreCoord& core, uint32_t addr) {
        std::vector<uint32_t> words;
        detail::ReadFromDeviceL1(device_, core, addr, sizeof(uint32_t), words);
        return words.at(0);
    }

    // Every core ran `runs` images, in walk order, with each image's own binary and runtime arg.
    void expect_walk(uint32_t runs) {
        for (uint32_t c = 0; c < cores_.size(); ++c) {
            std::vector<uint32_t> log;
            const uint32_t written = std::min(runs, LOG_CAPACITY);
            detail::ReadFromDeviceL1(device_, cores_[c], log_addr_, (1 + written) * sizeof(uint32_t), log);
            ASSERT_EQ(log[0], runs) << "core " << cores_[c].str();
            for (uint32_t i = 0; i < written; ++i) {
                EXPECT_EQ(log[1 + i], log_entry(i % NUM_IMAGES, c)) << "core " << cores_[c].str() << " run " << i;
            }
            EXPECT_EQ(read_word(cores_[c], table_addr_ + offsetof(reload_test_table_t, installs)), runs - 1)
                << "core " << cores_[c].str();
        }
    }

    const std::vector<CoreCoord> cores_{{0, 0}, {1, 0}};
    const CoreCoord holder_{2, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_;
    IDevice* device_ = nullptr;
    uint32_t log_addr_ = 0;
    uint32_t table_addr_ = 0;

private:
    llrt::BriscFirmwareVariant previous_variant_{};
    std::string previous_header_;
    bool previous_disable_precompiled_ = false;
    bool restore_ = false;
};

}  // namespace

TEST_F(FirmwareExtensionFixture, TensixWalksImagesFromOneLaunch) {
    constexpr uint32_t num_rounds = 2;
    Program program = start_walk({.persistent = false, .num_rounds = num_rounds});
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

// The kernel-end contract allows a kernel to return with nonposted write acks still arriving. The walk must
// survive acks that land after quiesce_core_noc().
TEST_F(FirmwareExtensionFixture, TensixWalkSurvivesLateWriteAcksDedicated) {
    constexpr uint32_t num_rounds = 20;
    Program program = start_walk({.num_rounds = num_rounds, .late_ack = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

TEST_F(FirmwareExtensionFixture, TensixWalkWithAckedMulticastControlDedicated) {
    constexpr uint32_t num_rounds = 20;
    Program program = start_walk({.num_rounds = num_rounds, .late_ack = true, .ack_barrier = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

TEST_F(FirmwareExtensionFixture, TensixWalkWithAckedMulticastControlDynamic) {
    constexpr uint32_t num_rounds = 20;
    Program program = start_walk(
        {.num_rounds = num_rounds, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC, .late_ack = true, .ack_barrier = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

// The next image's kernel syncs its counters from the hardware at start (dedicated mode, brisck/ncrisck)
// or BRISC does at the top of each pass (dynamic mode). Acks the previous image left in flight must not
// be lost there.
TEST_F(FirmwareExtensionFixture, TensixNextKernelBarrierAfterLateAcksDedicated) {
    constexpr uint32_t num_rounds = 20;
    Program program = start_walk({.num_rounds = num_rounds, .late_ack = true, .probe = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

TEST_F(FirmwareExtensionFixture, TensixNextKernelBarrierAfterLateAcksDynamic) {
    constexpr uint32_t num_rounds = 20;
    Program program =
        start_walk({.num_rounds = num_rounds, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC, .late_ack = true, .probe = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

TEST_F(FirmwareExtensionFixture, TensixWalkSurvivesLateWriteAcksDynamic) {
    constexpr uint32_t num_rounds = 20;
    Program program =
        start_walk({.num_rounds = num_rounds, .noc_mode = NOC_MODE::DM_DYNAMIC_NOC, .late_ack = true});
    wait_for_stop(60);
    slow_dispatch::WaitProgramDone(*device_, program);
    expect_walk(NUM_IMAGES * num_rounds);
}

// The coordinator reads a stop word the host writes mid-walk (fw_ext::load_fresh), and every core stops at
// the same round boundary.
TEST_F(FirmwareExtensionFixture, TensixPersistentWalkStopsOnHostWord) {
    Program program = start_walk({.persistent = true});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    uint32_t seen = 0;
    while ((seen = read_word(cores_.front(), log_addr_)) < 2 * NUM_IMAGES) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "the walk did not advance; " << seen << " runs";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::vector<uint32_t> stop{1};
    detail::WriteToDeviceL1(device_, cores_.front(), table_addr_ + offsetof(reload_test_table_t, stop), stop);
    slow_dispatch::WaitProgramDone(*device_, program);

    const uint32_t runs = read_word(cores_.front(), log_addr_);
    EXPECT_GE(runs, seen);
    EXPECT_EQ(runs % NUM_IMAGES, 0u) << "the walk stopped mid-round";
    expect_walk(runs);
}

}  // namespace tt::tt_metal
