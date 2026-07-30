// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "internal/tt-2xx/quasar/overlay/meta/registers/overlay_reg_defines_core.h"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t kTensixOnlyTileCounter = 16;
constexpr std::uint32_t kMaxSteps = 4;

// Must match the scratch layout in
// tests/tt_metal/tt_metal/test_kernels/compute/tile_counter_overlay_alias.cpp.
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kTilesAvailableBaseIdx = 1;
constexpr std::uint32_t kSpaceAvailableBaseIdx = kTilesAvailableBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kSpaceAvailableBaseIdx + kMaxSteps;
constexpr std::uint32_t kCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kCapacityIdx + 1;
constexpr std::uint32_t kScratchWords = kNumStepsIdx + 1;

// Distinguishes "kernel wrote 0" from "kernel never wrote this word". Only the two handshake words start at
// zero, because the kernel polls them.
constexpr std::uint32_t kUnwritten = 0xFFFFFFFFu;

struct CreditCase {
    const char* name;
    std::uint32_t capacity;
    std::array<std::uint32_t, kMaxSteps> steps;  // a zero ends the series
};

// Pack issues every push before unpack pops, so each case needs sum(steps) <= capacity or WAIT_FREE would
// never be satisfied. Capacities and step sizes are varied so a scaling error in the credit path (a doubled
// push, a doubled pop, or a capacity that lands on the wrong counter) cannot be mistaken for the expected
// pattern of any other case.
constexpr CreditCase kCases[] = {
    {"cap32_mixed_1_2_1_3", 32, {1, 2, 1, 3}},  // 7 of 32
    {"cap32_fill_8x4", 32, {8, 8, 8, 8}},       // exactly fills capacity
    {"cap16_desc_5_4_3_2", 16, {5, 4, 3, 2}},   // 14 of 16
    {"cap8_ones", 8, {1, 1, 1, 1}},             // small capacity
    {"cap5_two_steps", 5, {2, 3, 0, 0}},        // odd capacity, fewer than kMaxSteps
    {"cap1_single_tile", 1, {1, 0, 0, 0}},      // minimum capacity
};

std::uint32_t read_core_register(IDevice* device, const CoreCoord& logical_core, std::uint32_t address) {
    const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
    return MetalContext::instance().get_cluster().read_core(
        device->id(), virtual_core, address, sizeof(std::uint32_t))[0];
}

void write_core_register(IDevice* device, const CoreCoord& logical_core, std::uint32_t address, std::uint32_t value) {
    const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
    MetalContext::instance().get_cluster().write_core_immediate(
        &value, sizeof(value), tt_cxy_pair(device->id(), virtual_core), address);
}

std::uint32_t count_steps(const CreditCase& credit_case) {
    std::uint32_t num_steps = 0;
    while (num_steps < kMaxSteps && credit_case.steps[num_steps] != 0) {
        num_steps++;
    }
    return num_steps;
}

void RunCreditCase(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& logical_core,
    const CreditCase& credit_case) {
    SCOPED_TRACE(credit_case.name);

    const std::uint32_t num_steps = count_steps(credit_case);
    const std::uint32_t total_tiles =
        std::accumulate(credit_case.steps.begin(), credit_case.steps.begin() + num_steps, 0u);
    ASSERT_GT(num_steps, 0u) << "case has no steps";
    ASSERT_LE(total_tiles, credit_case.capacity)
        << "pack pushes everything before unpack pops, so sum(steps) must fit in capacity";

    // TC16 is Tensix-only, so updating it must not change any of the 16 overlay counters.
    write_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR, 1);
    ASSERT_EQ(
        read_core_register(
            device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR),
        0u);

    const experimental::NodeCoord node{0, 0};
    const std::uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    std::vector<std::uint32_t> scratch_init(kScratchWords, kUnwritten);
    scratch_init[kReadyIdx] = 0;
    scratch_init[kPushesDoneIdx] = 0;
    tt_metal::detail::WriteToDeviceL1(device, node, l1_address, scratch_init);

    const experimental::KernelSpecName kernel_name{"tile_counter_overlay_alias"};
    experimental::KernelSpec kernel{
        .unique_id = kernel_name,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/tile_counter_overlay_alias.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"l1_address", "capacity", "step0", "step1", "step2", "step3"},
            },
        .hw_config = experimental::ComputeHardwareConfig{},
    };
    experimental::ProgramSpec spec{
        .name = "tile_counter_overlay_alias",
        .kernels = {kernel},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {kernel_name},
            .target_nodes = node,
        }},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = kernel_name,
        .runtime_arg_values =
            {{node,
              {{"l1_address", l1_address},
               {"capacity", credit_case.capacity},
               {"step0", credit_case.steps[0]},
               {"step1", credit_case.steps[1]},
               {"step2", credit_case.steps[2]},
               {"step3", credit_case.steps[3]}}}},
    }};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, true);

    std::vector<std::uint32_t> scratch(kScratchWords, 0);
    tt_metal::detail::ReadFromDeviceL1(device, node, l1_address, kScratchWords * sizeof(std::uint32_t), scratch);

    const Hal& hal = MetalContext::instance().hal();
    const std::uint32_t tensix_tc16_tiles_available_addr = hal.get_neo_tile_counters_base_addr() +
                                                           kTensixOnlyTileCounter * hal.get_neo_tile_counters_size() +
                                                           hal.get_neo_tile_counters_tiles_available_offset();
    const std::uint32_t tensix_tc16_tiles_available =
        read_core_register(device, logical_core, tensix_tc16_tiles_available_addr);
    const std::uint32_t overlay_tc0_posted = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);

    std::cout << "[" << credit_case.name << "] capacity=" << credit_case.capacity << " total_tiles=" << total_tiles
              << " scratch @L1 0x" << std::hex << l1_address << std::dec << ":" << std::endl;
    for (std::uint32_t i = 0; i < kScratchWords; i++) {
        std::string label;
        if (i == kReadyIdx) {
            label = "pack: tc16 reset + capacity programmed";
        } else if (i >= kTilesAvailableBaseIdx && i < kTilesAvailableBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kTilesAvailableBaseIdx;
            label = "pack: tiles_available after push " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step);
        } else if (i >= kSpaceAvailableBaseIdx && i < kSpaceAvailableBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kSpaceAvailableBaseIdx;
            label = "unpack: space_available after pop " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step);
        } else if (i == kPushesDoneIdx) {
            label = "pack: all pushes issued";
        } else if (i == kCapacityIdx) {
            label = "pack: buf_capacity read back";
        } else if (i == kNumStepsIdx) {
            label = "pack: step count from runtime args";
        }
        std::cout << "  scratch[" << i << "] = " << scratch[i] << "  (" << label << ")" << std::endl;
    }
    std::cout << "  T6 TC16 tiles_available=" << tensix_tc16_tiles_available
              << ", overlay TC0 posted=" << overlay_tc0_posted << std::endl;

    ASSERT_EQ(scratch[kReadyIdx], 1u) << "pack never published the ready flag";
    ASSERT_EQ(scratch[kPushesDoneIdx], 1u) << "pack never completed its push series";
    ASSERT_EQ(scratch[kNumStepsIdx], num_steps) << "kernel decoded a different step count than the host passed";
    ASSERT_EQ(scratch[kCapacityIdx], credit_case.capacity) << "tc16 buf_capacity did not read back as programmed";

    // The kernel reads tile_counters[].f.posted, which is the TILES_AVAILABLE view, and issues every push
    // before any pop. So tiles available tracks the running total of pushed tiles; a doubled credit shows up
    // as 2x the expected value.
    std::uint32_t pushed_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        pushed_so_far += credit_case.steps[step];
        EXPECT_EQ(scratch[kTilesAvailableBaseIdx + step], pushed_so_far)
            << "pack tiles_available is not the running push total at step " << step << " (2x means each push "
            << "was counted twice)";
    }
    ASSERT_EQ(pushed_so_far, total_tiles);

    // f.acked is the SPACE_AVAILABLE view, so after each pop the free space is the capacity minus whatever
    // is still unconsumed.
    std::uint32_t popped_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        popped_so_far += credit_case.steps[step];
        const std::uint32_t expected_space = credit_case.capacity - (total_tiles - popped_so_far);
        EXPECT_EQ(scratch[kSpaceAvailableBaseIdx + step], expected_space)
            << "unpack space_available does not match capacity - unconsumed at step " << step << " (each pop "
            << "freeing 2x means the credit was counted twice)";
    }
    ASSERT_EQ(popped_so_far, total_tiles);

    // Unused step slots must stay untouched, which catches a kernel that walks past its step count.
    for (std::uint32_t step = num_steps; step < kMaxSteps; step++) {
        EXPECT_EQ(scratch[kTilesAvailableBaseIdx + step], kUnwritten) << "pack pushed past its step count";
        EXPECT_EQ(scratch[kSpaceAvailableBaseIdx + step], kUnwritten) << "unpack popped past its step count";
    }

    // Equal push/pop drained the counter.
    EXPECT_EQ(tensix_tc16_tiles_available, 0u) << "after equal push/pop, tiles_available should be 0 (posted-acked)";
    EXPECT_EQ(overlay_tc0_posted, 0u)
        << "T6 TC16 incorrectly reached overlay TC0 (the 5-bit selector was truncated to 4 bits)";
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixTileCounter16DoesNotAliasOverlayTileCounter0) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This low-level RTL regression test only runs in simulation or emulation.";
    }

    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];
    constexpr CoreCoord logical_core{0, 0};

    for (const auto& credit_case : kCases) {
        RunCreditCase(*mesh_device, device, logical_core, credit_case);
    }
}

}  // namespace
}  // namespace tt::tt_metal
