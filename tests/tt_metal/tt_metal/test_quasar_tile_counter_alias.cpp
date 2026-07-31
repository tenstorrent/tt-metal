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

// Pack and unpack both use TC16 for push / wait / pop. The remapper also routes TC16 -> TC17 as a
// sacrificial shadow so the HW update copy does not alias into overlay 0-15. TC17 is never waited on,
// popped, or checked by this test.
constexpr std::uint32_t kProducerTileCounter = 16;
constexpr std::uint32_t kMaxSteps = 4;

// Must match the scratch layout in
// tests/tt_metal/tt_metal/test_kernels/compute/tile_counter_overlay_alias.cpp.
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kProducerPostedBaseIdx = 1;
constexpr std::uint32_t kProducerAckedBaseIdx = kProducerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kProducerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kProducerCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kProducerCapacityIdx + 1;

constexpr std::uint32_t kOverlay0BaselineCapIdx = kNumStepsIdx + 1;
constexpr std::uint32_t kOverlay0BaselinePostedIdx = kOverlay0BaselineCapIdx + 1;
constexpr std::uint32_t kOverlay0BaselineAckedIdx = kOverlay0BaselinePostedIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapCapIdx = kOverlay0BaselineAckedIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapPostedIdx = kOverlay0AfterCapCapIdx + 1;
constexpr std::uint32_t kOverlay0AfterCapAckedIdx = kOverlay0AfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlay0AfterPushPostedBaseIdx = kOverlay0AfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlay0AfterPopAckedBaseIdx = kOverlay0AfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay0FinalCapIdx = kOverlay0AfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay0FinalPostedIdx = kOverlay0FinalCapIdx + 1;
constexpr std::uint32_t kOverlay0FinalAckedIdx = kOverlay0FinalPostedIdx + 1;

constexpr std::uint32_t kOverlay1BaselineCapIdx = kOverlay0FinalAckedIdx + 1;
constexpr std::uint32_t kOverlay1BaselinePostedIdx = kOverlay1BaselineCapIdx + 1;
constexpr std::uint32_t kOverlay1BaselineAckedIdx = kOverlay1BaselinePostedIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapCapIdx = kOverlay1BaselineAckedIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapPostedIdx = kOverlay1AfterCapCapIdx + 1;
constexpr std::uint32_t kOverlay1AfterCapAckedIdx = kOverlay1AfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlay1AfterPushPostedBaseIdx = kOverlay1AfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlay1AfterPopAckedBaseIdx = kOverlay1AfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay1FinalCapIdx = kOverlay1AfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlay1FinalPostedIdx = kOverlay1FinalCapIdx + 1;
constexpr std::uint32_t kOverlay1FinalAckedIdx = kOverlay1FinalPostedIdx + 1;
constexpr std::uint32_t kScratchWords = kOverlay1FinalAckedIdx + 1;

// Distinguishes "kernel wrote 0" from "kernel never wrote this word". Only the two handshake words start at
// zero, because the kernel polls them.
constexpr std::uint32_t kUnwritten = 0xFFFFFFFFu;

// Written by unpack when credits never show up on TC16, so a missing push reports here instead of parking
// the kernel in TT_WAIT_TILES.
constexpr std::uint32_t kTimedOut = 0xFFFFFFFEu;

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

// The overlay names 16 counters per interface (0x40 apart, 0x400 per interface); a TRISC sees 32 counters per
// NEO (0x20 apart) at TILE_COUNTERS_BASE. The two decoders cover the same 0x400 window, so a TC16 value can
// surface under an overlay index other than 0. Dump every overlay-visible counter rather than trusting that a
// leak from TC16 must appear at overlay counter 0.
void dump_overlay_counters(IDevice* device, const CoreCoord& logical_core, std::uint32_t num_interfaces) {
    constexpr std::uint32_t interface_stride = TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_SIZE;
    constexpr std::uint32_t counter_stride =
        TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__REG_FILE_SIZE;
    constexpr std::uint32_t counters_per_interface = interface_stride / counter_stride;

    for (std::uint32_t intf = 0; intf < num_interfaces; intf++) {
        for (std::uint32_t counter = 0; counter < counters_per_interface; counter++) {
            const std::uint32_t block = TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_REG_MAP_BASE_ADDR +
                                        intf * interface_stride + counter * counter_stride;
            const std::uint32_t capacity = read_core_register(
                device,
                logical_core,
                block + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_OFFSET);
            const std::uint32_t posted = read_core_register(
                device,
                logical_core,
                block + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_OFFSET);
            const std::uint32_t acked = read_core_register(
                device,
                logical_core,
                block + TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_OFFSET);

            const bool all_zero = (capacity | posted | acked) == 0;
            if (all_zero && !(intf == 0 && counter == 0)) {
                continue;
            }
            std::cout << "  overlay intf" << intf << " counter" << counter << " @0x" << std::hex << block << std::dec
                      << ": capacity=" << capacity << " posted=" << posted << " acked=" << acked << std::endl;
        }
    }
}

std::uint32_t count_steps(const CreditCase& credit_case) {
    std::uint32_t num_steps = 0;
    while (num_steps < kMaxSteps && credit_case.steps[num_steps] != 0) {
        num_steps++;
    }
    return num_steps;
}

void expect_overlay_tc_untouched(
    const std::vector<std::uint32_t>& scratch,
    std::uint32_t num_steps,
    std::uint32_t tc16_capacity,
    std::uint32_t overlay_tc,
    std::uint32_t baseline_cap_idx,
    std::uint32_t baseline_posted_idx,
    std::uint32_t baseline_acked_idx,
    std::uint32_t after_cap_cap_idx,
    std::uint32_t after_cap_posted_idx,
    std::uint32_t after_cap_acked_idx,
    std::uint32_t after_push_posted_base_idx,
    std::uint32_t after_pop_acked_base_idx,
    std::uint32_t final_cap_idx,
    std::uint32_t final_posted_idx,
    std::uint32_t final_acked_idx,
    const char* phase) {
    const std::uint32_t baseline_cap = scratch[baseline_cap_idx];
    const std::uint32_t baseline_posted = scratch[baseline_posted_idx];
    const std::uint32_t baseline_acked = scratch[baseline_acked_idx];

    // Host resets overlay TC0/TC1 but does not program capacity. Programming TC16 must not change this
    // baseline — a leaked capacity write shows up as after_cap becoming tc16_capacity.
    EXPECT_EQ(scratch[after_cap_cap_idx], baseline_cap)
        << phase << ": programming TC16 capacity=" << tc16_capacity << " changed overlay TC" << overlay_tc
        << " capacity from " << baseline_cap << " to " << scratch[after_cap_cap_idx];
    if (baseline_cap != tc16_capacity) {
        EXPECT_NE(scratch[after_cap_cap_idx], tc16_capacity)
            << phase << ": overlay TC" << overlay_tc << " capacity was set to the TC16 capacity that was just "
            << "programmed";
    }
    EXPECT_EQ(scratch[after_cap_posted_idx], baseline_posted)
        << phase << ": TC16 capacity programming changed overlay TC" << overlay_tc << " posted";
    EXPECT_EQ(scratch[after_cap_acked_idx], baseline_acked)
        << phase << ": TC16 capacity programming changed overlay TC" << overlay_tc << " acked";

    for (std::uint32_t step = 0; step < num_steps; step++) {
        EXPECT_EQ(scratch[after_push_posted_base_idx + step], baseline_posted)
            << phase << ": TC16 push at step " << step << " leaked onto overlay TC" << overlay_tc << " posted";
        EXPECT_EQ(scratch[after_pop_acked_base_idx + step], baseline_acked)
            << phase << ": TC16 pop at step " << step << " leaked onto overlay TC" << overlay_tc << " acked";
    }
    for (std::uint32_t step = num_steps; step < kMaxSteps; step++) {
        EXPECT_EQ(scratch[after_push_posted_base_idx + step], kUnwritten)
            << phase << ": pack wrote overlay TC" << overlay_tc << " past its step count";
        EXPECT_EQ(scratch[after_pop_acked_base_idx + step], kUnwritten)
            << phase << ": unpack wrote overlay TC" << overlay_tc << " past its step count";
    }

    EXPECT_EQ(scratch[final_cap_idx], baseline_cap)
        << phase << ": overlay TC" << overlay_tc << " capacity changed by the end of the TC16 push/pop series";
    EXPECT_EQ(scratch[final_posted_idx], baseline_posted)
        << phase << ": overlay TC" << overlay_tc << " posted changed by the end of the TC16 series";
    EXPECT_EQ(scratch[final_acked_idx], baseline_acked)
        << phase << ": overlay TC" << overlay_tc << " acked changed by the end of the TC16 series";
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

    // Reset overlay TC0 and TC1 — do not program capacity. The kernel snapshots both before any TC16 op
    // and again after TC16 capacity / push / pop; those samples must match, and neither capacity must
    // become the TC16 capacity that pack programs.
    write_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR, 1);
    write_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__RESET_REG_ADDR, 1);
    EXPECT_EQ(
        read_core_register(
            device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR),
        0u)
        << "overlay TC0 posted was not 0 after reset";
    EXPECT_EQ(
        read_core_register(
            device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__READ_POSTED_REG_ADDR),
        0u)
        << "overlay TC1 posted was not 0 after reset";

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
    const auto neo_tiles_available = [&](std::uint32_t tc) {
        return read_core_register(
            device,
            logical_core,
            hal.get_neo_tile_counters_base_addr() + tc * hal.get_neo_tile_counters_size() +
                hal.get_neo_tile_counters_tiles_available_offset());
    };
    const std::uint32_t producer_tiles_available = neo_tiles_available(kProducerTileCounter);
    // Match the kernel's overlay samples: raw credit counters, not the derived availability views.
    const std::uint32_t overlay_tc0_capacity = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__BUFFER_CAPACITY_REG_ADDR);
    const std::uint32_t overlay_tc0_posted = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);
    const std::uint32_t overlay_tc0_acked = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_ACKED_REG_ADDR);
    const std::uint32_t overlay_tc1_capacity = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__BUFFER_CAPACITY_REG_ADDR);
    const std::uint32_t overlay_tc1_posted = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__READ_POSTED_REG_ADDR);
    const std::uint32_t overlay_tc1_acked = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__READ_ACKED_REG_ADDR);

    std::cout << "[" << credit_case.name << "] capacity=" << credit_case.capacity << " total_tiles=" << total_tiles
              << " scratch @L1 0x" << std::hex << l1_address << std::dec << ":" << std::endl;
    for (std::uint32_t i = 0; i < kScratchWords; i++) {
        std::string label;
        if (i == kReadyIdx) {
            label = "pack: TC16 reset + capacity programmed";
        } else if (i >= kProducerPostedBaseIdx && i < kProducerPostedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kProducerPostedBaseIdx;
            label = "pack: TC16 posted after push " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step);
        } else if (i >= kProducerAckedBaseIdx && i < kProducerAckedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kProducerAckedBaseIdx;
            label = "unpack: TC16 acked after pop " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step);
        } else if (i == kPushesDoneIdx) {
            label = "pack: all pushes issued";
        } else if (i == kProducerCapacityIdx) {
            label = "pack: TC16 buf_capacity read back";
        } else if (i == kNumStepsIdx) {
            label = "pack: step count from runtime args";
        } else if (i == kOverlay0BaselineCapIdx) {
            label = "pack: overlay TC0 capacity before TC16 ops";
        } else if (i == kOverlay0BaselinePostedIdx) {
            label = "pack: overlay TC0 posted before TC16 ops";
        } else if (i == kOverlay0BaselineAckedIdx) {
            label = "pack: overlay TC0 acked before TC16 ops";
        } else if (i == kOverlay0AfterCapCapIdx) {
            label = "pack: overlay TC0 capacity after TC16 capacity programmed";
        } else if (i == kOverlay0AfterCapPostedIdx) {
            label = "pack: overlay TC0 posted after TC16 capacity programmed";
        } else if (i == kOverlay0AfterCapAckedIdx) {
            label = "pack: overlay TC0 acked after TC16 capacity programmed";
        } else if (i >= kOverlay0AfterPushPostedBaseIdx && i < kOverlay0AfterPushPostedBaseIdx + kMaxSteps) {
            label =
                "pack: overlay TC0 posted after TC16 push step " + std::to_string(i - kOverlay0AfterPushPostedBaseIdx);
        } else if (i >= kOverlay0AfterPopAckedBaseIdx && i < kOverlay0AfterPopAckedBaseIdx + kMaxSteps) {
            label =
                "unpack: overlay TC0 acked after TC16 pop step " + std::to_string(i - kOverlay0AfterPopAckedBaseIdx);
        } else if (i == kOverlay0FinalCapIdx) {
            label = "unpack: overlay TC0 capacity after last pop";
        } else if (i == kOverlay0FinalPostedIdx) {
            label = "unpack: overlay TC0 posted after last pop";
        } else if (i == kOverlay0FinalAckedIdx) {
            label = "unpack: overlay TC0 acked after last pop";
        } else if (i == kOverlay1BaselineCapIdx) {
            label = "pack: overlay TC1 capacity before TC16 ops";
        } else if (i == kOverlay1BaselinePostedIdx) {
            label = "pack: overlay TC1 posted before TC16 ops";
        } else if (i == kOverlay1BaselineAckedIdx) {
            label = "pack: overlay TC1 acked before TC16 ops";
        } else if (i == kOverlay1AfterCapCapIdx) {
            label = "pack: overlay TC1 capacity after TC16 capacity programmed";
        } else if (i == kOverlay1AfterCapPostedIdx) {
            label = "pack: overlay TC1 posted after TC16 capacity programmed";
        } else if (i == kOverlay1AfterCapAckedIdx) {
            label = "pack: overlay TC1 acked after TC16 capacity programmed";
        } else if (i >= kOverlay1AfterPushPostedBaseIdx && i < kOverlay1AfterPushPostedBaseIdx + kMaxSteps) {
            label =
                "pack: overlay TC1 posted after TC16 push step " + std::to_string(i - kOverlay1AfterPushPostedBaseIdx);
        } else if (i >= kOverlay1AfterPopAckedBaseIdx && i < kOverlay1AfterPopAckedBaseIdx + kMaxSteps) {
            label =
                "unpack: overlay TC1 acked after TC16 pop step " + std::to_string(i - kOverlay1AfterPopAckedBaseIdx);
        } else if (i == kOverlay1FinalCapIdx) {
            label = "unpack: overlay TC1 capacity after last pop";
        } else if (i == kOverlay1FinalPostedIdx) {
            label = "unpack: overlay TC1 posted after last pop";
        } else if (i == kOverlay1FinalAckedIdx) {
            label = "unpack: overlay TC1 acked after last pop";
        }
        std::cout << "  scratch[" << i << "] = " << scratch[i] << "  (" << label << ")" << std::endl;
    }
    std::cout << "  T6 tiles_available: TC16=" << producer_tiles_available
              << ", overlay TC0 capacity/posted/acked=" << overlay_tc0_capacity << "/" << overlay_tc0_posted << "/"
              << overlay_tc0_acked << ", overlay TC1 capacity/posted/acked=" << overlay_tc1_capacity << "/"
              << overlay_tc1_posted << "/" << overlay_tc1_acked << std::endl;
    dump_overlay_counters(device, logical_core, 2);

    // Non-fatal so one mismatch still runs the rest of this case (and the remaining credit cases).
    EXPECT_EQ(scratch[kReadyIdx], 1u) << "pack never published the ready flag";
    EXPECT_EQ(scratch[kPushesDoneIdx], 1u) << "pack never completed its push series";
    EXPECT_EQ(scratch[kNumStepsIdx], num_steps) << "kernel decoded a different step count than the host passed";
    EXPECT_EQ(scratch[kProducerCapacityIdx], credit_case.capacity)
        << "TC16 buf_capacity did not read back as programmed";

    // Pack issues every push before any pop, so the running push total is the expected occupancy on TC16.
    std::uint32_t pushed_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        pushed_so_far += credit_case.steps[step];
        EXPECT_EQ(scratch[kProducerPostedBaseIdx + step], pushed_so_far)
            << "TC16 posted is not the running push total at step " << step;
    }
    EXPECT_EQ(pushed_so_far, total_tiles);

    // f.acked is the SPACE_AVAILABLE view on the counter being popped (TC16).
    const std::uint32_t producer_capacity = scratch[kProducerCapacityIdx];
    std::uint32_t popped_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        popped_so_far += credit_case.steps[step];
        EXPECT_NE(scratch[kProducerAckedBaseIdx + step], kTimedOut)
            << "unpack timed out at step " << step << ": credits never showed up on TC16";
        if (producer_capacity >= total_tiles) {
            const std::uint32_t expected_space = producer_capacity - (total_tiles - popped_so_far);
            EXPECT_EQ(scratch[kProducerAckedBaseIdx + step], expected_space)
                << "TC16 space_available does not match capacity - unconsumed at step " << step
                << " (each pop freeing 2x means the credit was counted twice)";
        }
    }
    EXPECT_EQ(popped_so_far, total_tiles);

    // Unused step slots must stay untouched, which catches a kernel that walks past its step count.
    for (std::uint32_t step = num_steps; step < kMaxSteps; step++) {
        EXPECT_EQ(scratch[kProducerPostedBaseIdx + step], kUnwritten) << "pack pushed past its step count";
        EXPECT_EQ(scratch[kProducerAckedBaseIdx + step], kUnwritten) << "unpack popped past its step count";
    }

    EXPECT_EQ(producer_tiles_available, 0u) << "after equal push/pop, TC16 tiles_available should be 0";

    // Kernel samples overlay TC0/TC1 through the NEO-local mirror; host reads the overlay interface. Both
    // must agree and must stay at the pre-TC16 baseline.
    expect_overlay_tc_untouched(
        scratch,
        num_steps,
        credit_case.capacity,
        /*overlay_tc=*/0,
        kOverlay0BaselineCapIdx,
        kOverlay0BaselinePostedIdx,
        kOverlay0BaselineAckedIdx,
        kOverlay0AfterCapCapIdx,
        kOverlay0AfterCapPostedIdx,
        kOverlay0AfterCapAckedIdx,
        kOverlay0AfterPushPostedBaseIdx,
        kOverlay0AfterPopAckedBaseIdx,
        kOverlay0FinalCapIdx,
        kOverlay0FinalPostedIdx,
        kOverlay0FinalAckedIdx,
        credit_case.name);
    expect_overlay_tc_untouched(
        scratch,
        num_steps,
        credit_case.capacity,
        /*overlay_tc=*/1,
        kOverlay1BaselineCapIdx,
        kOverlay1BaselinePostedIdx,
        kOverlay1BaselineAckedIdx,
        kOverlay1AfterCapCapIdx,
        kOverlay1AfterCapPostedIdx,
        kOverlay1AfterCapAckedIdx,
        kOverlay1AfterPushPostedBaseIdx,
        kOverlay1AfterPopAckedBaseIdx,
        kOverlay1FinalCapIdx,
        kOverlay1FinalPostedIdx,
        kOverlay1FinalAckedIdx,
        credit_case.name);

    EXPECT_EQ(overlay_tc0_capacity, scratch[kOverlay0BaselineCapIdx])
        << "overlay interface reads a different TC0 capacity than the TRISC mirror (TC16 capacity leaked)";
    EXPECT_EQ(overlay_tc0_posted, scratch[kOverlay0BaselinePostedIdx])
        << "overlay interface reads a different TC0 posted than the TRISC mirror (TC16 push leaked)";
    EXPECT_EQ(overlay_tc0_acked, scratch[kOverlay0BaselineAckedIdx])
        << "overlay interface reads a different TC0 acked than the TRISC mirror (TC16 pop leaked)";
    if (scratch[kOverlay0BaselineCapIdx] != credit_case.capacity) {
        EXPECT_NE(overlay_tc0_capacity, credit_case.capacity)
            << "host readout: overlay TC0 capacity equals the TC16 capacity that was programmed";
    }

    EXPECT_EQ(overlay_tc1_capacity, scratch[kOverlay1BaselineCapIdx])
        << "overlay interface reads a different TC1 capacity than the TRISC mirror (TC16 capacity leaked)";
    EXPECT_EQ(overlay_tc1_posted, scratch[kOverlay1BaselinePostedIdx])
        << "overlay interface reads a different TC1 posted than the TRISC mirror (TC16 push leaked)";
    EXPECT_EQ(overlay_tc1_acked, scratch[kOverlay1BaselineAckedIdx])
        << "overlay interface reads a different TC1 acked than the TRISC mirror (TC16 pop leaked)";
    if (scratch[kOverlay1BaselineCapIdx] != credit_case.capacity) {
        EXPECT_NE(overlay_tc1_capacity, credit_case.capacity)
            << "host readout: overlay TC1 capacity equals the TC16 capacity that was programmed";
    }
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
        log_info(tt::LogTest, "Finished running credit case");
        break;
    }
}

}  // namespace
}  // namespace tt::tt_metal
