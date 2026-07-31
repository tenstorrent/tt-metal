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

// Pack posts to the producer counter; the remapper routes the credit to the consumer counter that unpack
// waits on and pops. Two counters rather than a self-route, which would land both the native update and the
// routed copy on one counter and double the credit.
constexpr std::uint32_t kProducerTileCounter = 16;
constexpr std::uint32_t kConsumerTileCounter = 17;
constexpr std::uint32_t kMaxSteps = 4;

// Must match the scratch layout in
// tests/tt_metal/tt_metal/test_kernels/compute/tile_counter_overlay_alias.cpp.
constexpr std::uint32_t kReadyIdx = 0;
constexpr std::uint32_t kProducerPostedBaseIdx = 1;
constexpr std::uint32_t kConsumerAckedBaseIdx = kProducerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kPushesDoneIdx = kConsumerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kProducerCapacityIdx = kPushesDoneIdx + 1;
constexpr std::uint32_t kNumStepsIdx = kProducerCapacityIdx + 1;
constexpr std::uint32_t kConsumerPostedBaseIdx = kNumStepsIdx + 1;
constexpr std::uint32_t kProducerAckedBaseIdx = kConsumerPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kConsumerCapacityIdx = kProducerAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayBaselineCapIdx = kConsumerCapacityIdx + 1;
constexpr std::uint32_t kOverlayBaselinePostedIdx = kOverlayBaselineCapIdx + 1;
constexpr std::uint32_t kOverlayBaselineAckedIdx = kOverlayBaselinePostedIdx + 1;
constexpr std::uint32_t kOverlayAfterCapCapIdx = kOverlayBaselineAckedIdx + 1;
constexpr std::uint32_t kOverlayAfterCapPostedIdx = kOverlayAfterCapCapIdx + 1;
constexpr std::uint32_t kOverlayAfterCapAckedIdx = kOverlayAfterCapPostedIdx + 1;
constexpr std::uint32_t kOverlayAfterPushPostedBaseIdx = kOverlayAfterCapAckedIdx + 1;
constexpr std::uint32_t kOverlayAfterPopAckedBaseIdx = kOverlayAfterPushPostedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayFinalCapIdx = kOverlayAfterPopAckedBaseIdx + kMaxSteps;
constexpr std::uint32_t kOverlayFinalPostedIdx = kOverlayFinalCapIdx + 1;
constexpr std::uint32_t kOverlayFinalAckedIdx = kOverlayFinalPostedIdx + 1;
constexpr std::uint32_t kScratchWords = kOverlayFinalAckedIdx + 1;

// Distinguishes "kernel wrote 0" from "kernel never wrote this word". Only the two handshake words start at
// zero, because the kernel polls them.
constexpr std::uint32_t kUnwritten = 0xFFFFFFFFu;

// Written by unpack when the routed credits never reach the consumer counter, so a dead route reports here
// instead of parking the kernel in TT_WAIT_TILES.
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

void expect_overlay_tc0_untouched(
    const std::vector<std::uint32_t>& scratch,
    std::uint32_t num_steps,
    std::uint32_t tc16_capacity,
    const char* phase) {
    const std::uint32_t baseline_cap = scratch[kOverlayBaselineCapIdx];
    const std::uint32_t baseline_posted = scratch[kOverlayBaselinePostedIdx];
    const std::uint32_t baseline_acked = scratch[kOverlayBaselineAckedIdx];

    // Host only resets overlay TC0; it does not program capacity. Programming TC16 must not change this
    // baseline — a leaked capacity write shows up as after_cap becoming tc16_capacity.
    EXPECT_EQ(scratch[kOverlayAfterCapCapIdx], baseline_cap)
        << phase << ": programming TC16 capacity=" << tc16_capacity << " changed overlay TC0 capacity from "
        << baseline_cap << " to " << scratch[kOverlayAfterCapCapIdx];
    if (baseline_cap != tc16_capacity) {
        EXPECT_NE(scratch[kOverlayAfterCapCapIdx], tc16_capacity)
            << phase << ": overlay TC0 capacity was set to the TC16 capacity that was just programmed";
    }
    EXPECT_EQ(scratch[kOverlayAfterCapPostedIdx], baseline_posted)
        << phase << ": TC16 capacity programming changed overlay TC0 posted";
    EXPECT_EQ(scratch[kOverlayAfterCapAckedIdx], baseline_acked)
        << phase << ": TC16 capacity programming changed overlay TC0 acked";

    for (std::uint32_t step = 0; step < num_steps; step++) {
        EXPECT_EQ(scratch[kOverlayAfterPushPostedBaseIdx + step], baseline_posted)
            << phase << ": TC16 push at step " << step << " leaked onto overlay TC0 posted";
        EXPECT_EQ(scratch[kOverlayAfterPopAckedBaseIdx + step], baseline_acked)
            << phase << ": TC16 pop at step " << step << " leaked onto overlay TC0 acked";
    }
    for (std::uint32_t step = num_steps; step < kMaxSteps; step++) {
        EXPECT_EQ(scratch[kOverlayAfterPushPostedBaseIdx + step], kUnwritten)
            << phase << ": pack wrote overlay TC0 past its step count";
        EXPECT_EQ(scratch[kOverlayAfterPopAckedBaseIdx + step], kUnwritten)
            << phase << ": unpack wrote overlay TC0 past its step count";
    }

    EXPECT_EQ(scratch[kOverlayFinalCapIdx], baseline_cap)
        << phase << ": overlay TC0 capacity changed by the end of the TC16 push/pop series";
    EXPECT_EQ(scratch[kOverlayFinalPostedIdx], baseline_posted)
        << phase << ": overlay TC0 posted changed by the end of the TC16 series";
    EXPECT_EQ(scratch[kOverlayFinalAckedIdx], baseline_acked)
        << phase << ": overlay TC0 acked changed by the end of the TC16 series";
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

    // Reset overlay TC0 only — do not program its capacity. The kernel snapshots TC0 before any TC16 op
    // and again after TC16 capacity / push / pop; those samples must match, and TC0 capacity must never
    // become the TC16 capacity that pack programs.
    write_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR, 1);
    EXPECT_EQ(
        read_core_register(
            device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR),
        0u)
        << "overlay TC0 posted was not 0 after reset";

    // Routing TC16 -> TC17 could move the aliasing rather than remove it, so snapshot overlay counter 1 (the
    // index TC17 would shadow) before the run. It is not reset, because it may be in use, so the check is
    // that it does not change.
    const std::uint32_t overlay_tc1_capacity_before = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__BUFFER_CAPACITY_REG_ADDR);
    const std::uint32_t overlay_tc1_posted_before = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__READ_POSTED_REG_ADDR);
    const std::uint32_t overlay_tc1_acked_before = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_1__READ_ACKED_REG_ADDR);

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
    const std::uint32_t consumer_tiles_available = neo_tiles_available(kConsumerTileCounter);
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
        } else if (i >= kConsumerAckedBaseIdx && i < kConsumerAckedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kConsumerAckedBaseIdx;
            label = "unpack: TC17 acked after pop " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step);
        } else if (i == kPushesDoneIdx) {
            label = "pack: all pushes issued";
        } else if (i == kProducerCapacityIdx) {
            label = "pack: TC16 buf_capacity read back";
        } else if (i == kNumStepsIdx) {
            label = "pack: step count from runtime args";
        } else if (i >= kConsumerPostedBaseIdx && i < kConsumerPostedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kConsumerPostedBaseIdx;
            label = "pack: TC17 posted after push " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step) + " (routed credit)";
        } else if (i >= kProducerAckedBaseIdx && i < kProducerAckedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kProducerAckedBaseIdx;
            label = "unpack: TC16 acked after pop " + std::to_string(credit_case.steps[step]) + ", step " +
                    std::to_string(step) + " (reverse route)";
        } else if (i == kConsumerCapacityIdx) {
            label = "unpack: TC17 buf_capacity (routed from TC16's write, nothing programs it directly)";
        } else if (i == kOverlayBaselineCapIdx) {
            label = "pack: overlay TC0 capacity before TC16 ops";
        } else if (i == kOverlayBaselinePostedIdx) {
            label = "pack: overlay TC0 posted before TC16 ops";
        } else if (i == kOverlayBaselineAckedIdx) {
            label = "pack: overlay TC0 acked before TC16 ops";
        } else if (i == kOverlayAfterCapCapIdx) {
            label = "pack: overlay TC0 capacity after TC16 capacity programmed";
        } else if (i == kOverlayAfterCapPostedIdx) {
            label = "pack: overlay TC0 posted after TC16 capacity programmed";
        } else if (i == kOverlayAfterCapAckedIdx) {
            label = "pack: overlay TC0 acked after TC16 capacity programmed";
        } else if (i >= kOverlayAfterPushPostedBaseIdx && i < kOverlayAfterPushPostedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kOverlayAfterPushPostedBaseIdx;
            label = "pack: overlay TC0 posted after TC16 push step " + std::to_string(step);
        } else if (i >= kOverlayAfterPopAckedBaseIdx && i < kOverlayAfterPopAckedBaseIdx + kMaxSteps) {
            const std::uint32_t step = i - kOverlayAfterPopAckedBaseIdx;
            label = "unpack: overlay TC0 acked after TC16 pop step " + std::to_string(step);
        } else if (i == kOverlayFinalCapIdx) {
            label = "unpack: overlay TC0 capacity after last pop";
        } else if (i == kOverlayFinalPostedIdx) {
            label = "unpack: overlay TC0 posted after last pop";
        } else if (i == kOverlayFinalAckedIdx) {
            label = "unpack: overlay TC0 acked after last pop";
        }
        std::cout << "  scratch[" << i << "] = " << scratch[i] << "  (" << label << ")" << std::endl;
    }
    std::cout << "  T6 tiles_available: TC16=" << producer_tiles_available << " TC17=" << consumer_tiles_available
              << ", overlay TC0 capacity/posted/acked=" << overlay_tc0_capacity << "/" << overlay_tc0_posted << "/"
              << overlay_tc0_acked << std::endl;
    dump_overlay_counters(device, logical_core, 2);

    // Non-fatal so one mismatch still runs the rest of this case (and the remaining credit cases).
    EXPECT_EQ(scratch[kReadyIdx], 1u) << "pack never published the ready flag";
    EXPECT_EQ(scratch[kPushesDoneIdx], 1u) << "pack never completed its push series";
    EXPECT_EQ(scratch[kNumStepsIdx], num_steps) << "kernel decoded a different step count than the host passed";
    EXPECT_EQ(scratch[kProducerCapacityIdx], credit_case.capacity)
        << "TC16 buf_capacity did not read back as programmed";
    // No RISC writes TC17's capacity: pack programs TC16 and the route configures the far end, so the same
    // capacity has to show up on TC17.
    EXPECT_EQ(scratch[kConsumerCapacityIdx], credit_case.capacity)
        << "TC17 buf_capacity is not the capacity programmed on TC16; the route did not carry the capacity "
        << "write to the counter it delivers credits to";

    // Pack issues every push before any pop, so the running push total is the expected occupancy. The route
    // copies the credit onto TC17 rather than moving it: the push is a native T6 update of TC16, so both
    // counters must sit at exactly 1x the running total. A 0 means the event went missing on that counter and
    // a 2x means it was counted twice there, which is the aliasing/duplication this case is looking for.
    std::uint32_t pushed_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        pushed_so_far += credit_case.steps[step];
        EXPECT_EQ(scratch[kProducerPostedBaseIdx + step], pushed_so_far)
            << "TC16 posted is not the running push total at step " << step << "; pack pushed on TC16, so the "
            << "credit must appear there whether or not the route also delivers a copy to TC17";
        EXPECT_EQ(scratch[kConsumerPostedBaseIdx + step], pushed_so_far)
            << "TC17 posted is not the running push total at step " << step << "; the routed copy either did not "
            << "arrive (0) or was counted twice (2x)";
    }
    EXPECT_EQ(pushed_so_far, total_tiles);

    // f.acked is the SPACE_AVAILABLE view, so after each pop the free space on the counter being popped is
    // the capacity minus whatever is still unconsumed.
    // Free space is measured against TC17's own capacity, so use what the kernel observed rather than what was
    // programmed on TC16: if the capacity did not propagate, the assertion above names that as the one failure
    // instead of every step reporting a mismatch against a capacity TC17 never had.
    const std::uint32_t consumer_capacity = scratch[kConsumerCapacityIdx];
    std::uint32_t popped_so_far = 0;
    for (std::uint32_t step = 0; step < num_steps; step++) {
        popped_so_far += credit_case.steps[step];
        EXPECT_NE(scratch[kConsumerAckedBaseIdx + step], kTimedOut)
            << "unpack timed out at step " << step << ": the routed credits never reached TC17";
        if (consumer_capacity >= total_tiles) {
            const std::uint32_t expected_space = consumer_capacity - (total_tiles - popped_so_far);
            EXPECT_EQ(scratch[kConsumerAckedBaseIdx + step], expected_space)
                << "TC17 space_available does not match capacity - unconsumed at step " << step << " (each pop "
                << "freeing 2x means the credit was counted twice)";
        }
    }
    EXPECT_EQ(popped_so_far, total_tiles);

    // Unused step slots must stay untouched, which catches a kernel that walks past its step count.
    for (std::uint32_t step = num_steps; step < kMaxSteps; step++) {
        EXPECT_EQ(scratch[kProducerPostedBaseIdx + step], kUnwritten) << "pack pushed past its step count";
        EXPECT_EQ(scratch[kConsumerAckedBaseIdx + step], kUnwritten) << "unpack popped past its step count";
    }

    // Equal push/pop drained the counter that was popped. TC16 is not checked here: the pop is a native T6
    // update of TC17, so whether TC16 also drains depends on the route carrying the ack back, which is one of
    // the unknowns this case is measuring rather than asserting.
    EXPECT_EQ(consumer_tiles_available, 0u) << "after equal push/pop, TC17 tiles_available should be 0";

    // The kernel samples TC0 through the NEO-local mirror and the host through the overlay interface. Both
    // name counter 0, so they must agree; a disagreement means the two decoders do not resolve to the same
    // physical counter, which is itself the aliasing being hunted here.
    expect_overlay_tc0_untouched(scratch, num_steps, credit_case.capacity, credit_case.name);
    EXPECT_EQ(overlay_tc0_capacity, scratch[kOverlayBaselineCapIdx])
        << "overlay interface reads a different TC0 capacity than the TRISC mirror (TC16 capacity leaked)";
    EXPECT_EQ(overlay_tc0_posted, scratch[kOverlayBaselinePostedIdx])
        << "overlay interface reads a different TC0 posted than the TRISC mirror (TC16 push leaked)";
    EXPECT_EQ(overlay_tc0_acked, scratch[kOverlayBaselineAckedIdx])
        << "overlay interface reads a different TC0 acked than the TRISC mirror (TC16 pop leaked)";
    if (scratch[kOverlayBaselineCapIdx] != credit_case.capacity) {
        EXPECT_NE(overlay_tc0_capacity, credit_case.capacity)
            << "host readout: overlay TC0 capacity equals the TC16 capacity that was programmed";
    }

    // TC17 is the routed target, so overlay counter 1 is where the aliasing would reappear if the route only
    // shifted it rather than removing it.
    EXPECT_EQ(overlay_tc1_capacity, overlay_tc1_capacity_before)
        << "overlay TC1 capacity changed: the capacity the route put on TC17 aliased down";
    EXPECT_EQ(overlay_tc1_posted, overlay_tc1_posted_before)
        << "overlay TC1 posted changed: the credit routed to TC17 aliased down";
    EXPECT_EQ(overlay_tc1_acked, overlay_tc1_acked_before) << "overlay TC1 acked changed: the ack on TC17 aliased down";
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
