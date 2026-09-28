// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end kernel scratchpads on WH/BH silicon under slow dispatch.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/integration_tests/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecHWTest;

// ============================================================================
// Kernel Scratchpad: write / readback under slow dispatch
// ============================================================================

// The slow-dispatch counterpart of scratchpad_fast_dispatch.cpp's fast-dispatch ScratchpadWriteReadback. A
// scratchpad's framework-allocated L1 base address rides a common runtime arg; on the slow-dispatch
// LaunchProgram path it reaches the kernel because ConfigureDeviceWithProgram — which allocates the
// scratchpad and patches its base address into the CRTA buffer — now runs BEFORE WriteRuntimeArgsToDevice
// commits the CRTAs to the device. This fixture (ProgramSpecHWTest, on MeshDeviceFixture,
// TT_METAL_SLOW_DISPATCH_MODE=1) exercises exactly that path.
//
// One Gen1 DM kernel binds a 64-byte scratchpad, writes a known pattern into it, and reports its
// Scratchpad::get_base_address() to a host-known L1 address. The host reads the reported base, then
// reads the scratchpad L1 and confirms the pattern landed — closing the loop on both "the scratchpad is
// real, writable, node-local L1" and "the framework delivered its base address to the kernel".
TEST_F(ProgramSpecHWTest, ScratchpadWriteReadback) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t kScratchpadBytes = 64;                            // 16 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kReportAddr = 100 * 1024;                         // host-known fixed L1 addr
    constexpr uint32_t kPatternBase = 0xC0DE0000u;                       // must match the kernel

    const NodeCoord node{0, 0};

    KernelSpec dm_kernel{
        .unique_id = KernelSpecName{"scratch_kernel"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_write_pattern.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"report_addr"},
            },
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::NOC_0,
                    },
            },
    };
    dm_kernel.scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    ProgramSpec spec;
    spec.name = "scratchpad_write_readback_slow_dispatch";
    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}};
    spec.work_units = std::vector<WorkUnitSpec>{WorkUnitSpec{
        .name = "work_unit_0",
        .kernels = {KernelSpecName{"scratch_kernel"}},
        .target_nodes = node,
    }};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"scratch_kernel"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"report_addr", kReportAddr}}),
    }};
    SetProgramRunArgs(program, params);

    // Pre-zero the report location so a kernel that never wrote it would be caught (the readback base
    // address would be 0, which is not a valid scratchpad L1 address → the pattern check fails).
    std::vector<uint32_t> zero_report(1, 0u);
    slow_dispatch::WriteToL1(*mesh_device, node, kReportAddr, zero_report);

    // Dispatch via the slow-dispatch path (blocking — wait_until_cores_done defaults to true).
    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(*mesh_device, node, kReportAddr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    const uint32_t scratch_base = reported[0];
    EXPECT_NE(scratch_base, 0u) << "Kernel reported a 0 scratchpad base address (token not delivered?)";

    std::vector<uint32_t> scratch_contents;
    slow_dispatch::ReadFromL1(*mesh_device, node, scratch_base, kScratchpadBytes, scratch_contents);
    ASSERT_EQ(scratch_contents.size(), kNumElems);

    std::vector<uint32_t> expected(kNumElems);
    for (uint32_t i = 0; i < kNumElems; i++) {
        expected[i] = kPatternBase + i;
    }
    EXPECT_EQ(scratch_contents, expected) << "Scratchpad L1 at reported base 0x" << std::hex << scratch_base
                                          << " did not contain the pattern the kernel wrote";
}

// ============================================================================
// Kernel Scratchpad: base re-delivered after a DFB resize between enqueues (regression)
// ============================================================================

// A scratchpad's framework-allocated L1 base address rides a common-runtime-arg (CRTA) word, patched
// in by ProgramImpl::allocate_scratchpads. Scratchpads stack on top of the DFB allocators, so when a
// DFB size override (SetProgramRunArgs's dfb_run_overrides) re-lays-out the L1 region between
// enqueues, the scratchpad's base address MOVES — and the moved address must be re-patched into the
// CRTA before the next dispatch, or the kernel reads a stale base that now points into the grown DFB.
//
// The mechanism that makes this correct is one line: invalidate_dataflow_buffer_allocation() (which a
// size override triggers) clears the scratchpads_allocated_ latch UNCONDITIONALLY, so the next launch
// re-runs allocate_scratchpads — recomputing the base from the new layout and re-patching the CRTA —
// before WriteRuntimeArgsToDevice commits it. This test pins that line: a junior engineer who drops
// or mis-orders the latch reset would silently ship the stale base, and nothing else catches it.
//
// Setup: a BRISC producer binds a scratchpad AND a DFB (producer). It writes a known pattern into the
// scratchpad and stages the scratchpad's get_base_address() into the DFB; an NCRISC consumer drains
// that entry to DRAM. Both on node {0,0}, so the DFB lives on the producer's core and the scratchpad
// stacks above it. The producer has no named args — its only per-enqueue state is the
// framework-supplied scratchpad CRTA — so it is omitted from kernel_run_args (the binding-only second
// pass in SetProgramRunArgs installs its CRTA).
//
// Sequence (slow dispatch): launch with the small DFB → base_A; SetProgramRunArgs again with a
// dfb_run_overrides that GROWS the DFB → launch → base_B. The grown DFB pushes the stacked scratchpad
// up, so a correctly re-delivered base satisfies base_B != base_A, and the pattern must still land at
// base_B (the scratchpad is real, writable L1 at its new home). With the latch reset removed,
// allocate_scratchpads does not re-run: the CRTA keeps base_A, the kernel reports base_A
// (base_B == base_A → the NE check fails) and its pattern write lands inside the grown DFB's region.
TEST_F(ProgramSpecHWTest, ScratchpadBaseReDeliveredAfterDfbResize) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t entry_size = 1024;      // bytes per DFB entry (constant)
    constexpr uint32_t num_entries_small = 2;  // initial DFB depth
    constexpr uint32_t num_entries_large = 8;  // grown depth → +6 KB region → scratchpad moves
    constexpr uint32_t kScratchpadBytes = 64;  // 16 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kPatternBase = 0xC0DE0000u;                       // must match the kernel

    const NodeCoord node{0, 0};

    // Output buffer holds one DFB entry (single page → single bank).
    auto output_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = entry_size},
        {.page_size = entry_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto& cq = mesh_device->mesh_command_queue();

    ProgramSpec spec;
    spec.name = "scratchpad_base_redelivered_after_dfb_resize";

    // Producer (BRISC): write a pattern into the scratchpad, then stage its base address into the DFB.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t n = pad.size();
    for (uint32_t i = 0; i < n; i++) {
        pad[i] = 0xC0DE0000u + i;
    }
    DataflowBuffer buf(dfb::stage);
    buf.reserve_back(1);
    volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)buf.get_write_ptr();
    w[0] = pad.get_base_address();
    buf.push_back(1);
}
)"};
    producer.scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    // Consumer (NCRISC): drain the staged entry to DRAM.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    auto dst_addr = get_arg(args::dst_addr);
    auto bank_id = get_arg(args::bank_id);
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;
    DataflowBuffer buf(dfb::stage);
    buf.wait_front(1);
    noc.async_write(buf, dram_dst, buf.get_entry_size(), {}, {.bank_id = bank_id, .addr = dst_addr});
    noc.async_write_barrier();
    buf.pop_front(1);
}
)"};
    consumer.runtime_arg_schema.runtime_arg_names = {"dst_addr", "bank_id"};

    auto dfb = MakeMinimalDFB("stage", entry_size, num_entries_small);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"stage"}, "stage"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"stage"}, "stage"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, MakeProgramFromSpec(*mesh_device, spec));
    Program& program = workload.get_programs().at(device_range);

    std::vector<uint32_t> expected_pattern(kNumElems);
    for (uint32_t i = 0; i < kNumElems; i++) {
        expected_pattern[i] = kPatternBase + i;
    }

    // Set the consumer's RTAs (+ optional DFB num_entries override), launch (blocking), and return the
    // scratchpad base the kernel reported — verifying the pattern landed at that base.
    auto launch_and_read_base = [&](uint32_t dfb_num_entries) -> uint32_t {
        ProgramRunArgs params;
        params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"dst_addr", output_buffer->address()}, {"bank_id", 0u}}),
        }};
        params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"stage"}, .num_entries = dfb_num_entries});
        SetProgramRunArgs(program, params);

        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, output_buffer, /*blocking=*/true);
        EXPECT_FALSE(out.empty());
        const uint32_t base = out.empty() ? 0u : out[0];
        EXPECT_NE(base, 0u) << "Kernel reported a 0 scratchpad base address (token not delivered?)";

        // The scratchpad must be real, writable L1 at the reported base.
        std::vector<uint32_t> scratch_contents;
        slow_dispatch::ReadFromL1(*mesh_device, node, base, kScratchpadBytes, scratch_contents);
        EXPECT_EQ(scratch_contents, expected_pattern)
            << "Scratchpad L1 at reported base 0x" << std::hex << base << " did not contain the pattern";

        // The scratchpad must not physically alias the (possibly just-resized) DFB — the core
        // invariant the feature protects. This is a pure host-side range check against the DFB's
        // current allocated extent, so it catches a stale OR miscomputed base *deterministically*,
        // independent of how much of the DFB is actually written on any run. (The pattern check
        // above cannot: it writes and reads at the SAME reported base, so it round-trips even when
        // that base is stale — the classic "passes until DFB occupancy grows into it" trap.)
        auto stage_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("stage"));
        const uint32_t dfb_lo = stage_dfb->uniform_alloc_addr();
        const uint32_t dfb_hi = dfb_lo + stage_dfb->total_size();
        EXPECT_TRUE(base + kScratchpadBytes <= dfb_lo || base >= dfb_hi)
            << "Scratchpad [0x" << std::hex << base << ", +0x" << kScratchpadBytes << ") aliases DFB 'stage' [0x"
            << dfb_lo << ", 0x" << dfb_hi << ") — stale or miscomputed scratchpad base (address not re-delivered "
            << "after the DFB resize, or stacking math wrong).";
        return base;
    };

    // Enqueue #1: small DFB.
    const uint32_t base_a = launch_and_read_base(num_entries_small);
    // Enqueue #2: grow the DFB. The stacked scratchpad must relocate AND its new base be re-delivered.
    const uint32_t base_b = launch_and_read_base(num_entries_large);

    EXPECT_NE(base_a, base_b)
        << "Scratchpad base did not change after the DFB grew (both 0x" << std::hex << base_a
        << "). Either the resize did not relocate the scratchpad (allocator change — enlarge the growth delta) or the "
           "moved base was not re-delivered to the kernel (allocate_scratchpads did not re-run — check the "
           "scratchpads_allocated_ latch reset in invalidate_dataflow_buffer_allocation).";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
