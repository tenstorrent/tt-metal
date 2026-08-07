// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fast-dispatch real-hardware tests for the Metal 2.0 kernel-scratchpad feature (WH/BH).
//
// A kernel scratchpad is a private, blank, node-local L1 region bound to exactly one kernel for the
// program's execution lifetime. These tests use the fast-dispatch UnitMeshCQSingleCardFixture and the
// MeshWorkload enqueue path (mirror of test_single_dm_l1_write.cpp); the slow-dispatch LaunchProgram
// counterpart of the first one is ProgramSpecHWTest.ScratchpadWriteReadback in
// test_program_spec_hw.cpp.
//
// Tests in this file:
//   - ScratchpadWriteReadback: the scratchpad is real, writable L1 and the framework delivered its
//     base address to the kernel.
//   - ScratchpadNocEndpoints: the scratchpad works as either endpoint of a NoC transaction.
//
// Both rely on the same host-visibility trick: the kernel writes its Scratchpad::get_base_address()
// to a host-known fixed L1 address (passed as a named RTA), which lets the host find the
// framework-allocated region and read it back directly.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "command_queue_fixture.hpp"

namespace tt::tt_metal {
namespace {

// What this proves end-to-end:
//   (a) the scratchpad is real, writable, node-local L1, and
//   (b) the framework delivered the scratchpad's framework-allocated L1 base address to the kernel
//       via the binding token (the CRTA word the Scratchpad ctor reads).
//
// Verification approach (see scratchpad_write_pattern.cpp for the device side):
//   - One DM kernel binds a single 64-byte (16 x uint32_t) scratchpad.
//   - The kernel writes a known pattern {0xC0DE0000 + i} into scratchpad[0..16), then writes its
//     own Scratchpad::get_base_address() into a host-known fixed L1 address (passed as a named RTA).
//   - The host reads that fixed address to learn the scratchpad's allocated base address, then reads
//     64 bytes of L1 at that base address and checks the pattern.
//   This closes the loop on both (a) and (b): the host both learns where the framework put the
//   scratchpad AND confirms the kernel's writes actually landed there. A 0/stale base address, or a
//   non-L1-backed scratchpad, would fail the compare.
TEST_F(UnitMeshCQSingleCardFixture, ScratchpadWriteReadback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Gen1 only (mirrors test_program_spec_hw.cpp's guard). The scratchpad feature works on both
    // gens, but the device-side kernel here uses the Gen1 L1-readback idiom (plain volatile L1
    // write, host-visible after a blocking enqueue).
    if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
    }

    constexpr uint32_t kScratchpadBytes = 64;                            // 16 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kReportAddr = 100 * 1024;                         // host-known fixed L1 addr
    constexpr uint32_t kPatternBase = 0xC0DE0000u;                       // must match the kernel

    const experimental::NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Build ProgramSpec: one Gen1 DM kernel binding one scratchpad.
    // -------------------------------------------------------
    experimental::KernelSpec dm_kernel{
        .unique_id = experimental::KernelSpecName{"scratch_kernel"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_write_pattern.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"report_addr"},
            },
        .hw_config =
            experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
    };
    dm_kernel.scratchpad_bindings.push_back(experimental::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = experimental::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    experimental::ProgramSpec spec{
        .name = "scratchpad_write_readback",
        .kernels = {dm_kernel},
        .scratchpads = {experimental::ScratchpadSpec{
            .unique_id = experimental::ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}},
        .work_units = std::vector<experimental::WorkUnitSpec>{experimental::WorkUnitSpec{
            .name = "work_unit_0",
            .kernels = {experimental::KernelSpecName{"scratch_kernel"}},
            .target_nodes = node,
        }},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // -------------------------------------------------------
    // Runtime args: the host-known report L1 address.
    // -------------------------------------------------------
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = experimental::KernelSpecName{"scratch_kernel"},
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"report_addr", kReportAddr}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    // Pre-zero the report location so a kernel that never wrote it would be caught (the readback
    // base address would be 0, which is not a valid scratchpad L1 address → the pattern check fails).
    std::vector<uint32_t> zero_report(1, 0u);
    detail::WriteToDeviceL1(device, node, kReportAddr, zero_report);

    // -------------------------------------------------------
    // Dispatch via the fast/mesh path (blocking).
    // -------------------------------------------------------
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // -------------------------------------------------------
    // Verify: read the reported scratchpad base address, then read the scratchpad's L1 and compare.
    // -------------------------------------------------------
    std::vector<uint32_t> reported;
    detail::ReadFromDeviceL1(device, node, kReportAddr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    const uint32_t scratch_base = reported[0];
    EXPECT_NE(scratch_base, 0u) << "Kernel reported a 0 scratchpad base address (token not delivered?)";

    std::vector<uint32_t> scratch_contents;
    detail::ReadFromDeviceL1(device, node, scratch_base, kScratchpadBytes, scratch_contents);
    ASSERT_EQ(scratch_contents.size(), kNumElems);

    std::vector<uint32_t> expected(kNumElems);
    for (uint32_t i = 0; i < kNumElems; i++) {
        expected[i] = kPatternBase + i;
    }
    EXPECT_EQ(scratch_contents, expected) << "Scratchpad L1 at reported base 0x" << std::hex << scratch_base
                                          << " did not contain the pattern the kernel wrote";
}

// What this proves end-to-end:
//   (a) noc_traits_t<Scratchpad<T>> instantiates and compiles at all (nothing else in-tree
//       instantiates it, so without this test the specialization is never type-checked), and
//   (b) a Scratchpad resolves to the right local L1 address as both a NoC read destination and a
//       NoC write source, with offset_bytes actually applied.
//
// Verification approach (see scratchpad_noc_roundtrip.cpp for the device side):
//   - The host seeds a known pattern in a REMOTE core's L1 (a real cross-core NoC transaction, not a
//     self-loopback) and zeroes the remote destination.
//   - The kernel zeroes its scratchpad, NoC-reads the pattern into it at offset 32, NoC-writes it
//     back out from offset 32, and reports its base address.
//   - The host checks the pattern round-tripped to the remote destination, AND that the scratchpad's
//     own L1 holds the pattern at [32, 96) with zeros on both sides.
//   The round-trip alone would still pass if BOTH directions silently ignored offset_bytes (or both
//   applied the same wrong one), so the zero-flanks check on the scratchpad is what pins the offset
//   down. The zero suffix additionally catches a transfer that ran long.
TEST_F(UnitMeshCQSingleCardFixture, ScratchpadNocEndpoints) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Gen1 only, for the same reason as ScratchpadWriteReadback above.
    if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
    }

    constexpr uint32_t kScratchpadBytes = 128;                              // 32 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);     // 32
    constexpr uint32_t kOffsetBytes = 32;                                   // must match the kernel
    constexpr uint32_t kTransferBytes = 64;                                 // must match the kernel
    constexpr uint32_t kOffsetElems = kOffsetBytes / sizeof(uint32_t);      // 8
    constexpr uint32_t kTransferElems = kTransferBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kReportAddr = 100 * 1024;                            // host-known fixed L1 addr, kernel's core
    constexpr uint32_t kSrcAddr = 100 * 1024;                               // host-known fixed L1 addr, remote core
    constexpr uint32_t kDstAddr = 100 * 1024 + 256;                         // host-known fixed L1 addr, remote core
    constexpr uint32_t kPatternBase = 0x5EED0000u;                          // distinct from the other test's pattern

    // The scratchpad's L1 base is aligned to the allocator's DRAM alignment (>= 32 bytes on WH/BH),
    // so base + kOffsetBytes clears the 16-byte L1 alignment a NoC transfer requires. kSrcAddr and
    // kDstAddr are 32-byte aligned for the same reason.
    static_assert(kOffsetBytes % 32 == 0);
    static_assert(kTransferBytes % 32 == 0);
    static_assert(kOffsetBytes + kTransferBytes < kScratchpadBytes, "need a non-empty zero suffix");

    const experimental::NodeCoord node{0, 0};
    // The NoC peer: a different core, so this is a real cross-core transaction. It runs no kernel of
    // its own; the host uses its L1 as the source and destination buffer.
    const CoreCoord remote_core{1, 0};
    const CoreCoord remote_virtual_core = mesh_device->worker_core_from_logical_core(remote_core);

    // -------------------------------------------------------
    // Build ProgramSpec: one Gen1 DM kernel binding one scratchpad.
    // -------------------------------------------------------
    experimental::KernelSpec dm_kernel{
        .unique_id = experimental::KernelSpecName{"scratch_noc_kernel"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_noc_roundtrip.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"remote_noc_x", "remote_noc_y", "src_addr", "dst_addr", "report_addr"},
            },
        .hw_config =
            experimental::DataMovementGen1Config{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
    };
    dm_kernel.scratchpad_bindings.push_back(experimental::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = experimental::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    experimental::ProgramSpec spec{
        .name = "scratchpad_noc_endpoints",
        .kernels = {dm_kernel},
        .scratchpads = {experimental::ScratchpadSpec{
            .unique_id = experimental::ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}},
        .work_units = std::vector<experimental::WorkUnitSpec>{experimental::WorkUnitSpec{
            .name = "work_unit_0",
            .kernels = {experimental::KernelSpecName{"scratch_noc_kernel"}},
            .target_nodes = node,
        }},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = experimental::KernelSpecName{"scratch_noc_kernel"},
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"remote_noc_x", static_cast<uint32_t>(remote_virtual_core.x)},
             {"remote_noc_y", static_cast<uint32_t>(remote_virtual_core.y)},
             {"src_addr", kSrcAddr},
             {"dst_addr", kDstAddr},
             {"report_addr", kReportAddr}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    // -------------------------------------------------------
    // Seed the remote core: pattern at kSrcAddr, zeros at kDstAddr.
    // -------------------------------------------------------
    std::vector<uint32_t> source_pattern(kTransferElems);
    for (uint32_t i = 0; i < kTransferElems; i++) {
        source_pattern[i] = kPatternBase + i;
    }
    detail::WriteToDeviceL1(device, remote_core, kSrcAddr, source_pattern);

    std::vector<uint32_t> zeros(kTransferElems, 0u);
    detail::WriteToDeviceL1(device, remote_core, kDstAddr, zeros);

    // Pre-zero the report location, so a kernel that never wrote it reads back as 0 (see below).
    std::vector<uint32_t> zero_report(1, 0u);
    detail::WriteToDeviceL1(device, node, kReportAddr, zero_report);

    // -------------------------------------------------------
    // Dispatch via the fast/mesh path (blocking).
    // -------------------------------------------------------
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // -------------------------------------------------------
    // Verify (1): the pattern round-tripped remote -> scratchpad -> remote.
    // -------------------------------------------------------
    std::vector<uint32_t> round_tripped;
    detail::ReadFromDeviceL1(device, remote_core, kDstAddr, kTransferBytes, round_tripped);
    ASSERT_EQ(round_tripped.size(), kTransferElems);
    EXPECT_EQ(round_tripped, source_pattern)
        << "Pattern did not survive the remote -> scratchpad -> remote NoC round trip";

    // -------------------------------------------------------
    // Verify (2): inside the scratchpad, the data landed at kOffsetBytes and nowhere else.
    // -------------------------------------------------------
    std::vector<uint32_t> reported;
    detail::ReadFromDeviceL1(device, node, kReportAddr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    const uint32_t scratch_base = reported[0];
    ASSERT_NE(scratch_base, 0u) << "Kernel reported a 0 scratchpad base address (token not delivered?)";

    std::vector<uint32_t> scratch_contents;
    detail::ReadFromDeviceL1(device, node, scratch_base, kScratchpadBytes, scratch_contents);
    ASSERT_EQ(scratch_contents.size(), kNumElems);

    std::vector<uint32_t> expected(kNumElems, 0u);
    for (uint32_t i = 0; i < kTransferElems; i++) {
        expected[kOffsetElems + i] = kPatternBase + i;
    }
    EXPECT_EQ(scratch_contents, expected)
        << "Scratchpad L1 at reported base 0x" << std::hex << scratch_base
        << " does not hold the pattern at byte offset " << std::dec << kOffsetBytes
        << " with zeros on both sides (offset_bytes ignored, or the transfer ran long?)";
}

}  // namespace
}  // namespace tt::tt_metal
