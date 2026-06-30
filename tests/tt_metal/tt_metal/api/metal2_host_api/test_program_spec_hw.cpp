// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-hardware tests for Metal 2.0 Host API: ProgramSpec on WH/BH.
//
// These tests require a Wormhole B0 or Blackhole device and slow dispatch mode.
// They prove the ProgramSpec → MakeProgramFromSpec → compile → dispatch → verify pipeline
// end-to-end on real hardware, with particular focus on DFB local accessor names.
//
// Requires: TT_METAL_SLOW_DISPATCH_MODE=1
//
// TODO: Switch to using fast dispatch once the MeshWorkload code paths are added.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalComputeKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeShardedTensorParameter;

// ============================================================================
// Test Fixture
// ============================================================================

class ProgramSpecHWTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        // These tests target Gen1 (WH/BH) only
        if (devices_.at(0)->arch() != tt::ARCH::WORMHOLE_B0 && devices_.at(0)->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

// ============================================================================
// DFB Local Accessor Name Loopback Test
// ============================================================================
//
// Proves that DFB local accessor names work end-to-end on real WH/BH hardware:
//   1. kernel_bindings_generated.h is emitted correctly (dfb::buf resolves at compile time)
//   2. The DFBAccessor mechanism works (DFB ID maps to the correct underlying CB)
//   3. Data flows correctly through the DFB from producer to consumer
//
// Pipeline:
//   Host writes random data → DRAM input buffer (single page = one bank)
//   Producer DM kernel (BRISC) reads DRAM → DFB (using dfb::buf)
//   Consumer DM kernel (NCRISC) reads DFB → DRAM (using dfb::buf)
//   Host reads DRAM output buffer and verifies match

TEST_F(ProgramSpecHWTest, DFBAccessorNameLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Test parameters
    constexpr uint32_t entry_size = 1024;  // bytes per DFB entry
    constexpr uint32_t num_entries = 4;    // DFB depth (double-buffer + margin)
    constexpr uint32_t num_transfers = 8;  // total entries to move through the DFB
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    // Use a single core for simplicity
    const NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Create DRAM buffers (single-page so all data is on one bank)
    // -------------------------------------------------------
    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // -------------------------------------------------------
    // Build ProgramSpec
    // -------------------------------------------------------
    ProgramSpec spec;
    spec.name = "dfb_accessor_loopback";

    // Producer: BRISC reads from DRAM → DFB
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 3;

    // Consumer: NCRISC reads DFB → DRAM
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    // DFB: both kernels bind it, with different local accessor names
    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    // -------------------------------------------------------
    // Create Program
    // -------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // -------------------------------------------------------
    // Set runtime args
    // -------------------------------------------------------
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              input_buffer->address(),
                              0u,  // bank_id (single-page buffer → bank 0)
                              num_transfers,
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              output_buffer->address(),
                              0u,  // bank_id
                              num_transfers,
                          }}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    // -------------------------------------------------------
    // Fill input buffer with known data
    // -------------------------------------------------------
    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(input_buffer, input_data);

    // -------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------
    detail::LaunchProgram(device, program);

    // -------------------------------------------------------
    // Verify
    // -------------------------------------------------------
    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// Named RTA / CRTA / CTA Loopback Test
// ============================================================================
//
// End-to-end on real WH/BH hardware. Exercises the full Metal 2.0 kernel-args feature
// surface (single-node scope):
//
//   Named args: named RTA, named CRTA, named CTA — via get_arg(args::name).
//   Vararg RTAs: multiple indices per kernel (0/1/2 on producer, 0/1 on consumer) via
//       get_vararg(idx). Different vararg count per kernel verifies that the baked-in
//       named_rta_words offset is per-kernel, not shared state.
//   Vararg CRTAs: get_common_vararg(0) on both kernels.
//
// Not covered here (intentional):
//   - num_runtime_varargs_per_node (the per-node override path). The internal schema
//     representation is a per-coord unordered_map regardless of how it was populated, so
//     the dispatch-time behavior is identical to the scalar path covered here. The
//     override-specific semantics are covered by host-side unit tests
//     (VarargScalarDefaultWithSparseOverrideSucceeds,
//      VarargSparseOverrideZeroErasesScalarDefault,
//      VarargPerNodeOverrideMixedEntryTypesSucceeds).
//
// Verification trick — XOR cancellation:
//   Each kernel computes the XOR of all its vararg values into a scalar sum and folds
//   that sum into the first word of every DFB entry (producer on write, consumer on read).
//   The host arranges both kernels' vararg values so their sums are equal, which means
//   the two XORs cancel and the first word survives the round-trip unchanged. End-to-end
//   input/output match then implies every vararg offset was computed correctly: if any
//   index returned the wrong word (a named RTA, a past-the-end vararg, etc.), the two
//   sums wouldn't match, the cancellation wouldn't happen, and the first word of each
//   output entry would come back corrupted.

TEST_F(ProgramSpecHWTest, NamedArgsLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "named_args_loopback";

    // Producer: BRISC reads DRAM → DFB. 1 named RTA, 1 named CRTA, 2 named CTAs, 3 RTA
    // varargs, 1 CRTA vararg.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_producer.cpp";
    producer.runtime_arg_schema.runtime_arg_names = {"src_addr"};
    producer.runtime_arg_schema.common_runtime_arg_names = {"num_entries"};
    producer.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3, .num_common_runtime_varargs = 1};
    producer.compile_time_args = {{"bank_id", 0}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads DFB → DRAM. Uses default `args` namespace, 1 named RTA,
    // 1 named CRTA, 2 named CTAs, 2 RTA varargs (note: different count from producer —
    // this verifies the named_rta_words offset is baked per-kernel), 1 CRTA vararg.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_consumer.cpp";
    consumer.runtime_arg_schema.runtime_arg_names = {"dst_addr"};
    consumer.runtime_arg_schema.common_runtime_arg_names = {"num_entries"};
    consumer.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 1};
    consumer.compile_time_args = {{"bank_id", 0}, {"entry_size", entry_size}};

    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries_in_dfb);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "loopback_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "loopback_dfb"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // Vararg values picked so both kernels' XOR sums equal the same non-trivial S.
    // The kernels fold this into the first word of each DFB entry; S ^ S = 0, so data
    // survives the round-trip ONLY IF both kernels read the correct vararg values at
    // the correct offsets. Non-trivial bits maximize the chance of a wrong-offset read
    // producing a detectable mismatch rather than a coincidentally-equal XOR.
    constexpr uint32_t kTargetXorSum = 0xDEADBEEFu;
    constexpr uint32_t kProducerRta0 = 0x11112222u;
    constexpr uint32_t kProducerRta1 = 0x33334444u;
    constexpr uint32_t kProducerRta2 = 0x55556666u;
    constexpr uint32_t kProducerCrta0 = kTargetXorSum ^ kProducerRta0 ^ kProducerRta1 ^ kProducerRta2;
    constexpr uint32_t kConsumerRta0 = 0x77778888u;
    constexpr uint32_t kConsumerRta1 = 0x9999AAAAu;
    constexpr uint32_t kConsumerCrta0 = kTargetXorSum ^ kConsumerRta0 ^ kConsumerRta1;

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .runtime_arg_values = {{node, {{"src_addr", input_buffer->address()}}}},
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {kProducerRta0, kProducerRta1, kProducerRta2}}},
                    .common_runtime_varargs = {kProducerCrta0},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = {{node, {{"dst_addr", output_buffer->address()}}}},
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {kConsumerRta0, kConsumerRta1}}},
                    .common_runtime_varargs = {kConsumerCrta0},
                },
        },
    };
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(input_buffer, input_data);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// Named Args Loopback — Compute Producer
// ============================================================================
//
// Companion test for NamedArgsLoopback that exercises the named-args surface
// from the COMPUTE compile path (TRISC_UNPACK / TRISC_MATH / TRISC_PACK).
// The named-args helpers reach a compute kernel via a completely different
// include chain than a DM kernel.
//
// Pipeline:
//   Compute kernel (TRISC) — produces out_dfb. Reads named RTAs/CRTAs/CTAs +
//       RTA/CRTA varargs; writes the XOR sum of all of them into the first
//       uint32_t of every entry, zeros the rest.
//   DM Consumer (NCRISC) — out_dfb → DRAM output. Positional varargs only.
//
// The kernel does NOT use the unpack/math/pack tile pipeline — just raw L1 writes
// from PACK after reserve_back. This is a plumbing test only; didn't want to
// tangle with type conversions....
//
// Verification: the host arranges every named arg + every vararg so their XOR
// equals a known target. Output DRAM should contain {target, 0, 0, …} per
// entry, exactly. A wrong offset on any accessor → wrong sum → test fails on
// the byte-for-byte compare.

TEST_F(ProgramSpecHWTest, NamedArgsLoopbackCompute) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "named_args_loopback_compute";

    // Compute kernel: produces out_dfb. The kernel under test — exercises every
    // named-arg accessor (RTA / CRTA / two CTAs) plus RTA + CRTA varargs.
    auto compute = MakeMinimalComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/named_args_loopback_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 1};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads out_dfb → DRAM. Reuses dfb_accessor_loopback_consumer.cpp
    // verbatim (positional varargs only).
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, num_entries_in_dfb);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;

    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out_dfb"}, "out_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {compute, consumer};
    spec.dataflow_buffers = {out_dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // Pick non-trivial bits so a wrong-offset read is unlikely to coincidentally
    // produce the same XOR. The compute kernel's sum is:
    //   magic ^ entry_size ^ num_tiles ^ input_offset ^ va0 ^ va1 ^ cv0
    // Solve for cv0 to make the sum equal kTargetXorSum.
    constexpr uint32_t kTargetXorSum = 0xDEADBEEFu;
    constexpr uint32_t kMagic = 0xCAFE0001u;
    constexpr uint32_t kInputOffset = 0x12345678u;
    constexpr uint32_t kVararg0 = 0xAAAA1111u;
    constexpr uint32_t kVararg1 = 0xBBBB2222u;
    constexpr uint32_t kCommonVararg0 =
        kTargetXorSum ^ kMagic ^ entry_size ^ num_transfers ^ kInputOffset ^ kVararg0 ^ kVararg1;

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"compute"},
            .runtime_arg_values = {{node, {{"input_offset", kInputOffset}}}},
            .common_runtime_arg_values = {{"num_tiles", num_transfers}},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {kVararg0, kVararg1}}},
                    .common_runtime_varargs = {kCommonVararg0},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {output_buffer->address(), 0u, num_transfers}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    constexpr uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t), 0u);
    for (uint32_t e = 0; e < num_transfers; ++e) {
        expected[e * words_per_entry] = kTargetXorSum;
    }

    ASSERT_EQ(output_data.size(), expected.size());
    EXPECT_EQ(output_data, expected);
}

// ============================================================================
// TT_KERNEL ("1st world arguments") Loopback — Data Movement
// ============================================================================
//
// Same DRAM → DFB → DRAM loopback as NamedArgsLoopback, but the producer and consumer kernels
// are authored in the TT_KERNEL function/template-parameter syntax (CTAs as template params,
// RTA/CRTA as function params, no hand-written kernel_main() and no get_arg() calls — genfiles
// generates the kernel_main() shim). Proves the generated shim binds the named args correctly
// end-to-end on real hardware for the data-movement compile path. No varargs (the TT_KERNEL
// syntax doesn't express them), so verification is by plain data round-trip: a wrong binding for
// src_addr / dst_addr / entry_size / bank_id / num_entries corrupts input == output.

TEST_F(ProgramSpecHWTest, TtKernelNamedArgsLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "tt_kernel_named_args_loopback";

    // Producer (BRISC) reads DRAM → DFB. TT_KERNEL form: bank_id/entry_size are template params
    // (CTAs); src_addr (RTA) and num_entries (CRTA) are function params.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tt_kernel_named_args_producer.cpp";
    producer.runtime_arg_schema.runtime_arg_names = {"src_addr"};
    producer.runtime_arg_schema.common_runtime_arg_names = {"num_entries"};
    producer.compile_time_args = {{"bank_id", 0}, {"entry_size", entry_size}};

    // Consumer (NCRISC) reads DFB → DRAM. Same TT_KERNEL form with dst_addr.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tt_kernel_named_args_consumer.cpp";
    consumer.runtime_arg_schema.runtime_arg_names = {"dst_addr"};
    consumer.runtime_arg_schema.common_runtime_arg_names = {"num_entries"};
    consumer.compile_time_args = {{"bank_id", 0}, {"entry_size", entry_size}};

    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries_in_dfb);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "loopback_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "loopback_dfb"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .runtime_arg_values = {{node, {{"src_addr", input_buffer->address()}}}},
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = {{node, {{"dst_addr", output_buffer->address()}}}},
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
        },
    };
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(input_buffer, input_data);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// TT_KERNEL ("1st world arguments") Loopback — Compute Producer
// ============================================================================
//
// The TT_KERNEL counterpart to NamedArgsLoopbackCompute, and the test that proves the generated
// kernel_main() shim is emitted on the COMPUTE (TRISC) compile path — the gap fixed by routing
// both genfiles paths through the shared shim helper. The compute kernel is authored in TT_KERNEL
// form (magic/entry_size as template CTAs; input_offset (RTA) and num_tiles (CRTA) as function
// params); the DM consumer reuses the existing positional-vararg DFB consumer verbatim.
//
// Verification: the kernel writes magic ^ entry_size ^ input_offset ^ num_tiles into word 0 of
// every entry; the host solves input_offset so the XOR equals a known target. A wrong binding on
// the compute path → wrong sum → wrong DRAM word → test fails.

TEST_F(ProgramSpecHWTest, TtKernelNamedArgsLoopbackCompute) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "tt_kernel_named_args_loopback_compute";

    // Compute kernel authored in TT_KERNEL form. magic/entry_size are template params (CTAs);
    // input_offset (RTA) and num_tiles (CRTA) are function params. No varargs.
    auto compute = MakeMinimalComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/tt_kernel_named_args_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads out_dfb → DRAM. Reuses the existing positional-vararg consumer.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, num_entries_in_dfb);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;

    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out_dfb"}, "out_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {compute, consumer};
    spec.dataflow_buffers = {out_dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // sum = magic ^ entry_size ^ input_offset ^ num_tiles. Solve input_offset for a known target.
    constexpr uint32_t kTargetXorSum = 0xDEADBEEFu;
    constexpr uint32_t kMagic = 0xCAFE0001u;
    constexpr uint32_t kInputOffset = kTargetXorSum ^ kMagic ^ entry_size ^ num_transfers;

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"compute"},
            .runtime_arg_values = {{node, {{"input_offset", kInputOffset}}}},
            .common_runtime_arg_values = {{"num_tiles", num_transfers}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {output_buffer->address(), 0u, num_transfers}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    constexpr uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t), 0u);
    for (uint32_t e = 0; e < num_transfers; ++e) {
        expected[e * words_per_entry] = kTargetXorSum;
    }

    ASSERT_EQ(output_data.size(), expected.size());
    EXPECT_EQ(output_data, expected);
}

// ============================================================================
// Semaphore Accessor Name Loopback Test
// ============================================================================
//
// Targeted test for the semaphore accessor name plumbing.
// Very minimal: just one semaphore and two kernels that sync on it via
// different local accessor names.
//
//   - Producer (BRISC) and consumer (NCRISC) each resolve their accessor name
//     to a sem ID. Test only completes if both land on the same underlying ID.
//   - If producer's sem::signal ID != consumer's sem::waiter ID, consumer hangs
//     forever on wait(1).
//
// Proves: kernel_bindings_generated.h emits the sem:: namespace correctly, both
// kernels' views agree on the sem ID, Metal 2.0 allocates the sem (on Gen1).

TEST_F(ProgramSpecHWTest, SemaphoreAccessorNameLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    const NodeCoord node{0, 0};

    // A SemaphoreSpec describes a Program-scope semaphore: it identifies the sem by name and
    // declares which nodes will see it. Initial value defaults to 0.
    SemaphoreSpec sem{
        .unique_id = SemaphoreSpecName{"only_sem"},
        .target_nodes = node,
    };

    // A KernelSpec binds the semaphore by its `unique_id` and gives it a kernel-local
    // `accessor_name` — the name the kernel source uses to refer to it. The runtime emits
    // `sem::<accessor_name>` constants in `kernel_bindings_generated.h` for the kernel to
    // consume. The producer and consumer below choose different accessor names for the same
    // semaphore.
    // These two DM kernels only need to land on distinct DM processors. The READER/WRITER role
    // hints are the idiomatic way to get that — the producer writes the semaphore signal, the
    // consumer reads it — with no need to hand-pick a processor/NOC via an explicit Gen1Config.
    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_producer.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"only_sem"}, .accessor_name = "signal"}},
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::WRITER,
            },
    };
    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_consumer.cpp",
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = SemaphoreSpecName{"only_sem"}, .accessor_name = "waiter"}},
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::READER,
            },
    };

    // A WorkUnitSpec describes the kernels that run on a shared set of nodes.
    WorkUnitSpec work_unit{
        .name = "work_unit_0",
        .kernels = {KernelSpecName{"producer"}, KernelSpecName{"consumer"}},
        .target_nodes = node,
    };

    // The ProgramSpec aggregates everything and is consumed by `MakeProgramFromSpec`.
    ProgramSpec spec{
        .name = "semaphore_accessor_loopback",
        .kernels = {producer, consumer},
        .semaphores = {sem},
        .work_units = std::vector<WorkUnitSpec>{work_unit},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);
    detail::LaunchProgram(device, program);
    // If we got here, both kernels resolved their sem accessors to the same ID.
}

// ============================================================================
// TensorAccessor Binding End-to-End Loopback Test
// ============================================================================
//
// Proves that the Metal 2.0 TensorAccessor binding feature works end-to-end on real WH/BH:
//   1. Spec → MakeProgramFromSpec resolves the binding's TensorSpec into a correct CTA payload
//      (page size, args_config, bank coords, alignment).
//   2. Each binding's slot in the kernel's TensorBinding address section is filled with
//      MeshTensor::address() at enqueue.
//   3. kernel_bindings_generated.h emits a `tensor::` namespace with a working type alias + token.
//   4. TensorAccessor(tensor::name) constructs an accessor whose get_noc_addr returns
//      addresses that NoC reads/writes actually use correctly.
//
// Pipeline:
//   Host writes known data → input MeshTensor (DRAM)
//   Producer DM kernel (BRISC):  input MeshTensor → DFB,  via TensorAccessor(tensor::input_tensor)
//   Consumer DM kernel (NCRISC): DFB → output MeshTensor, via TensorAccessor(tensor::output_tensor)
//   Host reads output MeshTensor and verifies match
//
// DM-only on purpose: TensorAccessor is the NOC-capable accessor and only compiles on DM builds.
// The compute (TRISC) path uses LocalTensorAccessor instead — proven by
// LocalTensorAccessorBindingCompileComputeKernel below.

TEST_F(ProgramSpecHWTest, TensorAccessorBindingLoopback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Tensor: 8 pages × 1024 bytes (BFLOAT16, ROW_MAJOR, shape {8, 512} → page = row = 1024 B)
    constexpr uint32_t num_pages = 8;
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t total_bytes = num_pages * page_size;
    constexpr uint32_t num_dfb_entries = 4;

    const NodeCoord node{0, 0};

    // -------------------------------------------------------
    // Allocate input + output MeshTensors (DRAM, interleaved)
    // -------------------------------------------------------
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto tensor_spec = TensorSpec(Shape{num_pages, 512}, tensor_layout);

    MeshTensor input_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    MeshTensor output_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    // -------------------------------------------------------
    // Build ProgramSpec: 2 DM kernels + 1 DFB + 2 TensorParameters
    // -------------------------------------------------------
    ProgramSpec spec;
    spec.name = "ta_binding_loopback";

    // Producer (BRISC): reads input tensor via TA binding, pushes to DFB
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 1;

    // Consumer (NCRISC): pops from DFB, writes output tensor via TA binding
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 1;

    // DFB connecting the two kernels
    auto dfb = MakeMinimalDFB("input_dfb", page_size, num_dfb_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"input_dfb"}, "input_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"input_dfb"}, "input_dfb"));

    // TensorAccessor bindings: each kernel sees its own tensor under its accessor name
    BindTensorParameterToKernel(producer, "input_tensor", "input_tensor");
    BindTensorParameterToKernel(consumer, "output_tensor", "output_tensor");

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{"input_tensor"}, .spec = tensor_spec},
        TensorParameter{.unique_id = TensorParamName{"output_tensor"}, .spec = tensor_spec},
    };
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    // -------------------------------------------------------
    // Create Program
    // -------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // -------------------------------------------------------
    // Set runtime args
    // -------------------------------------------------------
    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {num_pages}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {num_pages}}},
                },
        },
    };
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{input_tensor}},
        {TensorParamName{"output_tensor"}, TensorArgument{output_tensor}},
    };
    SetProgramRunArgs(program, params);

    // -------------------------------------------------------
    // Fill input tensor with known data
    // -------------------------------------------------------
    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    detail::WriteToBuffer(*input_tensor.mesh_buffer().get_reference_buffer(), input_data);

    // -------------------------------------------------------
    // Dispatch
    // -------------------------------------------------------
    detail::LaunchProgram(device, program);

    // -------------------------------------------------------
    // Verify
    // -------------------------------------------------------
    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(*output_tensor.mesh_buffer().get_reference_buffer(), output_data);

    ASSERT_EQ(output_data.size(), input_data.size());
    EXPECT_EQ(output_data, input_data);
}

// ============================================================================
// LocalTensorAccessor Binding — Compute (TRISC) Compile + Token-Wiring Proof
// ============================================================================
//
// Proves the compute-kernel path for tensor bindings, which previously could not compile:
//   1. A compute (TRISC) kernel binds a tensor and constructs LocalTensorAccessor<uint32_t> from the
//      binding token. The generated header emits only the NOC-free token header on the TRISC build, so
//      this compiles (tensor_accessor.h would not — it needs NOC_INDEX, absent on compute builds).
//   2. ValidateProgramSpec accepts the compute-kernel tensor binding (the old guard is gone).
//   3. The binding's base-address CRTA is broadcast to the compute kernel and resolves to the local
//      L1 shard address — verified by comparing the reported address to the bound tensor's address.
//
// Pipeline (compute-only producer, no DM producer needed):
//   Compute kernel (TRISC) — constructs LocalTensorAccessor (token ctor) and a second via the legacy
//       base-address ctor, deposits {base_address, get_unsafe_ptr, &operator[], legacy-ctor base} into
//       each out_dfb entry (raw L1 writes from PACK); all four should equal the bound tensor's address.
//   DM consumer (NCRISC)   — out_dfb → DRAM output.
//
// The tensor is a single-shard L1 tensor on core (0,0) (the compute kernel's core), so its shard base
// address equals MeshTensor::address(). No dereference of the shard occurs (address-of only), so the
// proof does not depend on the shard's contents.

TEST_F(ProgramSpecHWTest, LocalTensorAccessorBindingCompileComputeKernel) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_tiles = 2;  // entries the compute kernel pushes
    constexpr uint32_t total_bytes = entry_size * num_tiles;

    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(dram_config);

    // Single-shard L1 tensor on core (0,0): one 32x32 BFLOAT16 tile.
    auto tensor_param = MakeShardedTensorParameter("local_t", Shape{32, 32}, {32, 32}, /*num_cores=*/1);
    MeshTensor local_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_param.spec, TensorTopology{});

    ProgramSpec spec;
    spec.name = "local_tensor_accessor_compute";

    // Compute kernel (the kernel under test) — binds the tensor, produces into out_dfb.
    auto compute = MakeMinimalComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/local_tensor_accessor_compute.cpp";
    compute.compile_time_args = {{"entry_size", entry_size}, {"num_tiles", num_tiles}};
    BindTensorParameterToKernel(compute, "local_t", "local_t");

    // Consumer (NCRISC): drains out_dfb → DRAM. Reuses dfb_accessor_loopback_consumer.cpp verbatim.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, num_entries_in_dfb);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;

    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out_dfb"}, "out_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {compute, consumer};
    spec.dataflow_buffers = {out_dfb};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"compute"}},
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {output_buffer->address(), 0u, num_tiles}}},
                },
        },
    };
    params.tensor_args = {
        {TensorParamName{"local_t"}, TensorArgument{local_tensor}},
    };
    SetProgramRunArgs(program, params);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    const uint32_t expected_address = static_cast<uint32_t>(local_tensor.address());
    constexpr uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    ASSERT_EQ(output_data.size(), total_bytes / sizeof(uint32_t));
    for (uint32_t e = 0; e < num_tiles; ++e) {
        const uint32_t* entry = output_data.data() + e * words_per_entry;
        EXPECT_EQ(entry[0], expected_address) << "entry " << e << ": get_bank_base_address mismatch";
        EXPECT_EQ(entry[1], expected_address) << "entry " << e << ": get_unsafe_ptr mismatch";
        EXPECT_EQ(entry[2], expected_address) << "entry " << e << ": &operator[] mismatch";
        EXPECT_EQ(entry[3], expected_address) << "entry " << e << ": legacy base-address ctor mismatch";
    }
}

// ============================================================================
// Multi-binding RISC-mask Uniformity (Gen1)
// ============================================================================
//
// On Gen1 (WH/BH) the per-kernel risc_mask is a deterministic function of the
// KernelSpec's config. Multi-binding requires all same-role KernelSpecs to
// share that mask; mismatched processor placement on the producer (or consumer)
// side is a user error and must be rejected with an actionable message.
TEST_F(ProgramSpecHWTest, MultiBindingProducerMaskMismatchFails) {
    auto mesh_device = devices_.at(0);

    NodeCoord node0{0, 0};
    NodeCoord node1{0, 1};

    auto producer_g1 = MakeMinimalGen1DMKernel("producer_g1", DataMovementProcessor::RISCV_0);
    auto producer_g2 = MakeMinimalGen1DMKernel("producer_g2", DataMovementProcessor::RISCV_1);
    auto consumer = MakeMinimalComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer_g1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_g2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    ProgramSpec spec;
    spec.name = "multi_binding_mask_mismatch";
    spec.kernels = {producer_g1, producer_g2, consumer};
    spec.dataflow_buffers = {dfb};
    // consumer in both WUs (single-KernelSpec multi-WU membership) → consumer-side mask is fine.
    // producer_g1 in wu_g1 (RISCV_0); producer_g2 in wu_g2 (RISCV_1) → mismatched producer masks.
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer_g1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer_g2", "consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple PRODUCER KernelSpecs ('producer_g1', 'producer_g2') with "
                                 "mismatched processor placement")));
}

// ============================================================================
// Kernel Scratchpad: write / readback under slow dispatch
// ============================================================================
//
// The slow-dispatch counterpart of test_scratchpad_hw.cpp's fast-dispatch ScratchpadWriteReadback. A
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
    IDevice* device = mesh_device->get_devices()[0];

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
                .gen1_config =
                    DataMovementHardwareConfig::Gen1Config{
                        .processor = DataMovementProcessor::RISCV_0,
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
        .runtime_arg_values = {{node, {{"report_addr", kReportAddr}}}},
    }};
    SetProgramRunArgs(program, params);

    // Pre-zero the report location so a kernel that never wrote it would be caught (the readback base
    // address would be 0, which is not a valid scratchpad L1 address → the pattern check fails).
    std::vector<uint32_t> zero_report(1, 0u);
    detail::WriteToDeviceL1(device, node, kReportAddr, zero_report);

    // Dispatch via the slow-dispatch path (blocking — wait_until_cores_done defaults to true).
    detail::LaunchProgram(device, program);

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

// ============================================================================
// Scratchpad + Common-Vararg CRTA Offset Under Partial Update (regression)
// ============================================================================
//
// Regression guard for the "A1" bug: UpdateProgramRunArgs once computed the common-vararg base as
// `named CRTAs + tensor_binding_section_words`, omitting the scratchpad section. The CRTA buffer
// layout is four sections — [named | tensor bindings | scratchpads | varargs] — and both
// SetProgramRunArgs and the device-side crta_layout.vararg_section_offset account for the scratchpad
// words. So for a kernel with BOTH a scratchpad binding AND a common vararg, a partial
// UpdateProgramRunArgs would write the vararg one word too early — into the scratchpad base-address
// slot — leaving the device reading the stale vararg from the (correct) vararg offset. The fix
// (program_run_args.cpp) adds scratchpad_section_words to the vararg base; this test pins it.
//
// Layout for the producer below: [0 named | 0 tensor bindings | 1 scratchpad | 1 vararg].
//   - scratchpad base-address slot: CRTA word 0
//   - common vararg slot:           CRTA word 1  (== vararg_section_offset)
//
// Sequence: SetProgramRunArgs (vararg = OLD), then UpdateProgramRunArgs (vararg = NEW), then
// dispatch. The producer stages [get_common_vararg(0), scratch_base] into a DFB entry; the consumer
// drains that entry to DRAM. The scratchpad is genuinely framework-allocated here, so scratch_base
// is a real L1 address (not a sentinel). Host then checks:
//   word 0 (vararg)       == NEW                      — the partial update reached the real vararg slot
//   word 1 (scratch base) != NEW (and != OLD, != 0)  — the scratchpad slot was NOT clobbered, and
//                                                       still holds a real allocated L1 address
//
// With the A1 bug the update writes NEW into CRTA word 0 (the scratchpad slot) instead of word 1, so
// the device sees word 0 (vararg) == OLD and word 1 (scratch base) == NEW — both checks fail.
TEST_F(ProgramSpecHWTest, ScratchpadCommonVarargPartialUpdate) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;  // bytes per DFB entry
    constexpr uint32_t num_entries = 4;    // DFB depth
    constexpr uint32_t kOldVararg = 0x11111111u;
    constexpr uint32_t kNewVararg = 0x22222222u;

    const NodeCoord node{0, 0};

    // Output buffer holds one DFB entry (single page → single bank).
    InterleavedBufferConfig dram_config{
        .device = device, .size = entry_size, .page_size = entry_size, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "scratchpad_common_vararg_partial_update";

    // Producer (BRISC): binds a scratchpad AND declares one common vararg — the exact combination
    // that trips the A1 offset bug. Stages [common_vararg, scratch_base] into the DFB.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    const uint32_t vararg_value = get_common_vararg(0);
    Scratchpad<int32_t> pad(scratch::scratch);
    const uint32_t scratch_base = pad.get_base_address();
    DataflowBuffer buf(dfb::stage);
    buf.reserve_back(1);
    volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)buf.get_write_ptr();
    w[0] = vararg_value;
    w[1] = scratch_base;
    buf.push_back(1);
}
)"};
    producer.advanced_options = KernelAdvancedOptions{.num_common_runtime_varargs = 1};
    producer.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    // Consumer (NCRISC): drains the DFB entry to DRAM. Uses named RTAs so it can be re-supplied on
    // the partial update below.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    auto dst_addr = get_arg(args::dst_addr);
    auto bank_id = get_arg(args::bank_id);
    DataflowBuffer buf(dfb::stage);
    buf.wait_front(1);
    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
    noc_async_write(buf.get_read_ptr(), dst_noc_addr, buf.get_entry_size());
    noc_async_write_barrier();
    buf.pop_front(1);
}
)"};
    consumer.runtime_arg_schema.runtime_arg_names = {"dst_addr", "bank_id"};

    auto dfb = MakeMinimalDFB("stage", entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"stage"}, "stage"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"stage"}, "stage"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // Initial full set: producer's common vararg = OLD; consumer's named RTAs point at the output.
    ProgramRunArgs set_params;
    set_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = {kOldVararg}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = {{node, {{"dst_addr", output_buffer->address()}, {"bank_id", 0u}}}},
        },
    };
    SetProgramRunArgs(program, set_params);

    // Partial update: refresh the producer's common vararg to NEW. This is the path that
    // mis-computed the vararg base when a scratchpad section is present. The consumer's named per-node
    // RTAs are not declared enqueue-invariant, so it is re-supplied unchanged; only the producer's
    // CRTA buffer is relevant to the bug.
    ProgramRunArgs update_params;
    update_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = {kNewVararg}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = {{node, {{"dst_addr", output_buffer->address()}, {"bank_id", 0u}}}},
        },
    };
    UpdateProgramRunArgs(program, update_params);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);

    ASSERT_GE(output_data.size(), 2u);
    EXPECT_EQ(output_data[0], kNewVararg)
        << "Partial UpdateProgramRunArgs must write the common vararg to the real vararg slot "
           "(after the scratchpad section). A wrong base writes it into the scratchpad slot instead, "
           "leaving the device reading the stale OLD vararg.";
    // The scratchpad base-address slot must survive the partial update. The scratchpad is really
    // allocated, so this slot holds a live L1 address — never the vararg value. With the A1 bug the
    // update overwrites it with kNewVararg.
    EXPECT_NE(output_data[1], kNewVararg)
        << "Scratchpad base-address slot was clobbered by the common-vararg partial update — the "
           "vararg base omitted the scratchpad section (A1 bug).";
    EXPECT_NE(output_data[1], kOldVararg) << "Scratchpad base-address slot holds a stale vararg value, not an L1 base.";
    EXPECT_NE(output_data[1], 0u)
        << "Scratchpad base-address slot is 0 — the framework-allocated base was not delivered.";
}

// ============================================================================
// Kernel Scratchpad: disjoint-node multi-binding (write / readback)
// ============================================================================
//
// Proves the multi-binding relaxation on real hardware: one ScratchpadSpec bound by TWO kernels on
// DISJOINT nodes. Each node hosts exactly one binding kernel, so each gets its own private per-node
// instance — allocation and CRTA delivery are per-binding-kernel (allocate_scratchpads stacks each
// kernel's scratchpad onto its own cores' allocators), so the two bindings never interact. Both
// kernels write a known pattern into their scratchpad and report its framework-allocated base to a
// host-known L1 address on their own node; the host reads each base and confirms the pattern landed.
TEST_F(ProgramSpecHWTest, ScratchpadMultiBindDisjointNodesWriteReadback) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t kScratchpadBytes = 64;                            // 16 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kReportAddr = 100 * 1024;                         // host-known fixed L1 addr (per node)
    constexpr uint32_t kPatternBase = 0xC0DE0000u;                       // must match the kernel

    const NodeCoord node_a{0, 0};
    const NodeCoord node_b{1, 0};  // disjoint from node_a

    // Two DM kernels, same source, binding the SAME scratchpad. Each on its own node — both on
    // RISCV_0 is fine because they never share a node.
    auto make_kernel = [](const std::string& name) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_write_pattern.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
            .hw_config =
                DataMovementHardwareConfig{
                    .gen1_config =
                        DataMovementHardwareConfig::Gen1Config{
                            .processor = DataMovementProcessor::RISCV_0,
                        },
                },
        };
        k.scratchpad_bindings.push_back(
            KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
        return k;
    };

    ProgramSpec spec;
    spec.name = "scratchpad_multibind_disjoint";
    spec.kernels = {make_kernel("scratch_kernel_a"), make_kernel("scratch_kernel_b")};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node_a, {"scratch_kernel_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"scratch_kernel_b"}),
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"scratch_kernel_a"},
            .runtime_arg_values = {{node_a, {{"report_addr", kReportAddr}}}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"scratch_kernel_b"},
            .runtime_arg_values = {{node_b, {{"report_addr", kReportAddr}}}},
        },
    };
    SetProgramRunArgs(program, params);

    // Pre-zero both report locations so a kernel that never wrote its base would be caught.
    std::vector<uint32_t> zero_report(1, 0u);
    detail::WriteToDeviceL1(device, node_a, kReportAddr, zero_report);
    detail::WriteToDeviceL1(device, node_b, kReportAddr, zero_report);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> expected(kNumElems);
    for (uint32_t i = 0; i < kNumElems; i++) {
        expected[i] = kPatternBase + i;
    }

    // Each node's binding kernel must have reached a real, private scratchpad and written the pattern.
    for (const NodeCoord& node : {node_a, node_b}) {
        std::vector<uint32_t> reported;
        detail::ReadFromDeviceL1(device, node, kReportAddr, sizeof(uint32_t), reported);
        ASSERT_EQ(reported.size(), 1u);
        const uint32_t scratch_base = reported[0];
        EXPECT_NE(scratch_base, 0u) << "Kernel on node " << node.str() << " reported a 0 scratchpad base address";

        std::vector<uint32_t> scratch_contents;
        detail::ReadFromDeviceL1(device, node, scratch_base, kScratchpadBytes, scratch_contents);
        ASSERT_EQ(scratch_contents.size(), kNumElems);
        EXPECT_EQ(scratch_contents, expected) << "Scratchpad L1 on node " << node.str() << " at base 0x" << std::hex
                                              << scratch_base << " did not contain the pattern the kernel wrote";
    }
}

// ============================================================================
// Kernel Scratchpad: base re-delivered after a DFB resize between enqueues (regression)
// ============================================================================
//
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
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;      // bytes per DFB entry (constant)
    constexpr uint32_t num_entries_small = 2;  // initial DFB depth
    constexpr uint32_t num_entries_large = 8;  // grown depth → +6 KB region → scratchpad moves
    constexpr uint32_t kScratchpadBytes = 64;  // 16 x uint32_t
    constexpr uint32_t kNumElems = kScratchpadBytes / sizeof(uint32_t);  // 16
    constexpr uint32_t kPatternBase = 0xC0DE0000u;                       // must match the kernel

    const NodeCoord node{0, 0};

    // Output buffer holds one DFB entry (single page → single bank).
    InterleavedBufferConfig dram_config{
        .device = device, .size = entry_size, .page_size = entry_size, .buffer_type = BufferType::DRAM};
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "scratchpad_base_redelivered_after_dfb_resize";

    // Producer (BRISC): write a pattern into the scratchpad, then stage its base address into the DFB.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
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
#include "experimental/kernel_args.h"
void kernel_main() {
    auto dst_addr = get_arg(args::dst_addr);
    auto bank_id = get_arg(args::bank_id);
    DataflowBuffer buf(dfb::stage);
    buf.wait_front(1);
    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
    noc_async_write(buf.get_read_ptr(), dst_noc_addr, buf.get_entry_size());
    noc_async_write_barrier();
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

    Program program = MakeProgramFromSpec(*mesh_device, spec);

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
            .runtime_arg_values = {{node, {{"dst_addr", output_buffer->address()}, {"bank_id", 0u}}}},
        }};
        params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"stage"}, .num_entries = dfb_num_entries});
        SetProgramRunArgs(program, params);

        detail::LaunchProgram(device, program);

        std::vector<uint32_t> out;
        detail::ReadFromBuffer(output_buffer, out);
        EXPECT_FALSE(out.empty());
        const uint32_t base = out.empty() ? 0u : out[0];
        EXPECT_NE(base, 0u) << "Kernel reported a 0 scratchpad base address (token not delivered?)";

        // The scratchpad must be real, writable L1 at the reported base.
        std::vector<uint32_t> scratch_contents;
        detail::ReadFromDeviceL1(device, node, base, kScratchpadBytes, scratch_contents);
        EXPECT_EQ(scratch_contents, expected_pattern)
            << "Scratchpad L1 at reported base 0x" << std::hex << base << " did not contain the pattern";
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
