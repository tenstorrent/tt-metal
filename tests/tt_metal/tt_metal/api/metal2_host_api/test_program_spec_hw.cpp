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
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

using test_helpers::BindDFBToKernel;
using test_helpers::MakeMinimalComputeKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

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
    spec.program_id = "dfb_accessor_loopback";

    // Producer: BRISC reads from DRAM → DFB
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp"};
    producer.runtime_arguments_schema.num_runtime_varargs = 3;

    // Consumer: NCRISC reads DFB → DRAM
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp"};
    consumer.runtime_arguments_schema.num_runtime_varargs = 3;

    // DFB: both kernels bind it, with different local accessor names
    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    BindDFBToKernel(producer, "loopback_dfb", "my_local_dfb_name", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "loopback_dfb", "a_dfb_named_bob", KernelSpec::DFBEndpointType::CONSUMER);

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
    ProgramRunParams params;
    params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "producer",
            .runtime_varargs =
                {{node,
                  {
                      input_buffer->address(),
                      0u,  // bank_id (single-page buffer → bank 0)
                      num_transfers,
                  }}},
        },
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "consumer",
            .runtime_varargs =
                {{node,
                  {
                      output_buffer->address(),
                      0u,  // bank_id
                      num_transfers,
                  }}},
        },
    };
    SetProgramRunParameters(program, params);

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
    spec.program_id = "named_args_loopback";

    // Producer: BRISC reads DRAM → DFB. 1 named RTA, 1 named CRTA, 2 named CTAs, 3 RTA
    // varargs, 1 CRTA vararg.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_producer.cpp"};
    producer.runtime_arguments_schema.named_runtime_args = {"src_addr"};
    producer.runtime_arguments_schema.named_common_runtime_args = {"num_entries"};
    producer.runtime_arguments_schema.num_runtime_varargs = 3;
    producer.runtime_arguments_schema.num_common_runtime_varargs = 1;
    producer.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads DFB → DRAM. Uses default `args` namespace, 1 named RTA,
    // 1 named CRTA, 2 named CTAs, 2 RTA varargs (note: different count from producer —
    // this verifies the named_rta_words offset is baked per-kernel), 1 CRTA vararg.
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/named_args_loopback_consumer.cpp"};
    consumer.runtime_arguments_schema.named_runtime_args = {"dst_addr"};
    consumer.runtime_arguments_schema.named_common_runtime_args = {"num_entries"};
    consumer.runtime_arguments_schema.num_runtime_varargs = 2;
    consumer.runtime_arguments_schema.num_common_runtime_varargs = 1;
    consumer.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", entry_size}};

    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, num_entries_in_dfb);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    BindDFBToKernel(producer, "loopback_dfb", "loopback_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "loopback_dfb", "loopback_dfb", KernelSpec::DFBEndpointType::CONSUMER);

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

    ProgramRunParams params;
    params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "producer",
            .named_runtime_args = {{.node = node, .args = {{"src_addr", input_buffer->address()}}}},
            .named_common_runtime_args = {{"num_entries", num_transfers}},
            .runtime_varargs = {{node, {kProducerRta0, kProducerRta1, kProducerRta2}}},
            .common_runtime_varargs = {kProducerCrta0},
        },
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "consumer",
            .named_runtime_args = {{.node = node, .args = {{"dst_addr", output_buffer->address()}}}},
            .named_common_runtime_args = {{"num_entries", num_transfers}},
            .runtime_varargs = {{node, {kConsumerRta0, kConsumerRta1}}},
            .common_runtime_varargs = {kConsumerCrta0},
        },
    };
    SetProgramRunParameters(program, params);

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
    spec.program_id = "named_args_loopback_compute";

    // Compute kernel: produces out_dfb. The kernel under test — exercises every
    // named-arg accessor (RTA / CRTA / two CTAs) plus RTA + CRTA varargs.
    auto compute = MakeMinimalComputeKernel("compute");
    compute.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/compute/named_args_loopback_compute.cpp"};
    compute.runtime_arguments_schema.named_runtime_args = {"input_offset"};
    compute.runtime_arguments_schema.named_common_runtime_args = {"num_tiles"};
    compute.runtime_arguments_schema.num_runtime_varargs = 2;
    compute.runtime_arguments_schema.num_common_runtime_varargs = 1;
    compute.compile_time_arg_bindings = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    // Consumer: NCRISC reads out_dfb → DRAM. Reuses dfb_accessor_loopback_consumer.cpp
    // verbatim (positional varargs only).
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source =
        KernelSpec::SourceFilePath{"tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp"};
    consumer.runtime_arguments_schema.num_runtime_varargs = 3;

    auto out_dfb = MakeMinimalDFB("out_dfb", entry_size, num_entries_in_dfb);
    out_dfb.data_format_metadata = tt::DataFormat::Float16_b;

    BindDFBToKernel(compute, "out_dfb", "out_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(consumer, "out_dfb", "a_dfb_named_bob", KernelSpec::DFBEndpointType::CONSUMER);

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

    ProgramRunParams params;
    params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "compute",
            .named_runtime_args = {{.node = node, .args = {{"input_offset", kInputOffset}}}},
            .named_common_runtime_args = {{"num_tiles", num_transfers}},
            .runtime_varargs = {{node, {kVararg0, kVararg1}}},
            .common_runtime_varargs = {kCommonVararg0},
        },
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "consumer",
            .runtime_varargs = {{node, {output_buffer->address(), 0u, num_transfers}}},
        },
    };
    SetProgramRunParameters(program, params);

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
        .unique_id = "only_sem",
        .target_nodes = node,
    };

    // A KernelSpec binds the semaphore by its `unique_id` and gives it a kernel-local
    // `accessor_name` — the name the kernel source uses to refer to it. The runtime emits
    // `sem::<accessor_name>` constants in `kernel_bindings_generated.h` for the kernel to
    // consume. The producer and consumer below choose different accessor names for the same
    // semaphore.
    KernelSpec producer{
        .unique_id = "producer",
        .source =
            KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_producer.cpp"},
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = "only_sem", .accessor_name = "signal"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                    },
            },
    };
    KernelSpec consumer{
        .unique_id = "consumer",
        .source =
            KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/semaphore_accessor_loopback_consumer.cpp"},
        .num_threads = 1,
        .semaphore_bindings = {{.semaphore_spec_name = "only_sem", .accessor_name = "waiter"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                    },
            },
    };

    // A WorkUnitSpec describes the kernels that run on a shared set of nodes.
    WorkUnitSpec work_unit{
        .unique_id = "work_unit_0",
        .kernels = {"producer", "consumer"},
        .target_nodes = node,
    };

    // The ProgramSpec aggregates everything and is consumed by `MakeProgramFromSpec`.
    ProgramSpec spec{
        .program_id = "semaphore_accessor_loopback",
        .kernels = {producer, consumer},
        .semaphores = {sem},
        .work_units = std::vector<WorkUnitSpec>{work_unit},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);
    detail::LaunchProgram(device, program);
    // If we got here, both kernels resolved their sem accessors to the same ID.
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
