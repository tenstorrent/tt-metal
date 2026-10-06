// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end kernel arguments on WH/BH silicon: named args, varargs, TT_KERNEL shims and CRTA sections.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecHWTest;

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

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    auto input_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto output_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto& cq = mesh_device->mesh_command_queue();

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
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"src_addr", input_buffer->address()}}),
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {kProducerRta0, kProducerRta1, kProducerRta2}}},
                    .common_runtime_varargs = {kProducerCrta0},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"dst_addr", output_buffer->address()}}),
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
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/true);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> output_data;
    distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

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
// Pipeline (compute-only; no DFB / DM consumer):
//   Compute kernel (TRISC) — reads named RTAs/CRTAs/CTAs + RTA/CRTA varargs; PACK writes the
//       XOR sum into L1 at the allocator base address on the compute core. Host reads that
//       word via ReadFromDeviceL1 after LaunchProgram.
//
// Verification: the host arranges every named arg + every vararg so their XOR equals a known
// target. A wrong offset on any accessor → wrong sum → test fails.

TEST_F(ProgramSpecHWTest, NamedArgsLoopbackCompute) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t entry_size = 1024;  // CTA value folded into the XOR (not a DFB size)
    constexpr uint32_t num_tiles = 8;      // CRTA value folded into the XOR
    const uint32_t report_addr = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);

    const NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "named_args_loopback_compute";

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/named_args_loopback_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset", "report_addr"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 2, .num_common_runtime_varargs = 1};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    spec.kernels = {compute};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute"})};

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
        kTargetXorSum ^ kMagic ^ entry_size ^ num_tiles ^ kInputOffset ^ kVararg0 ^ kVararg1;

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute"},
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(node, {{"input_offset", kInputOffset}, {"report_addr", report_addr}}),
        .common_runtime_arg_values = {{"num_tiles", num_tiles}},
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, {kVararg0, kVararg1}}},
                .common_runtime_varargs = {kCommonVararg0},
            },
    }};
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> zero_report(1, 0u);
    slow_dispatch::WriteToL1(*mesh_device, node, report_addr, zero_report);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(*mesh_device, node, report_addr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_EQ(reported[0], kTargetXorSum);
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

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries_in_dfb = 4;
    constexpr uint32_t num_transfers = 8;
    constexpr uint32_t total_bytes = entry_size * num_transfers;

    const NodeCoord node{0, 0};

    auto input_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto output_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = total_bytes},
        {.page_size = total_bytes, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto& cq = mesh_device->mesh_command_queue();

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
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"src_addr", input_buffer->address()}}),
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"dst_addr", output_buffer->address()}}),
            .common_runtime_arg_values = {{"num_entries", num_transfers}},
        },
    };
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<uint32_t>(i);
    }
    distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/true);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> output_data;
    distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

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
// form (magic/entry_size as template CTAs; input_offset + report_addr (RTAs) and num_tiles (CRTA)
// as function params). No DFB — PACK writes the XOR into L1 at the allocator base address on the
// compute core (same ReadFromDeviceL1 idiom as NamedArgsLoopbackCompute).
//
// Verification: the host solves input_offset so magic ^ entry_size ^ input_offset ^ num_tiles
// equals a known target. A wrong binding on the compute path → wrong sum → test fails.

TEST_F(ProgramSpecHWTest, TtKernelNamedArgsLoopbackCompute) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t entry_size = 1024;  // CTA value folded into the XOR (not a DFB size)
    constexpr uint32_t num_tiles = 8;      // CRTA value folded into the XOR
    const uint32_t report_addr = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);

    const NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "tt_kernel_named_args_loopback_compute";

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/tt_kernel_named_args_compute.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"input_offset", "report_addr"};
    compute.runtime_arg_schema.common_runtime_arg_names = {"num_tiles"};
    compute.compile_time_args = {{"magic", 0xCAFE0001u}, {"entry_size", entry_size}};

    spec.kernels = {compute};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    // sum = magic ^ entry_size ^ input_offset ^ num_tiles. Solve input_offset for a known target.
    constexpr uint32_t kTargetXorSum = 0xDEADBEEFu;
    constexpr uint32_t kMagic = 0xCAFE0001u;
    constexpr uint32_t kInputOffset = kTargetXorSum ^ kMagic ^ entry_size ^ num_tiles;

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute"},
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(node, {{"input_offset", kInputOffset}, {"report_addr", report_addr}}),
        .common_runtime_arg_values = {{"num_tiles", num_tiles}},
    }};
    SetProgramRunArgs(program, params);

    std::vector<uint32_t> zero_report(1, 0u);
    slow_dispatch::WriteToL1(*mesh_device, node, report_addr, zero_report);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(*mesh_device, node, report_addr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    EXPECT_EQ(reported[0], kTargetXorSum);
}

// ============================================================================
// CRTA buffer: all four sections read at correct offsets — set + partial update (regression)
// ============================================================================

// The common-runtime-arg (CRTA) buffer is laid out in four sections, in order:
//     [ named CRTAs | tensor bindings | scratchpads | common varargs ]
// Both the host (SetProgramRunArgs full assembly + UpdateProgramRunArgs in-place patch) and the
// device-side generated header compute each section's offset independently, so a boundary miscomputed
// on either side silently cross-contaminates regions — the classic sneaky offset bug. This generalizes
// the "A1" regression (which carried only a scratchpad + one vararg): here ONE kernel carries ALL FOUR
// sections at once, each with a distinct verifiable value, so any inter-section offset error surfaces
// as one region reading another's word.
//
// A BRISC producer binds 2 named CRTAs + 1 tensor + 1 scratchpad + 2 common varargs (plus a DFB it
// produces into). The tensor uses dynamic_tensor_shape, which on an interleaved ROW-MAJOR tensor WIDENS
// its binding CRTA section from 1 word (base address) to 2 (base + aligned_page_size). That multi-word
// binding is the crux: every downstream section (scratchpad, varargs) must shift by the binding's true
// WIDTH, not by binding COUNT — the `.size()` trap where an offset bug hides and only bites once a
// relaxation makes a binding wider than one word. The producer reads one value from each section and
// stages seven into a DFB entry (w0=named0, w1=named1, w2=tensor base, w3=scratch base, w4=vararg0,
// w5=vararg1, w6=page size); an NCRISC consumer drains it to DRAM. The named/vararg sentinels and the
// page size are exact-checked; the tensor base and scratchpad base are real addresses (checked nonzero
// and mutually distinct from every sentinel and each other — a mis-offset read of an address region
// surfaces as a known value or the other base).
//
// Two phases:
//   SET    — SetProgramRunArgs installs all four sections; verify every region reads its own value
//            (the full-assembly offset math for all four sections, binding two words wide).
//   UPDATE — UpdateProgramRunArgs patches the named CRTAs + common varargs to NEW sentinels; verify the
//            new values land AND the tensor-binding + scratchpad slots are untouched. The vararg base
//            here is named + tensor-binding(2 words) + scratchpad section words — the A1 sum, now with a
//            MULTI-WORD binding in the middle (River's original had named=0, binding=0).
TEST_F(ProgramSpecHWTest, CrtaAllFourSectionsSetAndPartialUpdate) {
    auto mesh_device = devices_.at(0);

    constexpr uint32_t entry_size = 1024;  // bytes per DFB entry
    constexpr uint32_t num_entries = 4;    // DFB depth
    constexpr uint32_t kScratchpadBytes = 64;
    // io tensor is Shape{1,512} bf16 ROW_MAJOR → page = last dim (512) * 2 B = 1024 B (already aligned),
    // delivered as the binding's second CRTA word under dynamic_tensor_shape.
    constexpr uint32_t kExpectedPageSize = 512 * sizeof(uint16_t);

    const NodeCoord node{0, 0};

    // Distinct, non-trivial sentinels: a wrong-offset read is detectable, and none looks like an L1/DRAM base.
    constexpr uint32_t kNamed0Set = 0xA1A10000u;
    constexpr uint32_t kNamed1Set = 0xB2B20000u;
    constexpr uint32_t kVararg0Set = 0xC3C30000u;
    constexpr uint32_t kVararg1Set = 0xD4D40000u;
    constexpr uint32_t kNamed0Upd = 0xA1A1FFFFu;
    constexpr uint32_t kNamed1Upd = 0xB2B2FFFFu;
    constexpr uint32_t kVararg0Upd = 0xC3C3FFFFu;
    constexpr uint32_t kVararg1Upd = 0xD4D4FFFFu;

    // Output buffer holds one DFB entry (single page → single bank).
    auto output_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = entry_size},
        {.page_size = entry_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    auto& cq = mesh_device->mesh_command_queue();

    // Input tensor for the tensor-binding section (interleaved DRAM; only its base address is read here).
    auto tensor_layout = TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    auto tensor_spec = TensorSpec(Shape{1, 512}, tensor_layout);
    MeshTensor io_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);

    ProgramSpec spec;
    spec.name = "crta_all_four_sections";

    // Producer (BRISC): read one value from each CRTA section, stage seven into the DFB entry:
    //   w[0]=named0 w[1]=named1 w[2]=tensor base w[3]=scratch base w[4]=vararg0 w[5]=vararg1 w[6]=page size
    // The page size is the binding's SECOND CRTA word (dynamic_tensor_shape); reading it exercises the
    // extra binding word directly, and its presence shifts the scratchpad + vararg offsets.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    const uint32_t named0 = get_arg(args::named0);
    const uint32_t named1 = get_arg(args::named1);
    TensorAccessor acc(tensor::io);
    const uint32_t tensor_base = acc.get_bank_base_address();
    const uint32_t page_size = acc.get_aligned_page_size();  // binding's 2nd CRTA word (dynamic_tensor_shape)
    Scratchpad<uint32_t> pad(scratch::pad);
    const uint32_t scratch_base = pad.get_base_address();
    const uint32_t vararg0 = get_common_vararg(0);
    const uint32_t vararg1 = get_common_vararg(1);
    DataflowBuffer buf(dfb::stage);
    buf.reserve_back(1);
    volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)buf.get_write_ptr();
    w[0] = named0; w[1] = named1; w[2] = tensor_base; w[3] = scratch_base; w[4] = vararg0; w[5] = vararg1;
    w[6] = page_size;
    buf.push_back(1);
}
)"};
    producer.runtime_arg_schema.common_runtime_arg_names = {"named0", "named1"};
    producer.advanced_options = KernelAdvancedOptions{.num_common_runtime_varargs = 2};
    producer.scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    BindTensorParameterToKernel(producer, "io", "io");

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

    auto dfb = MakeMinimalDFB("stage", entry_size, num_entries);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"stage"}, "stage"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"stage"}, "stage"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}};
    // The UPDATE phase omits this tensor (retaining its bound tensor) — exercises the
    // "omitted tensor retained across partial update" path. dynamic_tensor_shape widens the binding to
    // two CRTA words (base + aligned_page_size) — the multi-word binding this test exists to stress.
    spec.tensor_parameters = {TensorParameter{
        .unique_id = TensorParamName{"io"},
        .spec = tensor_spec,
        .relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true}}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, MakeProgramFromSpec(*mesh_device, spec));
    Program& program = workload.get_programs().at(device_range);

    // Consumer's per-node RTAs (re-supplied on every set/update in this test).
    auto consumer_args = [&]() {
        return ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"dst_addr", output_buffer->address()}, {"bank_id", 0u}}),
        };
    };

    // Blocking launch, then read back the seven staged words.
    auto launch_and_read = [&]() {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
        std::vector<uint32_t> out;
        distributed::EnqueueReadMeshBuffer(cq, out, output_buffer, /*blocking=*/true);
        EXPECT_GE(out.size(), 7u);
        out.resize(7);
        return out;
    };

    // ---- SET phase: install all four sections; every region must read its own value. ----
    ProgramRunArgs set_params;
    set_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .common_runtime_arg_values = {{"named0", kNamed0Set}, {"named1", kNamed1Set}},
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = {kVararg0Set, kVararg1Set}},
        },
        consumer_args(),
    };
    set_params.tensor_args = {{TensorParamName{"io"}, TensorArgument{io_tensor}}};
    SetProgramRunArgs(program, set_params);

    std::vector<uint32_t> s = launch_and_read();
    const uint32_t tensor_base = s[2];
    const uint32_t scratch_base = s[3];
    EXPECT_EQ(s[0], kNamed0Set) << "named CRTA 0 read the wrong offset";
    EXPECT_EQ(s[1], kNamed1Set) << "named CRTA 1 read the wrong offset";
    // The binding's 2nd word: a wrong value here means the extra binding CRTA word was not delivered.
    EXPECT_EQ(s[6], kExpectedPageSize) << "tensor binding's aligned_page_size (2nd binding word) is wrong";
    EXPECT_EQ(s[4], kVararg0Set)
        << "common vararg 0 read the wrong offset — vararg base must be named + binding(2 words) + scratchpad";
    EXPECT_EQ(s[5], kVararg1Set) << "common vararg 1 read the wrong offset";
    // The two address regions must be real and distinct from every sentinel and each other — a mis-offset
    // read of either would surface here as a known sentinel or as the other base.
    for (uint32_t sentinel : {kNamed0Set, kNamed1Set, kVararg0Set, kVararg1Set}) {
        EXPECT_NE(tensor_base, sentinel) << "tensor-binding slot read a CRTA sentinel — section offset wrong";
        EXPECT_NE(scratch_base, sentinel) << "scratchpad slot read a CRTA sentinel — section offset wrong";
    }
    EXPECT_NE(tensor_base, 0u);
    EXPECT_NE(scratch_base, 0u);
    EXPECT_NE(tensor_base, scratch_base) << "tensor-binding and scratchpad slots collided — section offset wrong";

    // ---- UPDATE phase: partial-update named CRTAs + varargs; bindings/scratchpad must be untouched. ----
    ProgramRunArgs upd_params;
    upd_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .common_runtime_arg_values = {{"named0", kNamed0Upd}, {"named1", kNamed1Upd}},
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = {kVararg0Upd, kVararg1Upd}},
        },
        consumer_args(),
    };
    UpdateProgramRunArgs(program, upd_params);

    std::vector<uint32_t> u = launch_and_read();
    EXPECT_EQ(u[0], kNamed0Upd) << "partial update: named CRTA 0 landed at the wrong offset";
    EXPECT_EQ(u[1], kNamed1Upd) << "partial update: named CRTA 1 landed at the wrong offset";
    EXPECT_EQ(u[4], kVararg0Upd)
        << "partial update: vararg 0 landed at the wrong offset — the vararg base must be "
           "named + tensor-binding(2 words) + scratchpad section words (the A1 sum with a multi-word binding).";
    EXPECT_EQ(u[5], kVararg1Upd) << "partial update: vararg 1 landed at the wrong offset";
    // Not touched by this update — the omitted tensor's binding (both words) and the scratchpad must survive.
    EXPECT_EQ(u[2], tensor_base) << "tensor-binding base slot was clobbered by the named/vararg partial update";
    EXPECT_EQ(u[6], kExpectedPageSize) << "tensor-binding page-size slot was clobbered by the partial update";
    EXPECT_EQ(u[3], scratch_base) << "scratchpad slot was clobbered by the named/vararg partial update";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
