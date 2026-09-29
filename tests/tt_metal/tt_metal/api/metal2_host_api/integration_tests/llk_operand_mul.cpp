// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end LLK operands on Blackhole silicon: mul_tiles with operands from a DFB, a
// LocalTensorAccessor and a Scratchpad.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>

#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/integration_tests/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecHWTest;

// ============================================================================
// LLKOperand integration: mul_tiles through each memory object (SPEC Parts II–III)
// ============================================================================
//
// One Blackhole test per source of LLKOperand (DFB / LocalTensorAccessor / Scratchpad). Each pipes a
// single Float16_b tile through experimental::mul_tiles and checks the product. Address math is
// covered implicitly by a correct product; the DFB test also keeps the MATH-thread front/back == 0 check.

namespace {

constexpr uint32_t kLlkTileBytes = 2048;  // one Float16_b 32x32 tile
constexpr uint32_t kLlkReportAddr = 100 * 1024;

TensorParameter MakeDramTileTensorParameter(std::string name, const Shape& logical_shape) {
    auto page_config = PageConfig(Layout::TILE);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    return TensorParameter{
        .unique_id = TensorParamName{std::move(name)},
        .spec = TensorSpec(logical_shape, std::move(tensor_layout)),
    };
}

void WriteConstantBf16(distributed::MeshDevice& mesh_device, MeshTensor& tensor, float value) {
    std::vector<bfloat16> data(tensor.logical_volume(), bfloat16(value));
    mesh_device.mesh_command_queue().enqueue_write_tensor(
        HostTensor::from_vector(std::move(data), tensor.tensor_spec()), tensor);
}

std::vector<bfloat16> ReadBf16(distributed::MeshDevice& mesh_device, const MeshTensor& tensor) {
    return mesh_device.mesh_command_queue().enqueue_read_tensor(tensor).to_vector<bfloat16>();
}

}  // namespace

// Elementwise mul of one Float16_b tile (C = A * B) through DFB operands: DRAM A/B are streamed into
// in0/in1, compute does mul_tiles(in0.front, in1.front) and packs to out.back, and a DM writer drains
// out to DRAM C. Proves LLKOperandFrom<dfb::…> + DataflowBuffer::front/back address the FIFO correctly.
TEST_F(ProgramSpecHWTest, LLKOperandDfbMul) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only: LLKOperand / 2.0 compute API does not compile on other architectures";
    }

    const NodeCoord node{0, 0};

    auto a_param = MakeDramTileTensorParameter("a", Shape{32, 32});
    auto b_param = MakeDramTileTensorParameter("b", Shape{32, 32});
    auto c_param = MakeDramTileTensorParameter("c", Shape{32, 32});
    MeshTensor a_tensor = MeshTensor::allocate_on_device(*mesh_device, a_param.spec);
    MeshTensor b_tensor = MeshTensor::allocate_on_device(*mesh_device, b_param.spec);
    MeshTensor c_tensor = MeshTensor::allocate_on_device(*mesh_device, c_param.spec);

    auto reader = MakeMinimalGen1DMKernel("reader", DataMovementProcessor::RISCV_0);
    reader.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;
    TensorAccessor a(tensor::a);
    TensorAccessor b(tensor::b);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);

    in0.reserve_back(1);
    noc.async_read(a, in0, in0.get_entry_size(), {.page_id = 0}, {});
    noc.async_read_barrier();
    in0.push_back(1);

    in1.reserve_back(1);
    noc.async_read(b, in1, in1.get_entry_size(), {.page_id = 0}, {});
    noc.async_read_barrier();
    in1.push_back(1);
}
)"};
    BindTensorParameterToKernel(reader, "a", "a");
    BindTensorParameterToKernel(reader, "b", "b");

    auto writer = MakeMinimalGen1DMKernel("writer", DataMovementProcessor::RISCV_1);
    writer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;
    TensorAccessor c(tensor::c);
    DataflowBuffer out(dfb::out);

    out.wait_front(1);
    noc.async_write(out, c, out.get_entry_size(), {}, {.page_id = 0});
    noc.async_write_barrier();
    out.pop_front(1);
}
)"};
    BindTensorParameterToKernel(writer, "c", "c");

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{R"(
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);

    using AOp = LLKOperandFrom<dfb::in0>;
    using BOp = LLKOperandFrom<dfb::in1>;
    using OutOp = LLKOperandFrom<dfb::out>;

    compute_kernel_hw_startup(AOp{0}, BOp{0}, OutOp{0});
    ckernel::experimental::mul_init(AOp{0}, /*acc_to_dest=*/false);

    in0.wait_front(1);
    in1.wait_front(1);
    out.reserve_back(1);

    tile_regs_acquire();
    ckernel::experimental::mul_tiles(in0.front<AOp>(), in1.front<BOp>(), 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    ckernel::experimental::pack_tile(out.back<OutOp>(), 0, 0);
    tile_regs_release();

#if defined(TRISC_MATH)
    const uint32_t report_addr = get_arg(args::report_addr);
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = in0.front<AOp>().l1_address;
    report[1] = in0.back<AOp>().l1_address;
#endif

    in0.pop_front(1);
    in1.pop_front(1);
    out.push_back(1);
}
)"};
    compute.runtime_arg_schema.runtime_arg_names = {"report_addr"};

    auto in0 = MakeMinimalDFB("in0", kLlkTileBytes, /*num_entries=*/2);
    in0.data_format_metadata = tt::DataFormat::Float16_b;
    auto in1 = MakeMinimalDFB("in1", kLlkTileBytes, /*num_entries=*/2);
    in1.data_format_metadata = tt::DataFormat::Float16_b;
    auto out = MakeMinimalDFB("out", kLlkTileBytes, /*num_entries=*/2);
    out.data_format_metadata = tt::DataFormat::Float16_b;

    reader.dfb_bindings.push_back(ProducerOf(DFBSpecName{"in0"}, "in0"));
    reader.dfb_bindings.push_back(ProducerOf(DFBSpecName{"in1"}, "in1"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"in0"}, "in0"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"in1"}, "in1"));
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out"}, "out"));
    writer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out"}, "out"));

    ProgramSpec spec;
    spec.name = "llk_operand_dfb_mul";
    spec.kernels = {reader, writer, compute};
    spec.dataflow_buffers = {in0, in1, out};
    spec.tensor_parameters = {a_param, b_param, c_param};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"reader", "writer", "compute"})};

    auto workload = MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;

    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute"},
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"report_addr", kLlkReportAddr}}),
    }};
    params.tensor_args = {
        {TensorParamName{"a"}, TensorArgument{a_tensor}},
        {TensorParamName{"b"}, TensorArgument{b_tensor}},
        {TensorParamName{"c"}, TensorArgument{c_tensor}},
    };
    SetProgramRunArgs(program, params);

    constexpr float kA = 2.0f;
    constexpr float kB = 3.0f;
    WriteConstantBf16(*mesh_device, a_tensor, kA);
    WriteConstantBf16(*mesh_device, b_tensor, kB);
    WriteConstantBf16(*mesh_device, c_tensor, 0.0f);

    std::vector<uint32_t> zero_report(2, 0u);
    slow_dispatch::WriteToL1(*mesh_device, node, kLlkReportAddr, zero_report);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<uint32_t> reported;
    slow_dispatch::ReadFromL1(*mesh_device, node, kLlkReportAddr, 2 * sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 2u);
    EXPECT_EQ(reported[0], 0u) << "MATH front must be 0";
    EXPECT_EQ(reported[1], 0u) << "MATH back must be 0";

    const auto output = ReadBf16(*mesh_device, c_tensor);
    EXPECT_EQ(output, std::vector<bfloat16>(c_tensor.logical_volume(), bfloat16(kA * kB)));
}

// Elementwise mul of one Float16_b tile already resident in L1: compute-only, no DFB/DM. Host writes
// sharded tensors A and B; compute does mul_tiles(a.operand, b.operand) and packs into C via
// LocalTensorAccessor::operand. Proves LLKOperandFrom<tensor::…> derives format/shape from TensorSpec.
TEST_F(ProgramSpecHWTest, LLKOperandLocalTensorMul) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only: LLKOperand / 2.0 compute API does not compile on other architectures";
    }

    const NodeCoord node{0, 0};

    auto a_param = MakeShardedTensorParameter("a", Shape{32, 32}, {32, 32}, /*num_cores=*/1);
    auto b_param = MakeShardedTensorParameter("b", Shape{32, 32}, {32, 32}, /*num_cores=*/1);
    auto c_param = MakeShardedTensorParameter("c", Shape{32, 32}, {32, 32}, /*num_cores=*/1);
    MeshTensor a_tensor = MeshTensor::allocate_on_device(*mesh_device, a_param.spec);
    MeshTensor b_tensor = MeshTensor::allocate_on_device(*mesh_device, b_param.spec);
    MeshTensor c_tensor = MeshTensor::allocate_on_device(*mesh_device, c_param.spec);

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{R"(
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/llk_operand_from_tokens.h"
#include "api/tensor/local_tensor_accessor.h"

void kernel_main() {
    LocalTensorAccessor<uint32_t> a(tensor::a);
    LocalTensorAccessor<uint32_t> b(tensor::b);
    LocalTensorAccessor<uint32_t> c(tensor::c);

    using AOp = LLKOperandFrom<tensor::a>;
    using BOp = LLKOperandFrom<tensor::b>;
    using COp = LLKOperandFrom<tensor::c>;

    compute_kernel_hw_startup(AOp{0}, BOp{0}, COp{0});
    ckernel::experimental::mul_init(AOp{0}, /*acc_to_dest=*/false);

    tile_regs_acquire();
    ckernel::experimental::mul_tiles(a.operand<AOp>(), b.operand<BOp>(), 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    ckernel::experimental::pack_tile(c.operand<COp>(), 0, 0);
    tile_regs_release();
}
)"};
    BindTensorParameterToKernel(compute, "a", "a");
    BindTensorParameterToKernel(compute, "b", "b");
    BindTensorParameterToKernel(compute, "c", "c");

    ProgramSpec spec;
    spec.name = "llk_operand_lta_mul";
    spec.kernels = {compute};
    spec.tensor_parameters = {a_param, b_param, c_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"compute"})};

    auto workload = MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;

    ProgramRunArgs params;
    params.tensor_args = {
        {TensorParamName{"a"}, TensorArgument{a_tensor}},
        {TensorParamName{"b"}, TensorArgument{b_tensor}},
        {TensorParamName{"c"}, TensorArgument{c_tensor}},
    };
    SetProgramRunArgs(program, params);

    constexpr float kA = 2.0f;
    constexpr float kB = 4.0f;
    WriteConstantBf16(*mesh_device, a_tensor, kA);
    WriteConstantBf16(*mesh_device, b_tensor, kB);
    WriteConstantBf16(*mesh_device, c_tensor, 0.0f);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

    const auto output = ReadBf16(*mesh_device, c_tensor);
    EXPECT_EQ(output, std::vector<bfloat16>(c_tensor.logical_volume(), bfloat16(kA * kB)));
}

// Elementwise mul whose operands live in a Scratchpad. The UNPACK thread fills pad_in with constant
// tiles A@0 and B@1 (it is the only thread that reads pad_in), then compute does
// mul_tiles(pad_in, pad_in, 0, 1) and packs the result to both pad_out and the out DFB; a DM writer
// drains out to DRAM C. Proves Scratchpad::operand as an LLK source and pack target without a
// read-after-pack hazard.
TEST_F(ProgramSpecHWTest, LLKOperandScratchpadMul) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only: LLKOperand / 2.0 compute API does not compile on other architectures";
    }

    const NodeCoord node{0, 0};

    auto c_param = MakeDramTileTensorParameter("c", Shape{32, 32});
    MeshTensor c_tensor = MeshTensor::allocate_on_device(*mesh_device, c_param.spec);

    auto writer = MakeMinimalGen1DMKernel("writer", DataMovementProcessor::RISCV_1);
    writer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;
    TensorAccessor c(tensor::c);
    DataflowBuffer out(dfb::out);

    out.wait_front(1);
    noc.async_write(out, c, out.get_entry_size(), {}, {.page_id = 0});
    noc.async_write_barrier();
    out.pop_front(1);
}
)"};
    BindTensorParameterToKernel(writer, "c", "c");

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{R"(
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "api/scratchpad.h"

// Two packed Float16_b values per word: bf16(2.0) = 0x4000, bf16(5.0) = 0x40A0.
constexpr uint32_t kAWord = 0x40004000u;
constexpr uint32_t kBWord = 0x40A040A0u;
constexpr uint32_t kTileWords = 2048 / sizeof(uint32_t);

void kernel_main() {
    Scratchpad<uint32_t> pad_in(scratch::pad_in);
    Scratchpad<uint32_t> pad_out(scratch::pad_out);
    DataflowBuffer out(dfb::out);

    using InOp = LLKOperandFrom<scratch::pad_in>;
    using OutOp = LLKOperandFrom<scratch::pad_out>;
    using DfbOutOp = LLKOperandFrom<dfb::out>;

#if defined(TRISC_UNPACK)
    for (uint32_t i = 0; i < kTileWords; ++i) {
        pad_in[i] = kAWord;
        pad_in[kTileWords + i] = kBWord;
    }
    // Drain the fill stores before the unpacker reads pad_in.
    asm volatile("fence" ::: "memory");
#endif

    compute_kernel_hw_startup(InOp{0}, InOp{0}, OutOp{0});
    ckernel::experimental::mul_init(InOp{0}, /*acc_to_dest=*/false);

    out.reserve_back(1);

    tile_regs_acquire();
    ckernel::experimental::mul_tiles(pad_in.operand<InOp>(), pad_in.operand<InOp>(), 0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    ckernel::experimental::pack_tile(pad_out.operand<OutOp>(), 0, 0);
    ckernel::experimental::pack_tile(out.back<DfbOutOp>(), 0, 0);
    tile_regs_release();

    out.push_back(1);
}
)"};
    compute.scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad_in"}, .accessor_name = "pad_in"});
    compute.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"pad_out"}, .accessor_name = "pad_out"});

    auto out = MakeMinimalDFB("out", kLlkTileBytes, /*num_entries=*/1);
    out.data_format_metadata = tt::DataFormat::Float16_b;
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"out"}, "out"));
    writer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"out"}, "out"));

    ProgramSpec spec;
    spec.name = "llk_operand_scratchpad_mul";
    spec.kernels = {writer, compute};
    spec.dataflow_buffers = {out};
    spec.scratchpads = {
        ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad_in"},
            .size_per_node = 2 * kLlkTileBytes,
            .data_format_metadata = tt::DataFormat::Float16_b,
        },
        ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad_out"},
            .size_per_node = kLlkTileBytes,
            .data_format_metadata = tt::DataFormat::Float16_b,
        },
    };
    spec.tensor_parameters = {c_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"writer", "compute"})};

    auto workload = MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;

    ProgramRunArgs params;
    params.tensor_args = {
        {TensorParamName{"c"}, TensorArgument{c_tensor}},
    };
    SetProgramRunArgs(program, params);

    // Must match kAWord / kBWord in the compute kernel.
    constexpr float kA = 2.0f;
    constexpr float kB = 5.0f;
    WriteConstantBf16(*mesh_device, c_tensor, 0.0f);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

    const auto output = ReadBf16(*mesh_device, c_tensor);
    EXPECT_EQ(output, std::vector<bfloat16>(c_tensor.logical_volume(), bfloat16(kA * kB)));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
