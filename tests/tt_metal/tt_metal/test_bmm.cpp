// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"
#include "impl/program/program_impl.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

struct BmmParams {
    uint32_t Mt, Kt, Nt;
    uint32_t B_total;       // total batch count (buffer sizing + validation)
    uint32_t B_per_core;    // batch count per core (kernel runtime args)
    uint32_t num_threads = 2;
    uint32_t num_input_tiles = 4;
    uint32_t num_output_tiles = 4;
    uint32_t single_tile_size = 2 * 1024;
};

struct BmmBuffers {
    std::shared_ptr<Buffer> src0, src1, dst;
};

BmmBuffers create_bmm_dram_buffers(IDevice* dev, const BmmParams& p) {
    const uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;
    const uint32_t bytesC = p.single_tile_size * p.Mt * p.Nt * p.B_total;
    return {
        CreateBuffer({.device = dev, .size = bytesA, .page_size = p.single_tile_size, .buffer_type = BufferType::DRAM}),
        CreateBuffer({.device = dev, .size = bytesB, .page_size = p.single_tile_size, .buffer_type = BufferType::DRAM}),
        CreateBuffer({.device = dev, .size = bytesC, .page_size = p.single_tile_size, .buffer_type = BufferType::DRAM}),
    };
}

struct BmmDFBHandles {
    uint32_t src0, src1, dst;
};

template <typename CoreSpec>
BmmDFBHandles create_bmm_quasar_dfbs(Program& program, const CoreSpec& cores, const BmmParams& p) {
    using namespace tt_metal::experimental::dfb;
    DataflowBufferConfig src0_cfg = {
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles,
        .num_producers = p.num_threads,
        .pap = AccessPattern::STRIDED,
        .num_consumers = p.num_threads,
        .cap = AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};
    DataflowBufferConfig src1_cfg = {
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles,
        .num_producers = p.num_threads,
        .pap = AccessPattern::STRIDED,
        .num_consumers = p.num_threads,
        .cap = AccessPattern::ALL,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};
    DataflowBufferConfig dst_cfg = {
        .entry_size = p.single_tile_size,
        .num_entries = p.num_output_tiles,
        .num_producers = p.num_threads,
        .pap = AccessPattern::STRIDED,
        .num_consumers = p.num_threads,
        .cap = AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};
    return {
        CreateDataflowBuffer(program, cores, src0_cfg),
        CreateDataflowBuffer(program, cores, src1_cfg),
        CreateDataflowBuffer(program, cores, dst_cfg),
    };
}

struct BmmKernelHandles {
    KernelHandle reader, writer, compute;
};

template <typename CoreSpec>
BmmKernelHandles create_bmm_quasar_kernels(
    Program& program,
    const CoreSpec& cores,
    const BmmParams& p,
    const std::vector<uint32_t>& reader_cta,
    const std::vector<uint32_t>& writer_cta) {
    using namespace tt_metal::experimental::quasar;
    return {
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
            cores,
            QuasarDataMovementConfig{.num_threads_per_cluster = p.num_threads, .compile_args = reader_cta}),
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
            cores,
            QuasarDataMovementConfig{.num_threads_per_cluster = p.num_threads, .compile_args = writer_cta}),
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
            cores,
            QuasarComputeConfig{
                .num_threads_per_cluster = p.num_threads,
                .compile_args = {p.B_per_core, p.Mt, p.Kt, p.Nt}}),
    };
}

void bind_bmm_dfbs(Program& program, const BmmDFBHandles& dfbs, const BmmKernelHandles& kernels) {
    using namespace tt_metal::experimental::dfb;
    BindDataflowBufferToProducerConsumerKernels(program, dfbs.src0, kernels.reader, kernels.compute);
    BindDataflowBufferToProducerConsumerKernels(program, dfbs.src1, kernels.reader, kernels.compute);
    BindDataflowBufferToProducerConsumerKernels(program, dfbs.dst, kernels.compute, kernels.writer);
}

bool validate_bmm_result(
    const BmmParams& p,
    const std::vector<uint32_t>& src0_vec,
    const std::vector<uint32_t>& src1_vec,
    const std::vector<uint32_t>& result_vec,
    int* argfail) {
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.05f;
        const float atol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };
    vector<uint32_t> shapeA = {1, p.B_total, p.Mt * 32, p.Kt * 32};
    vector<uint32_t> shapeB = {1, p.B_total, p.Kt * 32, p.Nt * 32};
    vector<uint32_t> shapeC = {1, p.B_total, p.Mt * 32, p.Nt * 32};
    auto u16_src0 = u16_from_u32_vector(src0_vec);
    auto u16_src1 = u16_from_u32_vector(src1_vec);
    auto src0_linear =
        convert_layout<uint16_t>(u16_src0, shapeA, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    auto src1_linear =
        convert_layout<uint16_t>(u16_src1, shapeB, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    auto ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);
    auto gold = u32_from_u16_vector(
        convert_layout<uint16_t>(ref_bmm, shapeC, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));
    return packed_uint32_t_vector_comparison(result_vec, gold, comparison_function, argfail);
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, Bmm) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    BmmParams p;
    if (dev->arch() != ARCH::QUASAR) {
        p.Mt = 4; p.Kt = 2; p.Nt = 3;
        p.B_total = 2; p.B_per_core = 2;
        p.num_input_tiles = 2; p.num_output_tiles = 2;
        p.num_threads = 1;
    } else {
        p.Mt = 2; p.Kt = 2; p.Nt = 2;
        p.B_total = 1; p.B_per_core = 1;
        p.num_threads = 2;
    }

    auto bufs = create_bmm_dram_buffers(dev, p);
    const uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;

    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(bufs.src0).append_to(reader_cta);
    TensorAccessorArgs(bufs.src1).append_to(reader_cta);
    std::vector<uint32_t> writer_cta;
    TensorAccessorArgs(bufs.dst).append_to(writer_cta);

    BmmKernelHandles kernels;
    if (dev->arch() != ARCH::QUASAR) {
        uint32_t src0_cb_index = 0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                p.num_input_tiles * p.single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, p.single_tile_size));

        uint32_t src1_cb_index = 1;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                p.num_input_tiles * p.single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, p.single_tile_size));

        uint32_t output_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                p.num_output_tiles * p.single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, p.single_tile_size));

        kernels.reader = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_cta});
        kernels.writer = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_cta});
        kernels.compute = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
            core,
            ComputeConfig{.compile_args = {p.B_per_core, p.Mt, p.Kt, p.Nt}});
    } else {
        auto dfbs = create_bmm_quasar_dfbs(program, core, p);
        kernels = create_bmm_quasar_kernels(program, core, p, reader_cta, writer_cta);
        bind_bmm_dfbs(program, dfbs, kernels);
    }

    auto src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
    auto src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
    detail::WriteToBuffer(bufs.src0, src0_vec);
    detail::WriteToBuffer(bufs.src1, src1_vec);

    constexpr uint32_t do_bcast = 0;
    SetRuntimeArgs(
        program,
        kernels.reader,
        core,
        {bufs.src0->address(), bufs.src1->address(), p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, do_bcast, 0u});
    SetRuntimeArgs(
        program,
        kernels.writer,
        core,
        {bufs.dst->address(), 0u, p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, 0u});

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(bufs.dst, result_vec);

    int argfail = -1;
    bool pass = validate_bmm_result(p, src0_vec, src1_vec, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// This needs to be a separate test because we don't have a way of querying the correct compute grid size
// when running a multi-neo emu/sim build. Otherwise its the same test with batch split across nodes.
TEST_F(MeshDeviceSingleCardFixture, BmmMultinode) {
    IDevice* dev = devices_[0]->get_devices()[0];
    if (dev->arch() != ARCH::QUASAR) {
        GTEST_SKIP();
    }

    Program program = CreateProgram();
    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {1, 0};
    CoreRange core_range = {core0, core1};

    BmmParams p;
    p.Mt = 2; p.Kt = 2; p.Nt = 2;
    p.B_total = 2;      // total batches across both cores
    p.B_per_core = 1;   // each core computes exactly one batch
    p.num_threads = 2;

    auto bufs = create_bmm_dram_buffers(dev, p);
    const uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;

    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(bufs.src0).append_to(reader_cta);
    TensorAccessorArgs(bufs.src1).append_to(reader_cta);
    std::vector<uint32_t> writer_cta;
    TensorAccessorArgs(bufs.dst).append_to(writer_cta);

    // Create DFBs on the full core range — each core gets its own independent DFB instances
    auto dfbs = create_bmm_quasar_dfbs(program, core_range, p);
    auto kernels = create_bmm_quasar_kernels(program, core_range, p, reader_cta, writer_cta);
    bind_bmm_dfbs(program, dfbs, kernels);

    auto src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
    auto src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
    detail::WriteToBuffer(bufs.src0, src0_vec);
    detail::WriteToBuffer(bufs.src1, src1_vec);

    constexpr uint32_t do_bcast = 0;
    // core0 handles batch 0, core1 handles batch 1 (batch_offset = core index)
    SetRuntimeArgs(
        program,
        kernels.reader,
        core0,
        {bufs.src0->address(), bufs.src1->address(), p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, do_bcast, 0u});
    SetRuntimeArgs(
        program,
        kernels.writer,
        core0,
        {bufs.dst->address(), 0u, p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, 0u});

    SetRuntimeArgs(
        program,
        kernels.reader,
        core1,
        {bufs.src0->address(), bufs.src1->address(), p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, do_bcast, 1u});
    SetRuntimeArgs(
        program,
        kernels.writer,
        core1,
        {bufs.dst->address(), 0u, p.Mt, p.Kt, p.Nt, p.Mt * p.Kt, p.Kt * p.Nt, p.B_per_core, 1u});

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(bufs.dst, result_vec);

    int argfail = -1;
    bool pass = validate_bmm_result(p, src0_vec, src1_vec, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
