// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mxfp4.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::mxfp4_typecast {

static vector<uint32_t> run_mxfp4_typecast(
    IDevice* dev,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<uint32_t>& src_vec,
    uint32_t num_tiles) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t input_tile_size = tt::tile_size(input_fmt);
    uint32_t output_tile_size = tt::tile_size(output_fmt);

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = num_tiles * input_tile_size,
        .page_size = input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = output_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

    tt_metal::experimental::dfb::DataflowBufferConfig l1_input_dfb_config = {
        .entry_size = input_tile_size,
        .num_entries = 2,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .data_format = input_fmt};

    tt_metal::experimental::dfb::DataflowBufferConfig l1_output_dfb_config = {
        .entry_size = output_tile_size,
        .num_entries = 2,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .data_format = output_fmt};

    uint32_t l1_input_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_input_dfb_config);
    uint32_t l1_output_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_output_dfb_config);

    KernelHandle reader = tt_metal::experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {l1_input_dfb, /*use_dfbs=*/1}});

    KernelHandle writer = tt_metal::experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {l1_output_dfb, /*use_dfbs=*/1}});

    KernelHandle compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1, .compile_args = {num_tiles, /*use_dfbs=*/1}});

    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, l1_input_dfb, reader, compute);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, l1_output_dfb, compute, writer);

    detail::WriteToBuffer(src_buffer, src_vec);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    detail::LaunchProgram(dev, program, true);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

static vector<uint32_t> create_random_vector_of_mxfp4(
    uint32_t num_tiles, int rand_max_float, int seed, float offset = 0.0f) {
    constexpr uint32_t tile_elements = 1024;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(rand_max_float));

    vector<float> fp32_vec(num_tiles * tile_elements, 0.0f);
    for (float& val : fp32_vec) {
        val = dist(rng) + offset;
    }

    return pack_as_mxfp4_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true);
}

static vector<float> mxfp4_to_floats(const vector<uint32_t>& packed) {
    return unpack_mxfp4_tiles_into_float_vec(tt::stl::make_const_span(packed), /*row_major_output=*/false);
}

static vector<float> bf16_to_floats(const vector<uint32_t>& packed) {
    auto bf16_vec = unpack_uint32_vec_into_bfloat16_vec(packed);
    vector<float> floats;
    floats.reserve(bf16_vec.size());
    for (const auto& v : bf16_vec) {
        floats.push_back(static_cast<float>(v));
    }
    return floats;
}

static bool check_floats_close(const vector<float>& a, const vector<float>& b, float rtol, float atol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!is_close(a[i], b[i], rtol, atol)) {
            return false;
        }
    }
    return true;
}

}  // namespace unit_tests::llk::mxfp4_typecast

using namespace unit_tests::llk::mxfp4_typecast;

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarMxFp4ToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 16;
    auto src_vec = create_random_vector_of_mxfp4(num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);

    auto result_vec = run_mxfp4_typecast(dev, tt::DataFormat::MxFp4, tt::DataFormat::Float16_b, src_vec, num_tiles);

    auto src_floats = mxfp4_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
}

}  // namespace tt::tt_metal
