// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

// ReluType encoding (matches ckernel::ReluType)
enum PackReluMode : uint32_t {
    NO_RELU = 0,
    ZERO_RELU = 1,
    MIN_THRESHOLD_RELU = 2,
    MAX_THRESHOLD_RELU = 3,
};

// Pack relu config: mode in bits [1:0], bfloat16 threshold in bits [31:16]
static uint32_t make_relu_config(PackReluMode mode, float threshold = 0.0f) {
    uint16_t thresh_bf16 = std::bit_cast<uint16_t>(bfloat16(threshold));
    return (static_cast<uint32_t>(thresh_bf16) << 16) | static_cast<uint32_t>(mode);
}

// Run a pack relu test using the same infrastructure as test_direct.cpp's quasar path:
// direct_reader_unary.cpp → eltwise_copy_pack_relu.cpp → direct_writer_unary.cpp
static void run_pack_relu_test(IDevice* dev, uint32_t relu_config, const std::function<float(float)>& golden_fn) {
    Program program = CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig src_config{
        .device = dev, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(src_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    InterleavedBufferConfig dst_config{
        .device = dev, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto dst_dram_buffer = CreateBuffer(dst_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    KernelHandle reader;
    KernelHandle writer;
    KernelHandle compute;

    if (dev->arch() != ARCH::QUASAR) {
        uint32_t src0_cb_index = tt::CBIndex::c_0;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t output_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        reader = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = {src0_cb_index}});

        writer = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {output_cb_index}});

        compute = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = {num_tiles}, .defines = {{"PACK_RELU", "1"}}});
    } else {
        // Same DFB config as test_direct.cpp quasar path
        tt_metal::experimental::dfb::DataflowBufferConfig l1_input_dfb_config = {
            .entry_size = single_tile_size,
            .num_entries = 2,
            .num_producers = 1,
            .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .num_consumers = 1,
            .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = true,
            .data_format = tt::DataFormat::Float16_b};
        tt_metal::experimental::dfb::DataflowBufferConfig l1_output_dfb_config = {
            .entry_size = single_tile_size,
            .num_entries = 2,
            .num_producers = 1,
            .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .num_consumers = 1,
            .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = true,
            .data_format = tt::DataFormat::Float16_b};

        uint32_t l1_input_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_input_dfb_config);
        uint32_t l1_output_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_output_dfb_config);

        // Same reader/writer as test_direct.cpp
        reader = tt_metal::experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {l1_input_dfb}});

        writer = tt_metal::experimental::quasar::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {l1_output_dfb}});

        compute = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::experimental::quasar::QuasarComputeConfig{
                .num_threads_per_cluster = 1, .compile_args = {num_tiles}, .defines = {{"PACK_RELU", "1"}}});

        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, l1_input_dfb, reader, compute);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, l1_output_dfb, compute, writer);
    }

    // Stimulus: random bfloat16 in [-1, 1]
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 1.0f, 0xCAFE);
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, reader, core, {dram_buffer_src_addr, 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dram_buffer_dst_addr, 0, num_tiles});
    SetRuntimeArgs(program, compute, core, {relu_config});

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Build golden
    std::vector<uint32_t> golden(src_vec.size());
    for (size_t i = 0; i < src_vec.size(); ++i) {
        auto [lo_bf, hi_bf] = unpack_two_bfloat16_from_uint32(src_vec[i]);
        float lo_f = golden_fn(static_cast<float>(lo_bf));
        float hi_f = golden_fn(static_cast<float>(hi_bf));
        golden[i] = pack_two_bfloat16_into_uint32({bfloat16(lo_f), bfloat16(hi_f)});
    }

    auto comparison_function = [](float a, float b) {
        const float atol = 0.02f;
        const float rtol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };

    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, golden, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// ZERO_RELU: max(0, x)
TEST_F(MeshDeviceSingleCardFixture, QuasarPackReluZero) {
    IDevice* dev = devices_[0]->get_devices()[0];
    run_pack_relu_test(dev, make_relu_config(ZERO_RELU), [](float x) { return std::max(0.0f, x); });
}

// MIN_THRESHOLD_RELU: x <= threshold ? 0 : x (threshold = 0.25)
TEST_F(MeshDeviceSingleCardFixture, QuasarPackReluMinThreshold) {
    IDevice* dev = devices_[0]->get_devices()[0];
    const float threshold = 0.25f;
    run_pack_relu_test(dev, make_relu_config(MIN_THRESHOLD_RELU, threshold), [threshold](float x) {
        return x <= threshold ? 0.0f : x;
    });
}

// MAX_THRESHOLD_RELU: clamp to [0, threshold] (threshold = 0.5)
TEST_F(MeshDeviceSingleCardFixture, QuasarPackReluMaxThreshold) {
    IDevice* dev = devices_[0]->get_devices()[0];
    const float threshold = 0.5f;
    run_pack_relu_test(dev, make_relu_config(MAX_THRESHOLD_RELU, threshold), [threshold](float x) {
        return x < 0.0f ? 0.0f : std::min(x, threshold);
    });
}
