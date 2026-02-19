// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <algorithm>
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
#include <tt-logger/tt-logger.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, TransposeHC) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();
    constexpr bool multibank = true;

    CoreCoord core = {0, 0};

    constexpr uint32_t subtile_elements = 16U;
    constexpr uint32_t subtile_line_bytes = subtile_elements * 2U;
    constexpr uint32_t tile_elements = 32U;
    constexpr uint32_t tile_size = tile_elements * tile_elements;

    const std::vector<uint32_t> shape = {2U, tile_elements * 3U, tile_elements * 5U, tile_elements * 2U};
    const uint32_t num_elements =
        std::accumulate(std::cbegin(shape), std::cend(shape), 1U, [](const uint32_t left, const uint32_t right) {
            return left * right;
        });
    const uint32_t num_tensor_tiles = num_elements / tile_size;

    const uint32_t single_tile_bytes = 2U * tile_size;
    const uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles;
    const uint32_t page_size = (!multibank) ? dram_buffer_bytes : single_tile_bytes;

    const InterleavedBufferConfig dram_config = {
        .device = dev, .size = dram_buffer_bytes, .page_size = page_size, .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config);
    const uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    const uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
    const uint32_t alignment = dst_dram_buffer->alignment();
    const bool misaligned = alignment > subtile_line_bytes;

    const uint32_t src0_cb_index = 0U;
    const uint32_t num_buffer_tiles = 2U;
    const CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_bytes);
    CreateCircularBuffer(program, core, cb_src0_config);

    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_bytes);
    CreateCircularBuffer(program, core, cb_output_config);

    if (misaligned) {
        const uint32_t src1_cb_index = 1U;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(alignment, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, alignment);
        CreateCircularBuffer(program, core, cb_src1_config);
    }

    const uint32_t W = shape[3U], H = shape[2U], C = shape[1U], N = shape[0U];
    const uint32_t HW = H * W;
    const uint32_t CHW = C * H * W;
    std::vector<uint32_t> reader_compile_time_args;
    reader_compile_time_args.emplace_back(alignment);
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    auto reader_kernel = CreateKernel(
        program,
        multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc_8bank.cpp"
                  : "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    auto unary_writer_kernel = CreateKernel(
        program,
        multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                  : "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> compute_kernel_args = {num_tensor_tiles};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100U, 0x1234);
    auto src_4f_16 = u16_from_u32_vector(src0_vec);
    detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    SetRuntimeArgs(program, reader_kernel, core, {dram_buffer_src0_addr, 0U, W, H, C, HW, N, CHW});
    SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, 0U, num_tensor_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    auto comparison_function = [](float a, float b) {
        const float rtol{0.001f};
        const float atol{1e-3f};
        const float maxabs{std::fmaxf(std::abs(a), std::abs(b))};
        float absdiff{std::abs(a - b)};
        return (absdiff <= atol) || (absdiff < rtol * maxabs);
    };

    std::vector<uint16_t> src_linear =
        convert_layout<uint16_t>(src_4f_16, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<uint16_t> gold_reduced = gold_transpose_hc(src_linear, shape);

    std::vector<uint32_t> shapeR = {shape[0U], shape[2U], shape[1U], shape[3U]};
    const auto gold_16_4f =
        convert_layout<uint16_t>(gold_reduced, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    const auto gold_4f_u32 = u32_from_u16_vector(gold_16_4f);

    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
