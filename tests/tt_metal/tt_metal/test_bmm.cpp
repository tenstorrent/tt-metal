// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include <tt-logger/tt-logger.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, Bmm) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t Mt = 4, Kt = 2, Nt = 3, B = 2;
    uint32_t num_tilesA = Mt * Kt * B;
    uint32_t num_tilesB = Kt * Nt * B;
    uint32_t num_tilesC = Mt * Nt * B;
    uint32_t bytesA = single_tile_size * num_tilesA;
    uint32_t bytesB = single_tile_size * num_tilesB;
    uint32_t bytesC = single_tile_size * num_tilesC;

    InterleavedBufferConfig src0_config{
        .device = dev, .size = bytesA, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = CreateBuffer(src0_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    InterleavedBufferConfig src1_config{
        .device = dev, .size = bytesB, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src1_dram_buffer = CreateBuffer(src1_config);
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();

    InterleavedBufferConfig dst_config{
        .device = dev, .size = bytesC, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto dst_dram_buffer = CreateBuffer(dst_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {B, Mt, Kt, Nt};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
    std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
    detail::WriteToBuffer(src0_dram_buffer, src0_vec);
    detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    uint32_t do_bcast = 0;
    SetRuntimeArgs(
        program,
        reader,
        core,
        {dram_buffer_src0_addr, dram_buffer_src1_addr, Mt, Kt, Nt, Mt * Kt, Kt * Nt, B, do_bcast});
    SetRuntimeArgs(program, writer, core, {dram_buffer_dst_addr, 0, Mt, Kt, Nt, Mt * Kt, Kt * Nt, B});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.05f;
        const float atol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };

    vector<uint32_t> shapeA = {1, B, Mt * 32, Kt * 32};
    vector<uint32_t> shapeB = {1, B, Kt * 32, Nt * 32};
    vector<uint32_t> shapeC = {1, B, Mt * 32, Nt * 32};
    auto u16_src0_vec = u16_from_u32_vector(src0_vec);
    auto u16_src1_vec = u16_from_u32_vector(src1_vec);
    vector<uint16_t> src0_linear =
        convert_layout<uint16_t>(u16_src0_vec, shapeA, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    vector<uint16_t> src1_linear =
        convert_layout<uint16_t>(u16_src1_vec, shapeB, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    vector<uint16_t> ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);

    auto gold_4f_u32 = u32_from_u16_vector(
        convert_layout<uint16_t>(ref_bmm, shapeC, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
