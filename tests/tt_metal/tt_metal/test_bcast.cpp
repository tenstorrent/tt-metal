// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

const char* get_reader_name(bool multibank, BcastDim::Enum bcast_dim) {
    TT_FATAL(multibank && "Only multibank is supported correctly.", "Error");
    if (bcast_dim == BcastDim::H) {
        return multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_h_8bank.cpp"
                         : "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_h.cpp";
    }
    if (bcast_dim == BcastDim::W) {
        return multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_w_8bank.cpp"
                         : "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_w.cpp";
    }
    if (bcast_dim == BcastDim::HW) {
        return multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_hw_8bank.cpp"
                         : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp";
    }
    TT_THROW("Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastDim::H: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_h.cpp";
        case BcastDim::W: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_w.cpp";
        case BcastDim::HW: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_hw.cpp";
        default: TT_THROW("Unexpected bcast_dim!");
    }
    return "";
}

const char* bdim_to_log_string[] = {"", "BCAST_H", "BCAST_W", "", "BCAST_HW"};
const char* op_id_to_op_define[] = {"add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"};
const char* op_id_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
const char* bdim_to_llkdim_define[] = {"", "BroadcastType::ROW", "BroadcastType::COL", "", "BroadcastType::SCALAR"};
const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};

void run_bcast_test(IDevice* dev, BcastDim::Enum bcast_dim, BcastOp::Enum bcast_op) {
    bool multibank = true;

    log_info(LogTest, "=============================================================");
    log_info(
        LogTest,
        "======= Running bcast test for bdim={}, op={}",
        bdim_to_log_string[bcast_dim],
        op_id_to_op_name[bcast_op]);

    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    vector<uint32_t> shape = {2, 4, 2 * TILE_HEIGHT, 3 * TILE_WIDTH};
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0], N = shape[0], C = shape[1];
    TT_FATAL(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0, "Error");
    TT_FATAL(H > 0 && W > 0 && NC > 0, "Error");
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t num_tensor_tiles = NC * H * W / (32 * 32);

    uint32_t single_tile_bytes = 2 * 1024;
    uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles;

    uint32_t page_size = single_tile_bytes;
    if (!multibank) {
        page_size = dram_buffer_bytes;
    }

    InterleavedBufferConfig buff_config{
        .device = dev, .size = dram_buffer_bytes, .page_size = page_size, .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_buffer_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_bytes);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_bytes);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_buffer_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_bytes);
    CreateCircularBuffer(program, core, cb_output_config);

    vector<uint16_t> tiled_bcast_values;
    vector<uint16_t> ref_bcast_values;
    float bcast_1value = 10.0f;
    unsigned num_bcast_tiles = 0;

    // build the constant tiles to be broadcast
    if (bcast_dim == BcastDim::HW) {
        ref_bcast_values.resize(NC, 0);
        vector<uint32_t> ref_bcast_shape_with_tile_padding = {N, C, TILE_HEIGHT, TILE_WIDTH};
        vector<uint16_t> ref_bcast_values_with_tile_padding;
        ref_bcast_values_with_tile_padding.resize(NC * TILE_HEIGHT * TILE_WIDTH, 0);
        for (uint32_t j = 0; j < NC; j++) {
            auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
            ref_bcast_values[j] = val;
            ref_bcast_values_with_tile_padding[j * TILE_HEIGHT * TILE_WIDTH] = val;
        }
        tiled_bcast_values = convert_layout<uint16_t>(
            ref_bcast_values_with_tile_padding,
            ref_bcast_shape_with_tile_padding,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);
        num_bcast_tiles = NC;
    } else if (bcast_dim == BcastDim::H) {
        TT_FATAL(W % 32 == 0, "Error");
        ref_bcast_values.resize(NC * W, 0);
        vector<uint32_t> ref_bcast_shape_with_tile_padding = {N, C, TILE_HEIGHT, W};
        vector<uint16_t> ref_bcast_values_with_tile_padding;
        ref_bcast_values_with_tile_padding.resize(NC * TILE_HEIGHT * W, 0);
        for (uint32_t j = 0; j < NC * W; j++) {
            auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
            ref_bcast_values[j] = val;
            ref_bcast_values_with_tile_padding[(j % W) + ((j / W) * TILE_HEIGHT * W)] = val;
        }
        tiled_bcast_values = convert_layout<uint16_t>(
            ref_bcast_values_with_tile_padding,
            ref_bcast_shape_with_tile_padding,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);
        num_bcast_tiles = NC * Wt;
    } else if (bcast_dim == BcastDim::W) {
        ref_bcast_values.resize(NC * H, 0);
        vector<uint32_t> ref_bcast_shape_with_tile_padding = {N, C, H, TILE_WIDTH};
        vector<uint16_t> ref_bcast_values_with_tile_padding;
        ref_bcast_values_with_tile_padding.resize(NC * H * TILE_WIDTH, 0);
        for (uint32_t j = 0; j < NC * H; j++) {
            auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
            ref_bcast_values[j] = val;
            ref_bcast_values_with_tile_padding[j * TILE_WIDTH] = val;
        }
        tiled_bcast_values = convert_layout<uint16_t>(
            ref_bcast_values_with_tile_padding,
            ref_bcast_shape_with_tile_padding,
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES);
        num_bcast_tiles = NC * Ht;
    }

    auto bcast_tiled_u32 = u32_from_u16_vector(tiled_bcast_values);
    auto bcast_vals_nbytes = bcast_tiled_u32.size() * sizeof(bcast_tiled_u32[0]);
    uint32_t src1_page_size = single_tile_bytes;
    if (!multibank) {
        src1_page_size = bcast_vals_nbytes;
    }

    InterleavedBufferConfig src1_config{
        .device = dev, .size = bcast_vals_nbytes, .page_size = src1_page_size, .buffer_type = BufferType::DRAM};

    auto src1_dram_buffer = CreateBuffer(src1_config);
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    detail::WriteToBuffer(src1_dram_buffer, bcast_tiled_u32);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);

    const char* reader_name = get_reader_name(multibank, bcast_dim);
    auto binary_reader_kernel = CreateKernel(
        program,
        reader_name,
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

    uint32_t nc1 = 0;
    SetRuntimeArgs(
        program,
        binary_reader_kernel,
        core,
        {dram_buffer_src0_addr,
         (std::uint32_t)0,
         num_tensor_tiles,
         dram_buffer_src1_addr,
         (std::uint32_t)0,
         num_bcast_tiles,
         NC * Ht * Wt,
         NC,
         Ht,
         Wt,
         nc1});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, (std::uint32_t)0, num_tensor_tiles});

    std::map<std::string, std::string> compute_defines = {
        {"BCAST_DIM", bdim_to_llkdim_define[bcast_dim]},
        {"BCAST_OP", op_id_to_op_define[bcast_op]},
        {"BCAST_LLKOP", op_id_to_llkop_define[bcast_op]}};
    auto eltwise_binary_kernel = CreateKernel(
        program, get_compute_name(bcast_dim), core, ComputeConfig{.compile_args = {}, .defines = compute_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {uint(NC), uint(Ht), uint(Wt)});

    // Execute
    vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 10.0f, 0x1234);
    detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    detail::LaunchProgram(dev, program);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.02f;
        const float atol = 1e-3f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };

    auto u16_src0_vec = u16_from_u32_vector(src0_vec);
    vector<uint16_t> src_linear =
        convert_layout<uint16_t>(u16_src0_vec, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    vector<uint16_t> gold_added = gold_bcast_op(src_linear, shape, ref_bcast_values, bcast_dim, bcast_op);

    vector<uint32_t> shapeR{shape[0], shape[1], shape[2], shape[3]};
    auto gold_4f_u32 = u32_from_u16_vector(
        convert_layout<uint16_t>(gold_added, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, BcastHAdd) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::H, BcastOp::ADD);
}
TEST_F(MeshDeviceSingleCardFixture, BcastHSub) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::H, BcastOp::SUB);
}
TEST_F(MeshDeviceSingleCardFixture, BcastHMul) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::H, BcastOp::MUL);
}

TEST_F(MeshDeviceSingleCardFixture, BcastWAdd) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::W, BcastOp::ADD);
}
TEST_F(MeshDeviceSingleCardFixture, BcastWSub) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::W, BcastOp::SUB);
}
TEST_F(MeshDeviceSingleCardFixture, BcastWMul) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::W, BcastOp::MUL);
}

TEST_F(MeshDeviceSingleCardFixture, BcastHWAdd) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::HW, BcastOp::ADD);
}
TEST_F(MeshDeviceSingleCardFixture, BcastHWSub) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::HW, BcastOp::SUB);
}
TEST_F(MeshDeviceSingleCardFixture, BcastHWMul) {
    run_bcast_test(devices_[0]->get_devices()[0], BcastDim::HW, BcastOp::MUL);
}
