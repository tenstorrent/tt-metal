// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::broadcast {

enum ApiConvention : uint8_t
{
    DEFAULT = 0,
    SHORT_INIT = 1, // call <op>_bcast_<dim>_init_short instead of init_bcast
    SHORT_CALL = 2, // call <op>_tiles_bcast_<dim> instead of <op>_tiles_bcast
    SHORT_BOTH = 3  // both SHORT_INIT and SHORT_CALL
};

enum EltwiseOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2
};

enum BroadcastDim : uint8_t {
    ROW = 0,
    COL = 1,
    SCALAR = 2
};

const map<EltwiseOp, string> eltwise_op_to_type = {
    {EltwiseOp::ADD, "EltwiseBinaryType::ELWADD"},
    {EltwiseOp::SUB, "EltwiseBinaryType::ELWSUB"},
    {EltwiseOp::MUL, "EltwiseBinaryType::ELWMUL"}
};

const map<EltwiseOp, string> eltwise_op_to_api_prefix = {
    {EltwiseOp::ADD, "add"},
    {EltwiseOp::SUB, "sub"},
    {EltwiseOp::MUL, "mul"}
};

const map<BroadcastDim, string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
};

const map<BroadcastDim, string> broadcast_dim_to_api_suffix = {
    {BroadcastDim::ROW, "rows"},
    {BroadcastDim::COL, "cols"},
    {BroadcastDim::SCALAR, "scalar"},
};

struct BroadcastConfig {
    ApiConvention api_convention;
    EltwiseOp eltwise_op;
    BroadcastDim broadcast_dim;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

void mask_src_b_for_broadcast(std::vector<tt::test_utils::df::bfloat16>& tile, const std::vector<uint32_t> &shape, BroadcastDim dim) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (((dim == BroadcastDim::ROW || dim == BroadcastDim::SCALAR) && i != 0) ||
                ((dim == BroadcastDim::ROW || dim == BroadcastDim::SCALAR) && j != 0)) {
                tile[i * num_cols + j] = 0.0f;
            }
        }
    }
}

std::vector<tt::test_utils::df::bfloat16> gold_broadcast(std::vector<tt::test_utils::df::bfloat16>& src_a, std::vector<tt::test_utils::df::bfloat16>& src_b, const std::vector<uint32_t> &shape, EltwiseOp op, BroadcastDim dim, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    uint16_t srca_fid_mask = 0xFFFF;
    uint16_t srcb_fid_mask = 0xFFFF;

    std::vector<tt::test_utils::df::bfloat16> golden(num_cols * num_rows);

    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: { break; }
        case MathFidelity::HiFi2: { srcb_fid_mask = 0xFFFE; break; }
        case MathFidelity::LoFi: { srca_fid_mask = 0xFFF8; srcb_fid_mask = 0xFFFE; break; }
        default: { TT_THROW("Unsupported MathFidelity={}", math_fidelity); break; }
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            tt::test_utils::df::bfloat16 broadcast_value;
            switch (dim)
            {
            case BroadcastDim::ROW: { broadcast_value = src_b[j]; break; }
            case BroadcastDim::COL: { broadcast_value = src_b[i * num_cols]; break; }
            case BroadcastDim::SCALAR: { broadcast_value = src_b[0]; break; }
            default: { TT_THROW("Unsupported BroadcastDim={}", dim); break; }
            }

            switch (op)
            {
            case EltwiseOp::ADD: { golden[i * num_cols + j] = src_a[i * num_cols + j].to_float() + broadcast_value.to_float(); break; }
            case EltwiseOp::SUB: { golden[i * num_cols + j] = src_a[i * num_cols + j].to_float() - broadcast_value.to_float(); break; }
            case EltwiseOp::MUL: {
                golden[i * num_cols + j] =
                    tt::test_utils::df::bfloat16(std::bit_cast<uint32_t>(src_a[i * num_cols + j].to_packed() & srca_fid_mask)).to_float() *
                    tt::test_utils::df::bfloat16(std::bit_cast<uint32_t>(broadcast_value.to_packed() & srcb_fid_mask)).to_float();
                break;
            }
            default: { TT_THROW("Unsupported EltwiseOp={}", op); break; }
            }
        }
    }

    return golden;
}

void run_single_core_broadcast(tt_metal::Device* device, const BroadcastConfig& test_config) {
    if (test_config.eltwise_op == EltwiseOp::SUB && test_config.broadcast_dim == BroadcastDim::ROW && test_config.api_convention != ApiConvention::DEFAULT) {
        GTEST_SKIP(); // FIXME sub_tiles_bcast_rows and sub_bcast_rows_init_short dont exist
    }

    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t single_tile_size = tile_width * tile_height * tt::test_utils::df::bfloat16::SIZEOF;

    tt_metal::InterleavedBufferConfig dram_config{
        .device=device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    auto src_a_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();
    auto src_a_dram_noc_xy = src_a_dram_buffer->noc_coordinates();
    tt_metal::CircularBufferConfig l1_src_a_cb_config = tt_metal::CircularBufferConfig(single_tile_size, {{0, tt::DataFormat::Float16_b}})
        .set_page_size(0, single_tile_size);
    auto l1_src_a_cb = tt_metal::CreateCircularBuffer(program, core, l1_src_a_cb_config);

    auto src_b_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();
    auto src_b_dram_noc_xy = src_b_dram_buffer->noc_coordinates();
    tt_metal::CircularBufferConfig l1_src_b_cb_config = tt_metal::CircularBufferConfig(single_tile_size, {{1, tt::DataFormat::Float16_b}})
        .set_page_size(1, single_tile_size);
    auto l1_src_b_cb = tt_metal::CreateCircularBuffer(program, core, l1_src_b_cb_config);

    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
    auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();
    tt_metal::CircularBufferConfig l1_dst_cb_config = tt_metal::CircularBufferConfig(single_tile_size, {{16, tt::DataFormat::Float16_b}})
        .set_page_size(16, single_tile_size);
    auto l1_dst_cb = tt_metal::CreateCircularBuffer(program, core, l1_dst_cb_config);

    std::map<string, string> defines = {
        {"BCAST_LLKOP", eltwise_op_to_type.at(test_config.eltwise_op)},
        {"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)},
        {"BCAST_OP", eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast"}
    };

    log_info("Testing BCAST_LLKOP={} BCAST_DIM={}", defines["BCAST_LLKOP"], defines["BCAST_DIM"]);

    if (test_config.api_convention == ApiConvention::SHORT_INIT || test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_bcast_" + broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";

        if ((test_config.eltwise_op == EltwiseOp::SUB || test_config.eltwise_op == EltwiseOp::MUL) && test_config.broadcast_dim == BroadcastDim::SCALAR) {
            // FIXME sub_bcast_scalar_init_short and mul_bcast_scalar_init_short are instead called sub_tiles_bcast_scalar_init_short and mul_tiles_bcast_scalar_init_short
            defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast_" + broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";
        }

        log_info("Init function is {}", defines["BCAST_OP_INIT"]);
    }
    else {
        log_info("Init function is init_bcast");
    }

    if (test_config.api_convention == ApiConvention::SHORT_CALL || test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_SPECIFIC"] = "1";
        defines["BCAST_OP"] = defines["BCAST_OP"] + "_" + broadcast_dim_to_api_suffix.at(test_config.broadcast_dim);
    }

    log_info("Compute function is {}", defines["BCAST_OP"]);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/broadcast.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = test_config.math_fidelity, .compile_args = {}, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)dram_buffer_src_a_addr,
            (uint32_t)src_a_dram_noc_xy.x,
            (uint32_t)src_a_dram_noc_xy.y,
            (uint32_t)dram_buffer_src_b_addr,
            (uint32_t)src_b_dram_noc_xy.x,
            (uint32_t)src_b_dram_noc_xy.y,
            (uint32_t)1,
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dram_buffer_dst_addr,
            (uint32_t)dst_dram_noc_xy.x,
            (uint32_t)dst_dram_noc_xy.y,
            (uint32_t)1,
        });

    std::vector<tt::test_utils::df::bfloat16> input0 = generate_uniform_random_vector<tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        single_tile_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<tt::test_utils::df::bfloat16> input1 = generate_uniform_random_vector<tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        single_tile_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    mask_src_b_for_broadcast(input1, {tile_width, tile_height}, test_config.broadcast_dim);

    std::vector<tt::test_utils::df::bfloat16> golden = gold_broadcast(input0, input1, {tile_width, tile_height}, test_config.eltwise_op, test_config.broadcast_dim, test_config.math_fidelity);

    auto packed_input0 = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(input0);
    auto packed_input1 = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(input1);
    auto packed_golden = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(golden);
    unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = tile_width/32,
        .num_tiles_c_dim = tile_height/32
    };
    auto tilized_input0 = unit_tests::compute::gold_standard_tilize(packed_input0, config);
    auto tilized_input1 = unit_tests::compute::gold_standard_tilize(packed_input1, config);

    tt_metal::detail::WriteToBuffer(src_a_dram_buffer, tilized_input0);
    tt_metal::detail::WriteToBuffer(src_b_dram_buffer, tilized_input1);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, dest_buffer_data);
    auto dest_buffer_data_untilized = unit_tests::compute::gold_standard_untilize(dest_buffer_data, config);

    bool result = is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        dest_buffer_data_untilized,
        packed_golden,
        [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) {
            return is_close(a, b, 0.0155);
        });
    ASSERT_TRUE(result);
}
}

class BroadcastParametrizedDeviceFixture : public DeviceFixture,
                                           public testing::WithParamInterface<unit_tests::compute::broadcast::BroadcastConfig> {
};

TEST_P(BroadcastParametrizedDeviceFixture, ComputeSingleTileBroadcast) {
    unit_tests::compute::broadcast::BroadcastConfig test_config = GetParam();
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) continue;
        log_info("Math Fidelity = {}", i);
        test_config.math_fidelity = MathFidelity(i);
        unit_tests::compute::broadcast::run_single_core_broadcast(this->devices_.at(0), test_config);
    }
}

using namespace unit_tests::compute::broadcast;

INSTANTIATE_TEST_SUITE_P(
    ComputeSingleTileBroadcast,
    BroadcastParametrizedDeviceFixture,
    ::testing::Values(
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT,    EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::SCALAR}
        ));
