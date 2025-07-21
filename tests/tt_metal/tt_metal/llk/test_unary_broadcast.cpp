// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat8.hpp>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::broadcast {

enum BroadcastDim : uint8_t { ROW, COL, SCALAR, NONE, NUM_DIMS };

const map<BroadcastDim, std::string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
    {BroadcastDim::NONE, "BroadcastType::NONE"}};

struct UnaryBroadcastConfig {
    BroadcastDim broadcast_dim_0;
    BroadcastDim broadcast_dim_1;
    tt::DataFormat in0_t;
    tt::DataFormat in1_t;
    tt::DataFormat out0_t;
    tt::DataFormat out1_t;
};

// Assume 1Xn tiles.
template <class T>
std::vector<T> get_broadcasted_vec(std::vector<T>& src, const std::vector<uint32_t>& shape, BroadcastDim dim) {
    int num_tiles = shape.at(0);
    int num_rows = shape.at(1);
    int num_cols = shape.at(2);
    int tile_elem_count = num_rows * num_cols;

    std::vector<T> vBroadcast(num_tiles * num_cols * num_rows);

    if (dim == BroadcastDim::NONE) {
        vBroadcast = src;
    } else {
        for (int t = 0; t < num_tiles; t++) {
            int tile_offset = tile_elem_count * t;
            for (int i = 0; i < num_rows; i++) {
                for (int j = 0; j < num_cols; j++) {
                    T broadcast_value;
                    switch (dim) {
                        case BroadcastDim::ROW: {
                            broadcast_value = src[tile_offset + j];
                            break;
                        }
                        case BroadcastDim::COL: {
                            broadcast_value = src[tile_offset + (i * num_cols)];
                            break;
                        }
                        case BroadcastDim::SCALAR: {
                            broadcast_value = src[tile_offset];
                            break;
                        }
                        default: {
                            TT_THROW("Unsupported BroadcastDim={}", dim);
                            break;
                        }
                    }

                    vBroadcast[tile_offset + (i * num_cols + j)] = broadcast_value;
                }
            }
        }
    }

    return vBroadcast;
}

// T_in : type of src vector
// T_out : type of data the packer will pack out
// Assume nx1 tiles, row major data layout.
template <class T_in>
std::vector<uint32_t> get_tilized_packed_golden_broadcast(
    std::vector<T_in>& src, const std::vector<uint32_t>& shape, BroadcastDim dim, tt::DataFormat T_out) {
    static_assert(
        std::is_same<bfloat16, T_in>::value || std::is_same<float, T_in>::value,
        "Only float & Float_16b type as input allowed");
    std::vector<uint32_t> tilized_packed_res;
    ::unit_tests::compute::GoldenConfig config = {.num_tiles_r_dim = shape.at(0), .num_tiles_c_dim = 1};
    std::vector<T_in> vBroadcast = get_broadcasted_vec(src, shape, dim);
    if constexpr (std::is_same<bfloat16, T_in>::value) {
        if (T_out == tt::DataFormat::Float16_b) {
            auto packed_vec = pack_vector<uint32_t, bfloat16>(vBroadcast);
            tilized_packed_res = ::unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            std::vector<float> tempfp32v;
            tempfp32v.resize(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp32v[i] = vBroadcast[i].to_float();
            }
            tilized_packed_res = pack_fp32_vec_as_bfp8_tiles(tempfp32v, true, false);
        } else {
            TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
        }
    } else if constexpr (std::is_same<float, T_in>::value) {
        if (T_out == tt::DataFormat::Float16_b) {
            std::vector<bfloat16> tempfp16bv;
            tempfp16bv.resize(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp16bv[i] = vBroadcast[i];
            }
            auto packed_vec = pack_vector<uint32_t, bfloat16>(tempfp16bv);
            tilized_packed_res = ::unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            tilized_packed_res = pack_fp32_vec_as_bfp8_tiles(vBroadcast, true, false);
        } else {
            TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
        }
    }
    return tilized_packed_res;
}

bool check_is_close(std::vector<uint32_t>& packed_golden, std::vector<uint32_t>& device_res, tt::DataFormat T_out) {
    bool result = true;
    if (T_out == tt::DataFormat::Float16_b) {
        result = is_close_packed_vectors<bfloat16, uint32_t>(
            packed_golden, device_res, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.0); });
    } else if (T_out == tt::DataFormat::Bfp8_b) {
        // Host side may do nearest to even but device side may do nearest rounding, with rounding up
        // in case of tie. Also need to note packer source format, which may lead to additional rounding.
        float atol = 0.03125f;
        auto gold_refloat = unpack_bfp8_tiles_into_float_vec(packed_golden, true, false);
        auto res_refloat = unpack_bfp8_tiles_into_float_vec(device_res, true, false);
        if (gold_refloat.size() != res_refloat.size()) {
            TT_THROW(
                "Mismatch in size of vectors for comparison A.size={} B.size={}",
                gold_refloat.size(),
                res_refloat.size());
        }
        for (int i = 0; i < gold_refloat.size(); i++) {
            if (std::fabs(gold_refloat[i] - res_refloat[i]) > atol) {
                TT_THROW("Mismatch  A={} B={} atol={}", gold_refloat[i], res_refloat[i], atol);
                result = false;
                break;
            }
        }
    } else {
        TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
    }

    return result;
}

auto CreateDramBuffer(tt_metal::IDevice* device, tt::DataFormat dformat, uint32_t num_tiles) {
    uint32_t single_tile_size = tile_size(dformat);
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    return CreateBuffer(dram_config);
}

CBHandle CreateCircularBufferHelper(
    Program& program, CoreCoord& core, uint32_t num_pages, tt::DataFormat dformat, uint32_t id) {
    uint32_t page_size = tile_size(dformat);
    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(num_pages * page_size, {{id, dformat}}).set_page_size(id, page_size);
    return tt_metal::CreateCircularBuffer(program, core, l1_cb_config);
}

void get_packed_tilized_input_output_pair(
    tt::DataFormat in_t,
    tt::DataFormat out_t,
    uint32_t num_tiles,
    BroadcastDim bcast_dim,
    std::vector<uint32_t>& packed_tilized_input,
    std::vector<uint32_t>& packed_tilized_output) {
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t num_single_tile_elem = tile_width * tile_height;
    if (in_t == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> input = generate_uniform_random_vector<bfloat16>(
            1.0f, 2.0f, num_tiles * num_single_tile_elem, std::chrono::system_clock::now().time_since_epoch().count());

        ::unit_tests::compute::GoldenConfig config = {.num_tiles_r_dim = num_tiles, .num_tiles_c_dim = 1};
        auto packed_input = pack_vector<uint32_t, bfloat16>(input);
        packed_tilized_input = ::unit_tests::compute::gold_standard_tilize(packed_input, config);
        packed_tilized_output =
            get_tilized_packed_golden_broadcast(input, {num_tiles, tile_width, tile_height}, bcast_dim, out_t);
    } else if (in_t == tt::DataFormat::Bfp8_b) {
        packed_tilized_input = create_random_vector_of_bfp8(num_tiles * tile_size(in_t), false, 1, 1.0);
        std::vector<float> input = unpack_bfp8_tiles_into_float_vec(packed_tilized_input, true, false);
        packed_tilized_output =
            get_tilized_packed_golden_broadcast(input, {num_tiles, tile_width, tile_height}, bcast_dim, out_t);
    }
}

void run_single_core_unary_broadcast(tt_metal::IDevice* device, const UnaryBroadcastConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t num_tiles = 32;
    constexpr uint32_t num_blocks = 4;
    constexpr uint32_t block_size = num_tiles / num_blocks;
    tt::DataFormat in0_t = test_config.in0_t;
    tt::DataFormat out0_t = test_config.out0_t;
    tt::DataFormat in1_t = test_config.in1_t;
    tt::DataFormat out1_t = test_config.out1_t;

    auto src_dram_buffer_0 = CreateDramBuffer(device, in0_t, num_tiles);
    auto dst_dram_buffer_0 = CreateDramBuffer(device, out0_t, num_tiles);
    auto src_dram_buffer_1 = CreateDramBuffer(device, in1_t, num_tiles);
    auto dst_dram_buffer_1 = CreateDramBuffer(device, out1_t, num_tiles);
    auto l1_src_cb_0 = CreateCircularBufferHelper(program, core, block_size * 2, in0_t, 0);
    auto l1_dst_cb_0 = CreateCircularBufferHelper(program, core, block_size * 2, out0_t, 16);
    auto l1_src_cb_1 = CreateCircularBufferHelper(program, core, block_size * 2, in1_t, 1);
    auto l1_dst_cb_1 = CreateCircularBufferHelper(program, core, block_size * 2, out1_t, 17);

    std::map<std::string, std::string> defines = {
        {"BCAST_DIM_0", broadcast_dim_to_type.at(test_config.broadcast_dim_0)},
        {"BCAST_DIM_1", broadcast_dim_to_type.at(test_config.broadcast_dim_1)}};

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_dual_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {num_blocks, block_size}, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)(src_dram_buffer_0->address()),
            (uint32_t)0,  // dram bank id
            (uint32_t)(src_dram_buffer_1->address()),
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)(dst_dram_buffer_0->address()),
            (uint32_t)0,  // dram bank id
            (uint32_t)(dst_dram_buffer_1->address()),
            (uint32_t)0,          // dram bank id
            (uint32_t)num_tiles,  // num tiles
        });

    std::vector<uint32_t> packed_tilized_input_0, golden_packed_tilized_output_0;
    get_packed_tilized_input_output_pair(
        in0_t, out0_t, num_tiles, test_config.broadcast_dim_0, packed_tilized_input_0, golden_packed_tilized_output_0);
    tt_metal::detail::WriteToBuffer(src_dram_buffer_0, packed_tilized_input_0);

    std::vector<uint32_t> packed_tilized_input_1, golden_packed_tilized_output_1;
    get_packed_tilized_input_output_pair(
        in1_t, out1_t, num_tiles, test_config.broadcast_dim_1, packed_tilized_input_1, golden_packed_tilized_output_1);
    tt_metal::detail::WriteToBuffer(src_dram_buffer_1, packed_tilized_input_1);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data_0;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_0, dest_buffer_data_0);
    std::vector<uint32_t> dest_buffer_data_1;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_1, dest_buffer_data_1);

    bool result = check_is_close(golden_packed_tilized_output_0, dest_buffer_data_0, out0_t);
    result &= check_is_close(golden_packed_tilized_output_1, dest_buffer_data_1, out1_t);

    ASSERT_TRUE(result);
}
}  // namespace unit_tests::compute::broadcast

using namespace unit_tests::compute::broadcast;

TEST_F(DeviceFixture, TensixComputeSingleTileUnaryBroadcast) {
    if (this->arch_ == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    for (BroadcastDim bcast_dim : {BroadcastDim::NONE, BroadcastDim::ROW, BroadcastDim::COL, BroadcastDim::SCALAR}) {
        for (tt::DataFormat in0_t_ : {tt::DataFormat::Bfp8_b, tt::DataFormat::Float16_b}) {
            for (tt::DataFormat out0_t_ : {tt::DataFormat::Bfp8_b, tt::DataFormat::Float16_b}) {
                UnaryBroadcastConfig test_config = {
                    .broadcast_dim_0 = bcast_dim,
                    .broadcast_dim_1 = (BroadcastDim)((bcast_dim + 1) % BroadcastDim::NUM_DIMS),
                    .in0_t = in0_t_,
                    .in1_t = (in0_t_ == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Float16_b : tt::DataFormat::Bfp8_b,
                    .out0_t = out0_t_,
                    .out1_t = (out0_t_ == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Float16_b : tt::DataFormat::Bfp8_b};

                log_info(
                    tt::LogTest,
                    "Testing UNARY BROADCAST BCAST_DIM_0={} in0_t={} out0_t={} | BCAST_DIM_1={} in1_t={} out1_t={}",
                    broadcast_dim_to_type.at(test_config.broadcast_dim_0),
                    test_config.in0_t,
                    test_config.out0_t,
                    broadcast_dim_to_type.at(test_config.broadcast_dim_1),
                    test_config.in1_t,
                    test_config.out1_t);
                unit_tests::compute::broadcast::run_single_core_unary_broadcast(this->devices_.at(0), test_config);
            }
        }
    }
}

}  // namespace tt::tt_metal
