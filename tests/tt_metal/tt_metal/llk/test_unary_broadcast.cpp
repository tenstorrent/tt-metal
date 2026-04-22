// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/bfloat8.hpp>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include <string_view>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
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
#include <umd/device/types/arch.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::unary_broadcast {

enum BroadcastDim : uint8_t { ROW, COL, SCALAR, NONE, NUM_DIMS };

const map<BroadcastDim, std::string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
    {BroadcastDim::NONE, "BroadcastType::NONE"}};

struct UnaryBroadcastConfig {
    BroadcastDim broadcast_dim;
    tt::DataFormat in_t;
    tt::DataFormat out_t;
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
                    T broadcast_value{};
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
        std::is_same_v<bfloat16, T_in> || std::is_same_v<float, T_in>, "Only float & Float_16b type as input allowed");
    std::vector<uint32_t> tilized_packed_res;
    ::unit_tests::compute::GoldenConfig config = {.num_tiles_r_dim = shape.at(0), .num_tiles_c_dim = 1};
    std::vector<T_in> vBroadcast = get_broadcasted_vec(src, shape, dim);
    if constexpr (std::is_same_v<bfloat16, T_in>) {
        if (T_out == tt::DataFormat::Float16_b) {
            auto packed_vec = pack_vector<uint32_t, bfloat16>(vBroadcast);
            tilized_packed_res = ::unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            std::vector<float> tempfp32v;
            tempfp32v.resize(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp32v[i] = static_cast<float>(vBroadcast[i]);
            }
            tilized_packed_res = pack_as_bfp8_tiles(tt::stl::make_const_span(tempfp32v), true, false);
        } else {
            TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
        }
    } else if constexpr (std::is_same_v<float, T_in>) {
        if (T_out == tt::DataFormat::Float16_b) {
            std::vector<bfloat16> tempfp16bv;
            tempfp16bv.resize(vBroadcast.size());
            for (int i = 0; i < vBroadcast.size(); i++) {
                tempfp16bv[i] = vBroadcast[i];
            }
            auto packed_vec = pack_vector<uint32_t, bfloat16>(tempfp16bv);
            tilized_packed_res = ::unit_tests::compute::gold_standard_tilize(packed_vec, config);
        } else if (T_out == tt::DataFormat::Bfp8_b) {
            tilized_packed_res = pack_as_bfp8_tiles(tt::stl::make_const_span(vBroadcast), true, false);
        } else {
            TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
        }
    }
    return tilized_packed_res;
}

bool check_is_close(
    std::vector<uint32_t>& packed_golden,
    std::vector<uint32_t>& device_res,
    tt::DataFormat T_out,
    std::string_view result_label) {
    if (T_out == tt::DataFormat::Float16_b) {
        if (packed_golden.size() != device_res.size()) {
            TT_THROW("{} mismatch: size golden={} device={}", result_label, packed_golden.size(), device_res.size());
        }
        auto gold_bf16 = unpack_vector<bfloat16, uint32_t>(packed_golden);
        auto res_bf16 = unpack_vector<bfloat16, uint32_t>(device_res);
        for (size_t i = 0; i < gold_bf16.size(); i++) {
            if (!is_close(gold_bf16[i], res_bf16[i], 0.0)) {
                TT_THROW(
                    "{} mismatch at index {} golden={} device={}",
                    result_label,
                    i,
                    static_cast<float>(gold_bf16[i]),
                    static_cast<float>(res_bf16[i]));
            }
        }
        return true;
    }
    if (T_out == tt::DataFormat::Bfp8_b) {
        constexpr float atol = 0.03125f;
        auto gold_refloat = unpack_bfp8_tiles_into_float_vec(packed_golden, true, false);
        auto res_refloat = unpack_bfp8_tiles_into_float_vec(device_res, true, false);
        if (gold_refloat.size() != res_refloat.size()) {
            TT_THROW("{} mismatch: size golden={} device={}", result_label, gold_refloat.size(), res_refloat.size());
        }
        for (size_t i = 0; i < gold_refloat.size(); i++) {
            if (std::fabs(gold_refloat[i] - res_refloat[i]) > atol) {
                TT_THROW(
                    "{} mismatch at index {} A={} B={} atol={}",
                    result_label,
                    i,
                    gold_refloat[i],
                    res_refloat[i],
                    atol);
            }
        }
        return true;
    }
    TT_THROW("Testing infrastructure not setup for output data type {}", T_out);
}

auto CreateDramBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, tt::DataFormat dformat, uint32_t num_tiles) {
    uint32_t single_tile_size = tile_size(dformat);
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

    return distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
}

CBHandle CreateCircularBufferHelper(
    distributed::MeshWorkload& workload, CoreCoord& core, uint32_t num_pages, tt::DataFormat dformat, uint32_t id) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    uint32_t page_size = tile_size(dformat);
    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(num_pages * page_size, {{id, dformat}}).set_page_size(id, page_size);
    return tt_metal::CreateCircularBuffer(workload.get_programs().at(device_range), core, l1_cb_config);
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

void run_single_core_unary_broadcast(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const UnaryBroadcastConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    CoreCoord core = {0, 0};

    constexpr uint32_t num_tiles = 32;
    constexpr uint32_t num_blocks = 4;
    constexpr uint32_t block_size = num_tiles / num_blocks;
    const tt::DataFormat in_t = test_config.in_t;
    const tt::DataFormat out_t = test_config.out_t;

    auto src_dram_buffer = CreateDramBuffer(mesh_device, in_t, num_tiles);
    auto dst_dram_buffer = CreateDramBuffer(mesh_device, out_t, num_tiles);

    const tt_metal::Tile tile_shape = tt_metal::Tile({constants::TILE_HEIGHT, constants::TILE_WIDTH});
    const uint32_t dfb_num_entries = block_size * 2;

    KernelHandle reader_kernel;
    KernelHandle writer_kernel;
    KernelHandle compute_kernel;

    uint32_t src_dfb = 0;
    uint32_t dst_dfb = 0;

    // Mesh DRAM: TensorAccessorArgs + reader_unary_8bank / writer_unary_8bank.
    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_args);

    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        tt_metal::experimental::dfb::DataflowBufferConfig dfb_src_config = {
            .entry_size = tile_size(in_t),
            .num_entries = dfb_num_entries,
            .num_producers = 1,
            .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .num_consumers = 1,
            .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = false,
            .data_format = in_t,
            .tile = tile_shape,
        };
        tt_metal::experimental::dfb::DataflowBufferConfig dfb_dst_config = {
            .entry_size = tile_size(out_t),
            .num_entries = dfb_num_entries,
            .num_producers = 1,
            .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .num_consumers = 1,
            .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
            .enable_implicit_sync = false,
            .data_format = out_t,
            .tile = tile_shape,
        };

        src_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, dfb_src_config);
        dst_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, dfb_dst_config);
        TT_FATAL(
            src_dfb == 0 && dst_dfb == 1,
            "Unary broadcast test expects src DFB id 0 and dst DFB id 1 (got src={} dst={})",
            src_dfb,
            dst_dfb);

        std::vector<uint32_t> writer_compile_args = {dst_dfb};
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_args);

        reader_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = reader_compile_args});

        writer_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = writer_compile_args});

    } else {
        CreateCircularBufferHelper(workload, core, dfb_num_entries, in_t, 0);
        CreateCircularBufferHelper(workload, core, dfb_num_entries, out_t, 16);

        std::vector<uint32_t> writer_compile_args = {static_cast<uint32_t>(tt::CBIndex::c_16)};
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_args);

        reader_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_args});

        writer_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_compile_args});
    }

    std::map<std::string, std::string> defines = {{"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)}};

    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        compute_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unary_bcast.cpp",
            core,
            tt_metal::experimental::quasar::QuasarComputeConfig{
                .num_threads_per_cluster = 1, .compile_args = {num_blocks, block_size}, .defines = defines});

        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, src_dfb, reader_kernel, compute_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, dst_dfb, compute_kernel, writer_kernel);
    } else {
        compute_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unary_bcast.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = {num_blocks, block_size}, .defines = defines});
    }

    // reader_unary_8bank: arg 3 is num_tiles.
    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)(src_dram_buffer->address()),
            (uint32_t)0,  // dram bank id
            (uint32_t)0,  // unused; keeps num_tiles at index 3
            (uint32_t)num_tiles,
        });

    // writer_unary_8bank: arg 0 = base addr, arg 2 = num_tiles
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)(dst_dram_buffer->address()),
            (uint32_t)0,  // unused
            (uint32_t)num_tiles,
        });

    std::vector<uint32_t> packed_tilized_input;
    std::vector<uint32_t> golden_packed_tilized_output;
    get_packed_tilized_input_output_pair(
        in_t, out_t, num_tiles, test_config.broadcast_dim, packed_tilized_input, golden_packed_tilized_output);
    distributed::WriteShard(cq, src_dram_buffer, packed_tilized_input, zero_coord);

    auto* device = mesh_device->get_devices()[0];
    const bool blocking = device->arch() == ARCH::QUASAR;
    distributed::EnqueueMeshWorkload(cq, workload, blocking);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);

    ASSERT_TRUE(check_is_close(golden_packed_tilized_output, dest_buffer_data, out_t, "unary_broadcast_dram_out"));
}
}  // namespace unit_tests::compute::unary_broadcast

using namespace unit_tests::compute::unary_broadcast;

// 32 tiles in 4 blocks of 8; single src→dst DFB path (Quasar). ROW/COL/SCALAR in loop; TODO #38092 runs
// SCALAR+Float16_b only.
TEST_F(MeshDeviceFixture, TensixComputeUnaryBroadcastQuasarDfb) {
    if (this->arch_ != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Unary broadcast DFB test requires Quasar";
    }
    constexpr BroadcastDim k_quasar_dims[] = {BroadcastDim::ROW, BroadcastDim::COL, BroadcastDim::SCALAR};
    constexpr struct {
        tt::DataFormat in_t;
        tt::DataFormat out_t;
    } k_formats[] = {
        {tt::DataFormat::Float16_b, tt::DataFormat::Float16_b},
        {tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8_b},
    };
    for (BroadcastDim bcast_dim : k_quasar_dims) {
        for (const auto& fmt : k_formats) {
            // TODO (#38092): Remove when we can run back to back tests on Quasar
            if (bcast_dim != BroadcastDim::SCALAR || fmt.in_t != tt::DataFormat::Float16_b) {
                continue;
            }
            UnaryBroadcastConfig test_config = {
                .broadcast_dim = bcast_dim,
                .in_t = fmt.in_t,
                .out_t = fmt.out_t,
            };

            log_info(
                tt::LogTest,
                "Testing UNARY BROADCAST bcast={} in_t={} out_t={}",
                broadcast_dim_to_type.at(test_config.broadcast_dim),
                test_config.in_t,
                test_config.out_t);
            run_single_core_unary_broadcast(this->devices_.at(0), test_config);
        }
    }
}

}  // namespace tt::tt_metal
