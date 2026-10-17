// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <array>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include "impl/program/program_impl.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "single_core_compute_runners.hpp"

namespace tt::tt_metal {

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::broadcast {

enum ApiConvention : std::uint8_t {
    DEFAULT = 0,
    SHORT_INIT = 1,  // call <op>_bcast_<dim>_init instead of bcast_init<OP, DIM>
    SHORT_CALL = 2,  // call <op>_tiles_bcast_<dim> instead of <op>_tiles_bcast
    SHORT_BOTH = 3   // both SHORT_INIT and SHORT_CALL
};

enum EltwiseOp : std::uint8_t { ADD = 0, SUB = 1, MUL = 2 };

enum BroadcastDim : std::uint8_t { ROW = 0, COL = 1, SCALAR = 2 };

enum TileShape : std::uint8_t { FULL_TILE = 0, TINY_TILE_16x32 = 1 };

const map<EltwiseOp, std::string> eltwise_op_to_type = {
    {EltwiseOp::ADD, "EltwiseBinaryType::ELWADD"},
    {EltwiseOp::SUB, "EltwiseBinaryType::ELWSUB"},
    {EltwiseOp::MUL, "EltwiseBinaryType::ELWMUL"}};

const map<EltwiseOp, std::string> eltwise_op_to_api_prefix = {
    {EltwiseOp::ADD, "add"}, {EltwiseOp::SUB, "sub"}, {EltwiseOp::MUL, "mul"}};

const map<BroadcastDim, std::string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
};

const map<BroadcastDim, std::string> broadcast_dim_to_api_suffix = {
    {BroadcastDim::ROW, "rows"},
    {BroadcastDim::COL, "cols"},
    {BroadcastDim::SCALAR, "scalar"},
};

const map<TileShape, tt_metal::Tile> tile_shape_to_tile = {
    {TileShape::FULL_TILE, tt_metal::Tile({constants::TILE_HEIGHT, constants::TILE_WIDTH})},
    {TileShape::TINY_TILE_16x32, tt_metal::Tile({constants::TILE_HEIGHT / 2, constants::TILE_WIDTH})},
};

// Per-element bfloat16 tolerance shared by both runners below.
constexpr float k_broadcast_rtol = 0.0155;

// Quasar drives dataflow through the Gen2 config and takes its DFB sync explicitly; Gen1 arches pin
// the processor/NOC pair per direction. Identical for every runner in this file, hence the helpers.
experimental::DataMovementHardwareConfig make_reader_hw_config(const distributed::MeshDevice& mesh_device) {
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        return experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    }
    return experimental::DataMovementHardwareConfig{
        .config_1xx =
            experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
            },
    };
}

experimental::DataMovementHardwareConfig make_writer_hw_config(const distributed::MeshDevice& mesh_device) {
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        return experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    }
    return experimental::DataMovementHardwareConfig{
        .config_1xx =
            experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
            },
    };
}

experimental::ComputeHardwareConfig make_compute_hw_config(
    const distributed::MeshDevice& mesh_device, MathFidelity math_fidelity) {
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        return experimental::ComputeHardwareConfig{.fpu_math_fidelity = math_fidelity};
    }
    return experimental::ComputeHardwareConfig{.fpu_math_fidelity = math_fidelity};
}

struct BroadcastConfig {
    ApiConvention api_convention;
    EltwiseOp eltwise_op;
    BroadcastDim broadcast_dim;
    TileShape tile_shape = TileShape::FULL_TILE;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    std::uint32_t bcast_row_idx = 0;
};

void mask_src_b_for_broadcast(
    std::vector<bfloat16>& tile, const std::vector<std::uint32_t>& shape, BroadcastDim dim, std::uint32_t row_idx = 0) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if ((dim == BroadcastDim::ROW && i != row_idx) ||
                (dim == BroadcastDim::SCALAR && (i != row_idx || j != 0))) {
                tile[(i * num_cols) + j] = 0.0f;
            }
        }
    }
}

std::vector<bfloat16> gold_broadcast(
    std::vector<bfloat16>& src_a,
    std::vector<bfloat16>& src_b,
    const std::vector<std::uint32_t>& shape,
    EltwiseOp op,
    BroadcastDim dim,
    std::uint32_t row_idx = 0,
    MathFidelity math_fidelity = MathFidelity::HiFi4) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    std::uint16_t srca_fid_mask = 0xFFFF;
    std::uint16_t srcb_fid_mask = 0xFFFF;

    std::vector<bfloat16> golden(num_cols * num_rows);

    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = 0xFFFE;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            bfloat16 broadcast_value{};
            switch (dim) {
                case BroadcastDim::ROW: {
                    broadcast_value = src_b[(row_idx * num_cols) + j];
                    break;
                }
                case BroadcastDim::COL: {
                    broadcast_value = src_b[i * num_cols];
                    break;
                }
                case BroadcastDim::SCALAR: {
                    broadcast_value = src_b[0];
                    break;
                }
                default: {
                    TT_THROW("Unsupported BroadcastDim={}", dim);
                    break;
                }
            }

            switch (op) {
                case EltwiseOp::ADD: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(src_a[(i * num_cols) + j]) + static_cast<float>(broadcast_value);
                    break;
                }
                case EltwiseOp::SUB: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(src_a[(i * num_cols) + j]) - static_cast<float>(broadcast_value);
                    break;
                }
                case EltwiseOp::MUL: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(
                            std::bit_cast<std::uint16_t>(src_a[(i * num_cols) + j]) & srca_fid_mask))) *
                        static_cast<float>(std::bit_cast<bfloat16>(
                            static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(broadcast_value) & srcb_fid_mask)));
                    break;
                }
                default: {
                    TT_THROW("Unsupported EltwiseOp={}", op);
                    break;
                }
            }
        }
    }

    return golden;
}

constexpr std::uint32_t k_num_tiles_broadcast_test = 1;

auto CreateDramBufferForPageSize(
    distributed::MeshDevice& mesh_device, std::uint32_t page_size_bytes, std::uint32_t num_pages) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = page_size_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = page_size_bytes * num_pages};
    return distributed::MeshBuffer::create(buffer_config, dram_config, &mesh_device);
}

void run_single_core_broadcast(distributed::MeshDevice& mesh_device, const BroadcastConfig& test_config) {
    auto& cq = mesh_device.mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};

    tt_metal::Tile tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    std::uint32_t tile_width = tile_dims.get_tile_shape()[1];
    std::uint32_t tile_height = tile_dims.get_tile_shape()[0];
    if (test_config.tile_shape != TileShape::FULL_TILE) {
        log_info(tt::LogTest, "Tile shape is {{{}, {}}}", tile_height, tile_width);
    }

    std::uint32_t single_tile_size = tile_width * tile_height * sizeof(bfloat16);

    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();

    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();

    auto dst_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    const bool is_quasar = mesh_device.arch() == ARCH::QUASAR;

    std::map<std::string, std::string> defines = {
        {"BCAST_LLKOP", eltwise_op_to_type.at(test_config.eltwise_op)},
        {"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)},
        {"BCAST_OP", eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast"},
        {"BCAST_ROW_IDX", std::to_string(test_config.bcast_row_idx)}};

    // Add a helper define to indicate if this is a row broadcast
    if (test_config.broadcast_dim == BroadcastDim::ROW) {
        defines["BCAST_IS_ROW"] = "1";
    }

    log_info(
        tt::LogTest,
        "Testing BCAST_LLKOP={} BCAST_DIM={} ROW_IDX={}",
        defines["BCAST_LLKOP"],
        defines["BCAST_DIM"],
        test_config.bcast_row_idx);

    if (test_config.api_convention == ApiConvention::SHORT_INIT ||
        test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_bcast_" +
                                   broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init";

        log_info(tt::LogTest, "Init function is {}", defines["BCAST_OP_INIT"]);
    } else {
        log_info(tt::LogTest, "Init function is bcast_init");
    }

    if (test_config.api_convention == ApiConvention::SHORT_CALL ||
        test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_SPECIFIC"] = "1";
        defines["BCAST_OP"] = defines["BCAST_OP"] + "_" + broadcast_dim_to_api_suffix.at(test_config.broadcast_dim);
    }

    log_info(tt::LogTest, "Compute function is {}", defines["BCAST_OP"]);

    experimental::KernelSpec::CompilerOptions::Defines defines_vec;
    for (auto& kv : defines) {
        defines_vec.emplace(kv.first, kv.second);
    }

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = k_num_tiles_broadcast_test,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile_dims,
        };
    };

    experimental::DataflowBufferSpec inp0_dfb_spec = make_dfb(INP0_DFB);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_dfb(INP1_DFB);
    experimental::DataflowBufferSpec out_dfb_spec = make_dfb(OUT_DFB);

    experimental::DataMovementHardwareConfig reader_hw_config = make_reader_hw_config(mesh_device);
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config = make_writer_hw_config(mesh_device);
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::ComputeHardwareConfig compute_hw_config =
        make_compute_hw_config(mesh_device, test_config.math_fidelity);

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/broadcast_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines_vec},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_broadcast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", static_cast<std::uint32_t>(dram_buffer_src_a_addr)},
                 {"src0_bank_id", 0u},
                 {"src1_addr", static_cast<std::uint32_t>(dram_buffer_src_b_addr)},
                 {"src1_bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(k_num_tiles_broadcast_test)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dram_buffer_dst_addr)},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(k_num_tiles_broadcast_test)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    std::vector<bfloat16> input0 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, single_tile_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> input1 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, single_tile_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());

    mask_src_b_for_broadcast(input1, {tile_height, tile_width}, test_config.broadcast_dim, test_config.bcast_row_idx);

    std::vector<bfloat16> golden = gold_broadcast(
        input0,
        input1,
        {tile_height, tile_width},
        test_config.eltwise_op,
        test_config.broadcast_dim,
        test_config.bcast_row_idx,
        test_config.math_fidelity);

    auto packed_input0 = pack_vector<std::uint32_t, bfloat16>(input0);
    auto packed_input1 = pack_vector<std::uint32_t, bfloat16>(input1);
    auto packed_golden = pack_vector<std::uint32_t, bfloat16>(golden);
    ::unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = 1,
        .num_tiles_c_dim = 1,
        .num_faces = tile_width / 16 * tile_height / 16,
        .tiny_tile = test_config.tile_shape != TileShape::FULL_TILE};
    auto tilized_input0 = ::unit_tests::compute::gold_standard_tilize(packed_input0, config);
    auto tilized_input1 = ::unit_tests::compute::gold_standard_tilize(packed_input1, config);

    distributed::WriteShard(cq, src_a_dram_buffer, tilized_input0, zero_coord);
    distributed::WriteShard(cq, src_b_dram_buffer, tilized_input1, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, is_quasar);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    auto dest_buffer_data_untilized = ::unit_tests::compute::gold_standard_untilize(dest_buffer_data, config);

    bool result = is_close_packed_vectors<bfloat16, std::uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, k_broadcast_rtol);
        });
    ASSERT_TRUE(result);
}

// --------------------------------------------------------------------------
// SDPA blocked bcast-col SUB with SrcB reuse (sub_tiles_bcast_cols_custom)
// --------------------------------------------------------------------------
// Distinct data flow from the stock broadcast path above, which is why it gets its own config and
// runner rather than another BroadcastConfig axis: each row of a block is ct_dim column tiles that
// all reuse ONE srcB tile, the op writes ct_dim dest slots itself, and srcB is never re-read from
// L1. Calling the stock one-tile broadcast ct_dim times would instead need ct_dim copies of srcB.
//
// A block is an rt_dim x ct_dim tile grid that fills one acquired dest section, so srcA is a
// (num_blocks * rt_dim) x ct_dim grid of tiles and srcB is rt_dim tiles (row r of every block reuses srcB tile r).
// rt_dim > 1 is the case that mirrors sub_exp_block_bcast_cols(): nonzero srcA / srcB tile indices
// and a nonzero dest base, all within one tile_regs_acquire().

struct SubBcastColCustomConfig {
    std::uint32_t ct_dim = 1;
    std::uint32_t rt_dim = 1;
    std::uint32_t num_blocks = 1;
    TileShape tile_shape = TileShape::FULL_TILE;
    MathFidelity math_fidelity = MathFidelity::LoFi;  // SUB is LoFi-only on Quasar (fidelity phases are MUL-only)
};

void run_sub_bcast_col_custom(distributed::MeshDevice& mesh_device, const SubBcastColCustomConfig& test_config) {
    auto& cq = mesh_device.mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};

    tt_metal::Tile tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    const std::uint32_t tile_width = tile_dims.get_tile_shape()[1];
    const std::uint32_t tile_height = tile_dims.get_tile_shape()[0];
    const std::uint32_t single_tile_size = tile_width * tile_height * sizeof(bfloat16);
    const std::uint32_t tiles_per_block = test_config.rt_dim * test_config.ct_dim;
    const std::uint32_t total_tiles = tiles_per_block * test_config.num_blocks;
    // Tile rows of the srcA / output grid, across all blocks. Tile columns are ct_dim.
    const std::uint32_t total_tile_rows = test_config.rt_dim * test_config.num_blocks;

    log_info(
        tt::LogTest,
        "Testing sub_bcast_cols_custom ct_dim={} rt_dim={} num_blocks={} tile={{{}, {}}}",
        test_config.ct_dim,
        test_config.rt_dim,
        test_config.num_blocks,
        tile_height,
        tile_width);

    // srcA and the output are total_tile_rows x ct_dim grids; srcB is rt_dim tiles, each reused by
    // its row in every block.
    //
    // One page per buffer rather than one per tile, because the reader and writer kernels walk a
    // single DRAM bank linearly (bank_id 0, the address advancing by one tile per transfer). A
    // page-per-tile buffer is interleaved, so page k lands in bank k % num_banks -- two banks on
    // Quasar -- and a linear walk of bank 0 would read page 2k for tile k and run off the end of
    // that bank's share of the buffer once k passes half the tile count. Giving the allocator one
    // page puts the whole buffer in one bank, which is what these kernels address.
    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size * total_tiles, 1);
    std::uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();

    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size * test_config.rt_dim, 1);
    std::uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();

    auto dst_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size * total_tiles, 1);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    const bool is_quasar = mesh_device.arch() == ARCH::QUASAR;

    experimental::KernelSpec::CompilerOptions::Defines defines_vec;
    defines_vec.emplace("CT_DIM", std::to_string(test_config.ct_dim));
    defines_vec.emplace("RT_DIM", std::to_string(test_config.rt_dim));
    defines_vec.emplace("NUM_BLOCKS", std::to_string(test_config.num_blocks));

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name, std::uint32_t num_entries) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile_dims,
        };
    };

    // in0/out hold one whole block: the compute kernel indexes every row of the block off the read
    // pointer inside a single dest acquire, so all rt_dim * ct_dim tiles must be resident. in1 holds
    // the rt_dim reused tiles.
    experimental::DataflowBufferSpec inp0_dfb_spec = make_dfb(INP0_DFB, tiles_per_block);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_dfb(INP1_DFB, test_config.rt_dim);
    experimental::DataflowBufferSpec out_dfb_spec = make_dfb(OUT_DFB, tiles_per_block);

    experimental::DataMovementHardwareConfig reader_hw_config = make_reader_hw_config(mesh_device);
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_bcast_col_reuse.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "num_bcast_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config = make_writer_hw_config(mesh_device);
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::ComputeHardwareConfig compute_hw_config =
        make_compute_hw_config(mesh_device, test_config.math_fidelity);

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/sub_bcast_col_custom.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines_vec},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_sub_bcast_col_custom",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", static_cast<std::uint32_t>(dram_buffer_src_a_addr)},
                 {"src0_bank_id", 0u},
                 {"src1_addr", static_cast<std::uint32_t>(dram_buffer_src_b_addr)},
                 {"src1_bank_id", 0u},
                 // srcA tile count, then the srcB (bcast) tile count: one per row of a block.
                 {"num_tiles", static_cast<std::uint32_t>(total_tiles)},
                 {"num_bcast_tiles", static_cast<std::uint32_t>(test_config.rt_dim)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dram_buffer_dst_addr)},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(total_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    const std::uint32_t tile_elems = tile_height * tile_width;
    // Row-major host geometry of the srcA / output tile grid, and of the srcB tile column.
    const std::uint32_t grid_width = tile_width * test_config.ct_dim;
    const std::uint32_t grid_height = tile_height * total_tile_rows;
    const std::uint32_t bcast_height = tile_height * test_config.rt_dim;

    std::vector<bfloat16> input0 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, tile_elems * total_tiles, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> input1 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, tile_elems * test_config.rt_dim, std::chrono::system_clock::now().time_since_epoch().count());

    // No mask_src_b_for_broadcast call: it is a no-op for COL (see its body): The FPU reads
    // column 0 of each srcB row and ignores the rest, which is what gold_broadcast models.
    // Tile srcB unmasked so host and device see identical bytes.

    // Replicate srcB across the grid so gold_broadcast's COL case (src_b[i * num_cols]) resolves to
    // column 0 of the right srcB row: the same tile subtracted from every srcA column tile of its
    // row, and (r % bcast_height) makes every block reuse the same rt_dim srcB tiles.
    std::vector<bfloat16> input1_grid(tile_elems * total_tiles);
    for (std::uint32_t r = 0; r < grid_height; r++) {
        for (std::uint32_t c = 0; c < grid_width; c++) {
            input1_grid[(r * grid_width) + c] = input1[((r % bcast_height) * tile_width) + (c % tile_width)];
        }
    }

    std::vector<bfloat16> golden = gold_broadcast(
        input0,
        input1_grid,
        {grid_height, grid_width},
        EltwiseOp::SUB,
        BroadcastDim::COL,
        0,
        test_config.math_fidelity);

    auto packed_input0 = pack_vector<std::uint32_t, bfloat16>(input0);
    auto packed_input1 = pack_vector<std::uint32_t, bfloat16>(input1);
    auto packed_golden = pack_vector<std::uint32_t, bfloat16>(golden);

    const int num_faces = static_cast<int>(tile_width / 16 * tile_height / 16);
    const bool tiny_tile = test_config.tile_shape != TileShape::FULL_TILE;
    // gold_standard_tilize/untilize walk a tile grid row-major, which is exactly the DRAM page order
    // the reader streams into in0: block b owns tile rows [b * rt_dim, (b + 1) * rt_dim).
    ::unit_tests::compute::GoldenConfig grid_config = {
        .num_tiles_r_dim = static_cast<int>(total_tile_rows),
        .num_tiles_c_dim = static_cast<int>(test_config.ct_dim),
        .num_faces = num_faces,
        .tiny_tile = tiny_tile};
    // srcB is a single column of rt_dim tiles.
    ::unit_tests::compute::GoldenConfig bcast_config = {
        .num_tiles_r_dim = static_cast<int>(test_config.rt_dim),
        .num_tiles_c_dim = 1,
        .num_faces = num_faces,
        .tiny_tile = tiny_tile};

    auto tilized_input0 = ::unit_tests::compute::gold_standard_tilize(packed_input0, grid_config);
    auto tilized_input1 = ::unit_tests::compute::gold_standard_tilize(packed_input1, bcast_config);

    distributed::WriteShard(cq, src_a_dram_buffer, tilized_input0, zero_coord);
    distributed::WriteShard(cq, src_b_dram_buffer, tilized_input1, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, is_quasar);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    auto dest_buffer_data_untilized = ::unit_tests::compute::gold_standard_untilize(dest_buffer_data, grid_config);

    bool result = is_close_packed_vectors<bfloat16, std::uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, k_broadcast_rtol);
        });
    ASSERT_TRUE(result);
}

// --------------------------------------------------------------------------
// indexer_score blocked bcast-col MUL with SrcB reuse (mul_tiles_bcast_cols_custom), Blackhole only
// --------------------------------------------------------------------------
// Same block/data flow as the SUB runner above, with two differences that are the point of this
// runner: the op MACs onto dest instead of overwriting it (so rows of one block can accumulate into
// the same slots, as indexer_score reduces heads), and the OUTPUT buffer always carries full 32x32
// tiles. For 16x32 inputs the packer therefore reads the whole 64-row dest slot, so dest rows 32-63
// (which a one-face-row tile must never write) come back to the host as the lower half of every
// output tile and must be exactly zero.
//
// Every program starts with both Src banks zeroed, so a walk that read Src rows a 16x32 tile never
// wrote would MAC zeros and pass that check. 16x32 cases therefore first run one 2x2 block of nonzero
// full 32x32 tiles through the same op in the same kernel (see mul_bcast_col_custom.cpp), which
// leaves nonzero data in rows 32-63 of every SrcA and SrcB bank; its own output is checked against
// the full-tile golden.

struct MulBcastColCustomConfig {
    std::uint32_t ct_dim = 1;
    std::uint32_t rt_dim = 1;
    std::uint32_t num_blocks = 1;
    TileShape tile_shape = TileShape::FULL_TILE;
    // Every row of a block MACs onto dest slots [0, ct_dim) instead of its own slots.
    bool accumulate_rows = false;
    // The scaffold issues one ELWMUL per 8-row block (fidelity phase 0 only), so the result is LoFi
    // arithmetic whatever fidelity the kernel is configured with; the golden must model LoFi.
    MathFidelity math_fidelity = MathFidelity::LoFi;
};

void run_mul_bcast_col_custom(distributed::MeshDevice& mesh_device, const MulBcastColCustomConfig& test_config) {
    auto& cq = mesh_device.mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};
    constexpr std::uint32_t num_pages = 1;

    const tt_metal::Tile in_tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    const tt_metal::Tile out_tile_dims = tile_shape_to_tile.at(TileShape::FULL_TILE);
    const std::uint32_t tile_width = in_tile_dims.get_tile_shape()[1];
    const std::uint32_t in_tile_height = in_tile_dims.get_tile_shape()[0];
    const std::uint32_t out_tile_height = out_tile_dims.get_tile_shape()[0];
    const std::uint32_t in_tile_size = tile_width * in_tile_height * sizeof(bfloat16);
    const std::uint32_t out_tile_size = tile_width * out_tile_height * sizeof(bfloat16);
    const std::uint32_t tiles_per_block = test_config.rt_dim * test_config.ct_dim;
    const std::uint32_t total_tiles = tiles_per_block * test_config.num_blocks;
    const std::uint32_t total_tile_rows = test_config.rt_dim * test_config.num_blocks;
    // Accumulating rows folds a block's rt_dim tile rows into one.
    const std::uint32_t out_tiles_per_block = test_config.accumulate_rows ? test_config.ct_dim : tiles_per_block;
    const std::uint32_t out_tile_rows = test_config.accumulate_rows ? test_config.num_blocks : total_tile_rows;
    const std::uint32_t total_out_tiles = out_tiles_per_block * test_config.num_blocks;

    // Priming block of full tiles for tiny-tile inputs (see above): two rows so both SrcB banks are
    // written, at least two columns so both SrcA banks are. Its tile count is a multiple of the main
    // blocks' output batch and sizes the output ring, because pack_tile does not wrap inside a batch,
    // so every batch must start at a ring boundary.
    const bool tiny_tile = test_config.tile_shape != TileShape::FULL_TILE;
    const std::uint32_t prime_rt_dim = tiny_tile ? 2 : 0;
    const std::uint32_t prime_ct_dim = !tiny_tile ? 0
                                       : (out_tiles_per_block >= 4 && out_tiles_per_block % 2 == 0)
                                           ? out_tiles_per_block / 2
                                       : (out_tiles_per_block == 1) ? 2
                                                                    : out_tiles_per_block;
    const std::uint32_t prime_tiles = prime_rt_dim * prime_ct_dim;
    const std::uint32_t out_ring_tiles = tiny_tile ? prime_tiles : out_tiles_per_block;
    // 32x32 tiles one acquired dest section holds (16-bit dest, half sync).
    constexpr std::uint32_t max_dest_tiles_per_section = 8;
    TT_FATAL(
        out_ring_tiles % out_tiles_per_block == 0 && prime_tiles <= max_dest_tiles_per_section,
        "output ring must hold whole batches and the priming block at most {} tiles",
        max_dest_tiles_per_section);

    log_info(
        tt::LogTest,
        "Testing mul_bcast_cols_custom ct_dim={} rt_dim={} num_blocks={} accumulate_rows={} in_tile={{{}, {}}} "
        "out_tile={{{}, {}}} prime_tiles={}",
        test_config.ct_dim,
        test_config.rt_dim,
        test_config.num_blocks,
        test_config.accumulate_rows,
        in_tile_height,
        tile_width,
        out_tile_height,
        tile_width,
        prime_tiles);

    // One page per buffer: the reader/writer walk one DRAM bank linearly (see the SUB runner).
    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, in_tile_size * total_tiles, num_pages);
    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, in_tile_size * test_config.rt_dim, num_pages);
    auto dst_dram_buffer =
        CreateDramBufferForPageSize(mesh_device, out_tile_size * (prime_tiles + total_out_tiles), num_pages);
    std::shared_ptr<distributed::MeshBuffer> prime_a_dram_buffer;
    std::shared_ptr<distributed::MeshBuffer> prime_b_dram_buffer;
    if (tiny_tile) {
        prime_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, out_tile_size * prime_tiles, num_pages);
        prime_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, out_tile_size * prime_rt_dim, num_pages);
    }

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    compute_defines.emplace("CT_DIM", std::to_string(test_config.ct_dim));
    compute_defines.emplace("RT_DIM", std::to_string(test_config.rt_dim));
    compute_defines.emplace("NUM_BLOCKS", std::to_string(test_config.num_blocks));
    compute_defines.emplace("ACCUMULATE_ROWS", test_config.accumulate_rows ? "1" : "0");
    experimental::KernelSpec::CompilerOptions::Defines reader_defines;
    if (tiny_tile) {
        compute_defines.emplace("PRIME_FULL_TILES", "1");
        compute_defines.emplace("PRIME_RT_DIM", std::to_string(prime_rt_dim));
        compute_defines.emplace("PRIME_CT_DIM", std::to_string(prime_ct_dim));
        reader_defines.emplace("PRIME_FULL_TILES", "1");
    }

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName INP2_DFB{"inp2_dfb"};
    const experimental::DFBSpecName INP3_DFB{"inp3_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name,
                        std::uint32_t num_entries,
                        std::uint32_t entry_size,
                        const tt_metal::Tile& tile) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile,
        };
    };
    auto bind =
        [](const experimental::DFBSpecName& name, const std::string& accessor, experimental::DFBEndpointType endpoint) {
            return experimental::KernelSpec::DFBBinding{
                .dfb_spec_name = name,
                .accessor_name = accessor,
                .endpoint_type = endpoint,
                .access_pattern = experimental::DFBAccessPattern::STRIDED,
            };
        };

    std::vector<experimental::DataflowBufferSpec> dfb_specs = {
        make_dfb(INP0_DFB, tiles_per_block, in_tile_size, in_tile_dims),
        make_dfb(INP1_DFB, test_config.rt_dim, in_tile_size, in_tile_dims),
        make_dfb(OUT_DFB, out_ring_tiles, out_tile_size, out_tile_dims),
    };
    std::vector<experimental::KernelSpec::DFBBinding> reader_bindings = {
        bind(INP0_DFB, "in0", experimental::DFBEndpointType::PRODUCER),
        bind(INP1_DFB, "in1", experimental::DFBEndpointType::PRODUCER),
    };
    std::vector<experimental::KernelSpec::DFBBinding> compute_bindings = {
        bind(INP0_DFB, "in0", experimental::DFBEndpointType::CONSUMER),
        bind(INP1_DFB, "in1", experimental::DFBEndpointType::CONSUMER),
        bind(OUT_DFB, "out", experimental::DFBEndpointType::PRODUCER),
    };
    std::vector<std::string> reader_arg_names = {
        "src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "num_bcast_tiles"};
    if (tiny_tile) {
        dfb_specs.push_back(make_dfb(INP2_DFB, prime_tiles, out_tile_size, out_tile_dims));
        dfb_specs.push_back(make_dfb(INP3_DFB, prime_rt_dim, out_tile_size, out_tile_dims));
        reader_bindings.push_back(bind(INP2_DFB, "in2", experimental::DFBEndpointType::PRODUCER));
        reader_bindings.push_back(bind(INP3_DFB, "in3", experimental::DFBEndpointType::PRODUCER));
        compute_bindings.push_back(bind(INP2_DFB, "in2", experimental::DFBEndpointType::CONSUMER));
        compute_bindings.push_back(bind(INP3_DFB, "in3", experimental::DFBEndpointType::CONSUMER));
        for (const char* name :
             {"src2_addr", "src2_bank_id", "src3_addr", "src3_bank_id", "num_prime_tiles", "num_prime_bcast_tiles"}) {
            reader_arg_names.emplace_back(name);
        }
    }

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_bcast_col_reuse.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_bindings,
        .runtime_arg_schema = {.runtime_arg_names = reader_arg_names},
        .hw_config = make_reader_hw_config(mesh_device),
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = make_writer_hw_config(mesh_device),
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/mul_bcast_col_custom.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = compute_bindings,
        .hw_config = make_compute_hw_config(mesh_device, test_config.math_fidelity),
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_mul_bcast_col_custom",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dfb_specs,
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    experimental::AddRuntimeArgsForNode(
        reader_run_args.runtime_arg_values,
        node,
        {{"src0_addr", static_cast<std::uint32_t>(src_a_dram_buffer->address())},
         {"src0_bank_id", 0u},
         {"src1_addr", static_cast<std::uint32_t>(src_b_dram_buffer->address())},
         {"src1_bank_id", 0u},
         {"num_tiles", static_cast<std::uint32_t>(total_tiles)},
         {"num_bcast_tiles", static_cast<std::uint32_t>(test_config.rt_dim)}});
    if (tiny_tile) {
        experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            node,
            {{"src2_addr", static_cast<std::uint32_t>(prime_a_dram_buffer->address())},
             {"src2_bank_id", 0u},
             {"src3_addr", static_cast<std::uint32_t>(prime_b_dram_buffer->address())},
             {"src3_bank_id", 0u},
             {"num_prime_tiles", static_cast<std::uint32_t>(prime_tiles)},
             {"num_prime_bcast_tiles", static_cast<std::uint32_t>(prime_rt_dim)}});
    }

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        reader_run_args,
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dst_dram_buffer->address())},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(prime_tiles + total_out_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    // Row-major host side of one srcA grid x srcB tile column pair: random inputs, the bcast-col MUL
    // golden (every block reuses the same rt_dim srcB tiles), and the tilized device inputs.
    struct HostOperands {
        std::vector<bfloat16> input0;
        std::vector<bfloat16> input1;
        std::vector<bfloat16> golden;  // product grid, tile_height * tile_rows x width
        std::vector<std::uint32_t> tilized_input0;
        std::vector<std::uint32_t> tilized_input1;
    };
    auto make_operands = [&](std::uint32_t tile_height,
                             std::uint32_t rt_dim,
                             std::uint32_t ct_dim,
                             std::uint32_t num_blocks) -> HostOperands {
        const std::uint32_t tile_elems = tile_height * tile_width;
        const std::uint32_t tile_rows = rt_dim * num_blocks;
        const std::uint32_t grid_width = tile_width * ct_dim;
        const std::uint32_t grid_height = tile_height * tile_rows;
        const std::uint32_t bcast_height = tile_height * rt_dim;
        HostOperands ops;
        ops.input0 = generate_uniform_random_vector<bfloat16>(
            -1.0f, 1.0f, tile_elems * tile_rows * ct_dim, std::chrono::system_clock::now().time_since_epoch().count());
        ops.input1 = generate_uniform_random_vector<bfloat16>(
            -1.0f, 1.0f, tile_elems * rt_dim, std::chrono::system_clock::now().time_since_epoch().count());

        // Replicate srcB across the srcA grid so gold_broadcast's COL case reads column 0 of the right
        // srcB row; (r % bcast_height) makes every block reuse the same rt_dim srcB tiles.
        std::vector<bfloat16> input1_grid(tile_elems * tile_rows * ct_dim);
        for (std::uint32_t r = 0; r < grid_height; r++) {
            for (std::uint32_t c = 0; c < grid_width; c++) {
                input1_grid[(r * grid_width) + c] = ops.input1[((r % bcast_height) * tile_width) + (c % tile_width)];
            }
        }
        ops.golden = gold_broadcast(
            ops.input0,
            input1_grid,
            {grid_height, grid_width},
            EltwiseOp::MUL,
            BroadcastDim::COL,
            0,
            test_config.math_fidelity);

        const int num_faces = static_cast<int>(tile_width / 16 * tile_height / 16);
        const bool tiny = tile_height != out_tile_height;
        ::unit_tests::compute::GoldenConfig grid_config = {
            .num_tiles_r_dim = static_cast<int>(tile_rows),
            .num_tiles_c_dim = static_cast<int>(ct_dim),
            .num_faces = num_faces,
            .tiny_tile = tiny};
        ::unit_tests::compute::GoldenConfig bcast_config = {
            .num_tiles_r_dim = static_cast<int>(rt_dim),
            .num_tiles_c_dim = 1,
            .num_faces = num_faces,
            .tiny_tile = tiny};
        ops.tilized_input0 =
            ::unit_tests::compute::gold_standard_tilize(pack_vector<std::uint32_t, bfloat16>(ops.input0), grid_config);
        ops.tilized_input1 =
            ::unit_tests::compute::gold_standard_tilize(pack_vector<std::uint32_t, bfloat16>(ops.input1), bcast_config);
        return ops;
    };

    HostOperands main_ops =
        make_operands(in_tile_height, test_config.rt_dim, test_config.ct_dim, test_config.num_blocks);
    distributed::WriteShard(cq, src_a_dram_buffer, main_ops.tilized_input0, zero_coord);
    distributed::WriteShard(cq, src_b_dram_buffer, main_ops.tilized_input1, zero_coord);
    HostOperands prime_ops;
    if (tiny_tile) {
        prime_ops = make_operands(out_tile_height, prime_rt_dim, prime_ct_dim, 1);
        distributed::WriteShard(cq, prime_a_dram_buffer, prime_ops.tilized_input0, zero_coord);
        distributed::WriteShard(cq, prime_b_dram_buffer, prime_ops.tilized_input1, zero_coord);
    }

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    ASSERT_EQ(dest_buffer_data.size() * sizeof(std::uint32_t), out_tile_size * (prime_tiles + total_out_tiles));
    const auto prime_end = dest_buffer_data.begin() + (prime_tiles * out_tile_size / sizeof(std::uint32_t));

    auto golden_close = [&](const std::vector<std::uint32_t>& result_tilized, const std::vector<bfloat16>& golden) {
        return is_close_packed_vectors<bfloat16, std::uint32_t>(
            result_tilized, pack_vector<std::uint32_t, bfloat16>(golden), [&](const bfloat16& a, const bfloat16& b) {
                return is_close(a, b, k_broadcast_rtol);
            });
    };

    // Priming block: plain full-tile products, packed first.
    if (tiny_tile) {
        const std::vector<std::uint32_t> prime_result(dest_buffer_data.begin(), prime_end);
        ::unit_tests::compute::GoldenConfig prime_config = {
            .num_tiles_r_dim = static_cast<int>(prime_rt_dim),
            .num_tiles_c_dim = static_cast<int>(prime_ct_dim),
            .num_faces = 4,
            .tiny_tile = false};
        auto prime_untilized = ::unit_tests::compute::gold_standard_untilize(prime_result, prime_config);
        EXPECT_TRUE(golden_close(prime_untilized, prime_ops.golden)) << "priming full-tile block";
    }

    // Main blocks. Output tile row t holds the product rows of input tile row t (or, accumulating,
    // the sum over the block's rt_dim input tile rows) in its first in_tile_height rows; the rest of
    // the 32-row tile maps to dest rows the input tile does not own and must stay at the
    // acquire-zeroed value.
    const std::uint32_t grid_width = tile_width * test_config.ct_dim;
    std::vector<bfloat16> golden(out_tile_height * out_tile_rows * grid_width, bfloat16(0.0f));
    const std::uint32_t rows_per_out_row = test_config.accumulate_rows ? test_config.rt_dim : 1;
    for (std::uint32_t t = 0; t < out_tile_rows; t++) {
        for (std::uint32_t r = 0; r < in_tile_height; r++) {
            for (std::uint32_t c = 0; c < grid_width; c++) {
                float sum = 0.0f;
                for (std::uint32_t k = 0; k < rows_per_out_row; k++) {
                    const std::uint32_t in_tile_row = (t * rows_per_out_row) + k;
                    sum += static_cast<float>(main_ops.golden[(((in_tile_row * in_tile_height) + r) * grid_width) + c]);
                }
                golden[(((t * out_tile_height) + r) * grid_width) + c] = bfloat16(sum);
            }
        }
    }

    const std::vector<std::uint32_t> main_result(prime_end, dest_buffer_data.end());
    ::unit_tests::compute::GoldenConfig out_grid_config = {
        .num_tiles_r_dim = static_cast<int>(out_tile_rows),
        .num_tiles_c_dim = static_cast<int>(test_config.ct_dim),
        .num_faces = 4,
        .tiny_tile = false};
    auto result_untilized = ::unit_tests::compute::gold_standard_untilize(main_result, out_grid_config);
    auto result = unpack_vector<bfloat16, std::uint32_t>(result_untilized);
    ASSERT_EQ(result.size(), golden.size());

    // Exact check of the rows the input tile does not own: any nonzero there means the FPU walked a
    // face-row that was never unpacked for this tile shape (stale SrcA/SrcB MAC'd into dest).
    std::uint32_t stray_datums = 0;
    std::string stray_rows;
    for (std::uint32_t t = 0; t < out_tile_rows; t++) {
        std::uint32_t stray_in_row = 0;
        for (std::uint32_t r = in_tile_height; r < out_tile_height; r++) {
            for (std::uint32_t c = 0; c < grid_width; c++) {
                if (static_cast<float>(result[(((t * out_tile_height) + r) * grid_width) + c]) != 0.0f) {
                    stray_in_row++;
                }
            }
        }
        stray_datums += stray_in_row;
        stray_rows += " tile_row " + std::to_string(t) + ": " + std::to_string(stray_in_row);
    }
    EXPECT_EQ(stray_datums, 0u) << "nonzero datums in dest rows the " << in_tile_height << "x" << tile_width
                                << " input tile does not own;" << stray_rows;

    // Accumulated rows can cancel, so a pure relative tolerance is too tight there: allow the
    // per-product rounding of every summed row as an absolute term (|product| <= 1 for these inputs).
    const float atol = test_config.accumulate_rows ? test_config.rt_dim * k_broadcast_rtol : 0.001f;
    bool close = is_close_packed_vectors<bfloat16, std::uint32_t>(
        result_untilized, pack_vector<std::uint32_t, bfloat16>(golden), [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, k_broadcast_rtol, atol);
        });
    // Non-fatal so every case in the caller's list reports its own verdict.
    EXPECT_TRUE(close);
}
}  // namespace unit_tests::compute::broadcast

class BroadcastParameterizedDeviceFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<unit_tests::compute::broadcast::BroadcastConfig> {};

TEST_P(BroadcastParameterizedDeviceFixture, TensixComputeSingleTileBroadcast) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar uses TensixComputeBinaryBroadcastQuasarDfb";
    }
    unit_tests::compute::broadcast::BroadcastConfig test_config = GetParam();
    test_config.math_fidelity = MathFidelity::HiFi2;
    unit_tests::compute::broadcast::run_single_core_broadcast(*this->devices_.at(0), test_config);
}

using namespace unit_tests::compute::broadcast;

INSTANTIATE_TEST_SUITE_P(
    ComputeSingleTileBroadcast,
    BroadcastParameterizedDeviceFixture,
    ::testing::Values(
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::SCALAR},
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
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          15},  // Row 15 (middle)
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::SUB,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          15},  // Row 15 (middle)
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          31},  // Row 31 (last)
        (BroadcastConfig){ApiConvention::SHORT_CALL,
                          EltwiseOp::MUL,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          31},  // Row 31 with MUL
        (BroadcastConfig){ApiConvention::SHORT_BOTH,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          20}));  // Row 20

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixComputeBinaryBroadcastQuasarDfb) {
    for (std::uint8_t op = std::uint8_t(EltwiseOp::ADD); op <= std::uint8_t(EltwiseOp::MUL); op++) {
        for (std::uint8_t dim = std::uint8_t(BroadcastDim::ROW); dim <= std::uint8_t(BroadcastDim::SCALAR); dim++) {
            for (std::uint8_t math_fid = std::uint8_t(MathFidelity::LoFi);
                 math_fid <= std::uint8_t(MathFidelity::HiFi4);
                 math_fid++) {
                // MathFidelity : {0, 2, 3, 4};
                if (math_fid == 1) {
                    continue;
                }
                if (!(EltwiseOp(op) == EltwiseOp::ADD && BroadcastDim(dim) == BroadcastDim::ROW &&
                      MathFidelity(math_fid) == MathFidelity::LoFi)) {
                    // TODO (#38092): Remove when we can run back to back tests on Quasar
                    continue;
                }
                unit_tests::compute::broadcast::BroadcastConfig cfg = {
                    .api_convention = ApiConvention::DEFAULT,
                    .eltwise_op = EltwiseOp(op),
                    .broadcast_dim = BroadcastDim(dim),
                    .tile_shape = TileShape::FULL_TILE,
                    .math_fidelity = MathFidelity(math_fid),
                    .bcast_row_idx = 0,
                };
                log_info(
                    tt::LogTest,
                    "Quasar binary broadcast DFB op={} dim={} math_fid={}",
                    eltwise_op_to_type.at(EltwiseOp(op)),
                    broadcast_dim_to_type.at(BroadcastDim(dim)),
                    math_fid);
                unit_tests::compute::broadcast::run_single_core_broadcast(this->device(), cfg);
            }
        }
    }
}

// Blocked bcast-col SUB with SrcB reuse. All cases run back-to-back in one TEST_F.
// Cases are ordered so the first red one localizes the bug:
//   - ct_dim=1 carries no reuse at all, so a failure there means the srcB face traversal (the
//     +8/-8/+24 addr-mod walk) is wrong.
//   - ct_dim>1 adds the reuse itself: a failure there means the srcB hold is wrong i.e. the
//     per-tile SETRWC(CLR_A) is releasing srcB, or the L1 srcB tile pointer is advancing.
//   - num_blocks>1 additionally exercises dest-section switching and the per-block srcB re-unpack,
//     whose 1:1 pairing with the math thread's single CLR_B is what keeps blocks from deadlocking.
//   - rt_dim>1 is the only shape that exercises nonzero srcA/srcB tile indices and a nonzero dest
//     base, so a failure there (with the rt_dim=1 cases green) points at the API wrappers' index
//     arithmetic rather than the face walk.
//   - the tiny-tile (16x32) cases come last and mostly repeat a full-tile shape, so a red tiny-tile
//     case with its full-tile twin green isolates the shape-driven parts (face-row count, dest slot
//     stride, per-face unpack, buffer-descriptor z_dim).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, ComputeSubBcastColCustom) {
    using unit_tests::compute::broadcast::SubBcastColCustomConfig;

    const std::vector<SubBcastColCustomConfig> cases = {
        // Full 32x32 tiles, one row per block: every call uses tile_index_a/b = 0 and dst_index = 0.
        // ct_dim=8 fills half-dest, 7 covers a non-power-of-two block width.
        {.ct_dim = 1, .num_blocks = 1},
        {.ct_dim = 4, .num_blocks = 1},
        {.ct_dim = 7, .num_blocks = 1},
        {.ct_dim = 8, .num_blocks = 2},
        // One multi-row block, mirroring sub_exp_block_bcast_cols(): row r reads srcA tile r * ct_dim
        // and srcB tile r and writes dest slots [r * ct_dim, (r + 1) * ct_dim), all inside one
        // acquire. rt_dim * ct_dim is capped by the same half-dest budget as ct_dim above.
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1},
        // Tiny 16x32 tiles: one face-row, so the srcB walk runs its four-op face-row block once and
        // dest slots are 32 rows apart instead of 64. The full-tile shapes above are mirrored one for
        // one, because the tile shape changes the per-tile unpack (one UNPACR per face on Quasar), the
        // dest slot stride and the pack granularity all at once. The one shape with no full-tile twin
        // is ct_dim=16: a 32-row slot means twice as many tiles fit in a dest section, so this is the
        // only case that reaches the tiny-tile dest budget (and the only one where dst_index + ct_dim
        // lands exactly on the shape-derived max_dest_tiles bound).
        {.ct_dim = 1, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 4, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 7, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 16, .num_blocks = 2, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
    };

    for (const auto& cfg : cases) {
        SCOPED_TRACE(
            "ct_dim=" + std::to_string(cfg.ct_dim) + " rt_dim=" + std::to_string(cfg.rt_dim) + " num_blocks=" +
            std::to_string(cfg.num_blocks) + " tiny_tile=" + std::to_string(cfg.tile_shape != TileShape::FULL_TILE));
        unit_tests::compute::broadcast::run_sub_bcast_col_custom(this->device(), cfg);
        if (HasFatalFailure()) {
            // Later cases are not diagnostic once an earlier one is red.
            break;
        }
    }
}

// ============================================================================
// Id-free (2.0) binary broadcast MUL (rows / cols / scalar), validated against the gold_broadcast host golden
// (not a legacy kernel). c_0 = A (full tile), c_1 = B (masked bcast tile), MUL-broadcast -> c_16. Both inputs
// are tilized host-side (the device reads tiled operands); the golden broadcast is computed row-major then
// tilized. Single Float16_b tile, is_close(rtol 0.0155). Runs on Blackhole (BH-only API).
// ============================================================================
namespace {
// Map the test-side BroadcastDim to the kernel-side ckernel BroadcastType numeric value (COL=1, ROW=2,
// SCALAR=3) passed as get_compile_time_arg_val(1) to the fused bcast_mul_2_0.cpp kernel.
std::uint32_t bcast_type_arg(BroadcastDim dim) {
    switch (dim) {
        case BroadcastDim::ROW: return 2;
        case BroadcastDim::COL: return 1;
        case BroadcastDim::SCALAR: return 3;
    }
    return 0;
}

void expect_bcast_mul_matches_golden(distributed::MeshDevice& md, BroadcastDim dim) {
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/compute/bcast_mul_2_0.cpp";
    ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .face_r_dim = 16, .face_c_dim = 16, .num_faces = 4};
    auto in0_packed = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b), /*rand_max_float=*/2, /*seed=*/42, /*offset=*/-1.0f);
    auto in1_packed = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b), /*rand_max_float=*/2, /*seed=*/7, /*offset=*/-1.0f);

    auto in0_bf16 = unpack_vector<bfloat16, std::uint32_t>(in0_packed);
    auto in1_bf16 = unpack_vector<bfloat16, std::uint32_t>(in1_packed);
    // Zero the non-broadcast positions of B so device and golden read the same broadcast operand.
    mask_src_b_for_broadcast(in1_bf16, {32, 32}, dim, /*row_idx=*/0);
    auto golden_rm = gold_broadcast(in0_bf16, in1_bf16, {32, 32}, EltwiseOp::MUL, dim, /*row_idx=*/0);

    auto c0_input = ::unit_tests::compute::gold_standard_tilize(in0_packed, config);
    auto c1_input = ::unit_tests::compute::gold_standard_tilize(pack_vector<std::uint32_t, bfloat16>(in1_bf16), config);
    auto golden_tiled =
        ::unit_tests::compute::gold_standard_tilize(pack_vector<std::uint32_t, bfloat16>(golden_rm), config);

    auto result = unit_tests::llk::single_core::run_binary(
        md,
        c0_input,
        c1_input,
        /*num_tiles=*/1,
        kernel,
        /*compute_defines=*/{},
        /*cb_depth_tiles=*/1,
        /*out_tiles=*/0,
        /*extra_compile_args=*/{bcast_type_arg(dim)});

    EXPECT_EQ(golden_tiled.size(), result.size());
    bool close = is_close_packed_vectors<bfloat16, std::uint32_t>(
        result, golden_tiled, [](const bfloat16& a, const bfloat16& b) { return is_close(a, b, k_broadcast_rtol); });
    EXPECT_TRUE(close);
}
}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, TensixBcastMulRowsIdFreeGolden) {
    expect_bcast_mul_matches_golden(this->device(), BroadcastDim::ROW);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixBcastMulColsIdFreeGolden) {
    expect_bcast_mul_matches_golden(this->device(), BroadcastDim::COL);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixBcastMulScalarIdFreeGolden) {
    expect_bcast_mul_matches_golden(this->device(), BroadcastDim::SCALAR);
}

// Blocked bcast-col MUL with SrcB reuse through the Compute API, Blackhole only (the MUL scaffold and
// its wrapper exist only there). 16x32 cases prime the Src banks with nonzero full tiles inside the
// same program (see the runner), so a walk over a second face-row shows up as nonzero data in dest
// rows 32-63 rather than as zeros MAC'd onto zeros.
TEST_F(LLKBlackholeSingleCardFixture, ComputeMulBcastColCustom) {
    using unit_tests::compute::broadcast::MulBcastColCustomConfig;
    using unit_tests::compute::broadcast::TileShape;

    const std::vector<MulBcastColCustomConfig> cases = {
        // Full tiles: the 32x32 path.
        {.ct_dim = 4, .num_blocks = 2},
        // 16x32 inputs, 32x32 outputs: the lower half of every output tile must be exactly zero.
        {.ct_dim = 4, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 1, .num_blocks = 2, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 2, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        // Multi-row blocks: row r must land in dest slots [r * ct_dim, (r + 1) * ct_dim), i.e. the
        // wrapper must honour a nonzero dst_index.
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1},
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        // Rows accumulated into the same slots (the indexer_score head reduction): the op must MAC
        // onto dest, not overwrite it.
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1, .accumulate_rows = true},
        {.ct_dim = 2, .rt_dim = 3, .num_blocks = 2, .tile_shape = TileShape::TINY_TILE_16x32, .accumulate_rows = true},
    };

    for (const auto& cfg : cases) {
        SCOPED_TRACE(
            "ct_dim=" + std::to_string(cfg.ct_dim) + " rt_dim=" + std::to_string(cfg.rt_dim) + " num_blocks=" +
            std::to_string(cfg.num_blocks) + " tiny_tile=" + std::to_string(cfg.tile_shape != TileShape::FULL_TILE) +
            " accumulate_rows=" + std::to_string(cfg.accumulate_rows));
        unit_tests::compute::broadcast::run_mul_bcast_col_custom(this->device(), cfg);
        if (HasFatalFailure()) {
            break;
        }
    }
}

// Compile-time contract of the custom bcast-col shape guards (SUB on Blackhole and Wormhole, MUL on
// Blackhole): the real unpack-init and math-execute entry points, handed a literal shape, must build
// for 32x32 and 16x32 and must fail to build for a one-face-column (32x16) or short-face (8x32) tile,
// with the guard's own message in the compiler log. A guard that stopped using the strict predicate
// would let its rejected cases compile. Each entry point is its own site so none hides another.
// Nothing is launched.
TEST_F(LLKMeshDeviceFixture, ComputeBcastColCustomShapeCompileContract) {
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "the guards under test are the Wormhole and Blackhole ones";
    }
    struct ShapeCase {
        std::uint32_t face_r_dim;
        std::uint32_t num_faces_r_dim;
        std::uint32_t num_faces_c_dim;
        bool supported;
        const char* name;
    };
    const std::array<ShapeCase, 4> shapes{{
        {16, 2, 2, true, "32x32"},
        {16, 1, 2, true, "16x32"},
        {16, 2, 1, false, "32x16"},
        {8, 1, 2, false, "8x32"},
    }};
    // Site 2 is the Blackhole-only MUL scaffold entry.
    const std::array<const char*, 3> site_names{{"unpack init", "SUB math execute", "MUL math execute"}};
    const std::uint32_t num_sites = this->arch_ == tt::ARCH::BLACKHOLE ? 3 : 2;
    constexpr const char* guard_message = "custom bcast-col path supports 32x32 and 16x32 tiles only";
    auto* device = this->devices_.at(0).get();
    for (std::uint32_t site = 0; site < num_sites; site++) {
        for (const auto& shape : shapes) {
            SCOPED_TRACE(::testing::Message() << site_names.at(site) << ", " << shape.name);
            Program program = CreateProgram();
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/bcast_col_custom_shape_contract.cpp",
                CoreCoord{0, 0},
                ComputeConfig{.compile_args = {site, shape.face_r_dim, shape.num_faces_r_dim, shape.num_faces_c_dim}});
            if (shape.supported) {
                EXPECT_NO_THROW(program.impl().compile(device));
                continue;
            }
            // The build must fail, and for the guard's reason: the JIT exception carries the compiler log.
            try {
                program.impl().compile(device);
                ADD_FAILURE() << "kernel built with an unsupported shape";
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find(guard_message), std::string::npos)
                    << "build failed for another reason: " << e.what();
            }
        }
    }
}

}  // namespace tt::tt_metal
