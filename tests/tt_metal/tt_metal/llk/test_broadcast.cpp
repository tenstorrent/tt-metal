// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
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

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::broadcast {

enum ApiConvention : uint8_t {
    DEFAULT = 0,
    SHORT_INIT = 1,  // call <op>_bcast_<dim>_init_short instead of init_bcast
    SHORT_CALL = 2,  // call <op>_tiles_bcast_<dim> instead of <op>_tiles_bcast
    SHORT_BOTH = 3   // both SHORT_INIT and SHORT_CALL
};

enum EltwiseOp : uint8_t { ADD = 0, SUB = 1, MUL = 2 };

enum BroadcastDim : uint8_t { ROW = 0, COL = 1, SCALAR = 2 };

enum TileShape : uint8_t { FULL_TILE = 0, TINY_TILE_16x32 = 1 };

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

struct BroadcastConfig {
    ApiConvention api_convention;
    EltwiseOp eltwise_op;
    BroadcastDim broadcast_dim;
    TileShape tile_shape = TileShape::FULL_TILE;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t bcast_row_idx = 0;
};

void mask_src_b_for_broadcast(
    std::vector<bfloat16>& tile, const std::vector<uint32_t>& shape, BroadcastDim dim, uint32_t row_idx = 0) {
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
    const std::vector<uint32_t>& shape,
    EltwiseOp op,
    BroadcastDim dim,
    uint32_t row_idx = 0,
    MathFidelity math_fidelity = MathFidelity::HiFi4) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    uint16_t srca_fid_mask = 0xFFFF;
    uint16_t srcb_fid_mask = 0xFFFF;

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
                        static_cast<float>(std::bit_cast<bfloat16>(static_cast<uint16_t>(
                            std::bit_cast<uint16_t>(src_a[(i * num_cols) + j]) & srca_fid_mask))) *
                        static_cast<float>(std::bit_cast<bfloat16>(
                            static_cast<uint16_t>(std::bit_cast<uint16_t>(broadcast_value) & srcb_fid_mask)));
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

constexpr uint32_t k_num_tiles_broadcast_test = 1;

auto CreateDramBufferForPageSize(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t page_size_bytes, uint32_t num_pages) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = page_size_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = page_size_bytes * num_pages};
    return distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
}

void run_single_core_broadcast(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const BroadcastConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};

    tt_metal::Tile tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    uint32_t tile_width = tile_dims.get_tile_shape()[1];
    uint32_t tile_height = tile_dims.get_tile_shape()[0];
    if (test_config.tile_shape != TileShape::FULL_TILE) {
        log_info(tt::LogTest, "Tile shape is {{{}, {}}}", tile_height, tile_width);
    }

    uint32_t single_tile_size = tile_width * tile_height * sizeof(bfloat16);

    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();

    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();

    auto dst_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto* device = mesh_device->get_devices().empty() ? nullptr : mesh_device->get_devices().front();
    TT_FATAL(device != nullptr, "mesh_device has no backing devices");
    const bool is_quasar = device->arch() == ARCH::QUASAR;

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
                                   broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";

        if ((test_config.eltwise_op == EltwiseOp::SUB || test_config.eltwise_op == EltwiseOp::MUL) &&
            test_config.broadcast_dim == BroadcastDim::SCALAR) {
            // FIXME sub_bcast_scalar_init_short and mul_bcast_scalar_init_short are instead called
            // sub_tiles_bcast_scalar_init_short and mul_tiles_bcast_scalar_init_short
            defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast_" +
                                       broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";
        }

        log_info(tt::LogTest, "Init function is {}", defines["BCAST_OP_INIT"]);
    } else {
        log_info(tt::LogTest, "Init function is init_bcast");
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
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{
                        .disable_implicit_sync_for = {INP0_DFB, INP1_DFB}}},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{.disable_implicit_sync_for = {OUT_DFB}}},
    };

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
        .hw_config =
            experimental::ComputeHardwareConfig{
                .math_fidelity = test_config.math_fidelity,
            },
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

    Program built_program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src0_addr", static_cast<uint32_t>(dram_buffer_src_a_addr)},
                   {"src0_bank_id", 0u},
                   {"src1_addr", static_cast<uint32_t>(dram_buffer_src_b_addr)},
                   {"src1_bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(k_num_tiles_broadcast_test)}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", static_cast<uint32_t>(dram_buffer_dst_addr)},
                   {"bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(k_num_tiles_broadcast_test)}}}},
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

    auto packed_input0 = pack_vector<uint32_t, bfloat16>(input0);
    auto packed_input1 = pack_vector<uint32_t, bfloat16>(input1);
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);
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

    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    auto dest_buffer_data_untilized = ::unit_tests::compute::gold_standard_untilize(dest_buffer_data, config);

    bool result = is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155);
        });
    ASSERT_TRUE(result);
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
    unit_tests::compute::broadcast::run_single_core_broadcast(this->devices_.at(0), test_config);
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

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixComputeBinaryBroadcastQuasarDfb) {
    for (uint8_t op = uint8_t(EltwiseOp::ADD); op <= uint8_t(EltwiseOp::MUL); op++) {
        for (uint8_t dim = uint8_t(BroadcastDim::ROW); dim <= uint8_t(BroadcastDim::SCALAR); dim++) {
            for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
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
                unit_tests::compute::broadcast::run_single_core_broadcast(this->devices_.at(0), cfg);
            }
        }
    }
}

}  // namespace tt::tt_metal
