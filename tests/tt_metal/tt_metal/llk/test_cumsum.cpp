// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::cumsum {

struct CumsumConfig {
    uint32_t N;
    uint32_t Wt;
    uint32_t Ht;
    bool rowwise;
};

std::vector<bfloat16> gold_cumsum(std::vector<bfloat16>& src, const std::vector<uint32_t>& shape, bool rowwise) {
    int N = shape.at(0);
    int W = shape.at(1);
    int H = shape.at(2);

    std::vector<bfloat16> golden(N * W * H);

    int dim_a = rowwise ? H : W;
    int dim_b = rowwise ? W : H;
    int j_mul = rowwise ? 1 : W;
    int k_mul = rowwise ? W : 1;

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < dim_a; k++) {
            float res = 0;
            for (int j = 0; j < dim_b; j++) {
                res += static_cast<float>(src[(i * W * H) + (j * j_mul) + (k * k_mul)]);
                golden[(i * W * H) + (j * j_mul) + (k * k_mul)] = res;
            }
        }
    }

    return golden;
}

// A flat DRAM-interleaved buffer of `num_tiles` tile-sized pages, bound to the reader/writer as a tensor.
static TensorSpec make_flat_dram_tensor_spec(uint32_t tile_size, uint32_t num_tiles) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{num_tiles, tile_size / static_cast<uint32_t>(sizeof(uint32_t))}, tensor_layout);
}

// Quasar runs the data movement kernels on its 2xx DM cores; everything else uses the 1xx NCRISC/BRISC pair.
static experimental::DataMovementHardwareConfig make_dm_hw_config(
    const distributed::MeshDevice& mesh_device, DataMovementProcessor processor, NOC noc) {
    if (mesh_device.arch() == ARCH::QUASAR) {
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
                .processor = processor,
                .noc = noc,
            },
    };
}

void run_single_core_cumsum(distributed::MeshDevice& mesh_device, const CumsumConfig& test_config) {
    const experimental::NodeCoord node{0, 0};

    constexpr uint32_t single_tile_size = constants::TILE_HW * sizeof(bfloat16);
    constexpr uint32_t num_buffer_tiles = 2;

    const uint32_t W = test_config.Wt * constants::TILE_WIDTH;
    const uint32_t H = test_config.Ht * constants::TILE_HEIGHT;
    const uint32_t HtWt = test_config.Ht * test_config.Wt;
    const uint32_t num_tiles = test_config.N * HtWt;
    const uint32_t dram_buffer_size = single_tile_size * num_tiles;

    auto in_tensor =
        MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(single_tile_size, num_tiles));
    auto out_tensor =
        MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(single_tile_size, num_tiles));

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};
    const experimental::TensorParamName OUT_TENSOR{"out_tensor"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = DataFormat::Float16_b,
    };

    // Columnwise chains cumsum down H, so tiles stream in (and back out) in NWH order. Rowwise transposes each
    // tile in and out of Dest, so the plain NHW tile order already chains across W.
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .hw_config = make_dm_hw_config(mesh_device, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default),
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .hw_config = make_dm_hw_config(mesh_device, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
    };
    experimental::ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = READER};
    experimental::ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = WRITER};

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    uint32_t compute_ht = test_config.Ht;
    uint32_t compute_wt = test_config.Wt;

    if (test_config.rowwise) {
        reader_spec.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp";
        reader_spec.compile_time_args = {{"num_entries_per_producer", num_tiles}, {"implicit_sync", 0}};
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
        reader_run_args.runtime_arg_values =
            experimental::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0}, {"entries_per_core", num_tiles}});

        writer_spec.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank_2_0.cpp";
        writer_spec.runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}};
        writer_run_args.runtime_arg_values =
            experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}});

        compute_defines.emplace("ROWWISE", "1");
        std::swap(compute_ht, compute_wt);
    } else {
        auto make_nwh_args = [&]() {
            return experimental::MakeRuntimeArgsForSingleNode(
                node, {{"N", test_config.N}, {"Ht", test_config.Ht}, {"Wt", test_config.Wt}, {"HtWt", HtWt}});
        };

        reader_spec.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_8bank.cpp";
        reader_spec.runtime_arg_schema = {.runtime_arg_names = {"N", "Ht", "Wt", "HtWt"}};
        reader_run_args.runtime_arg_values = make_nwh_args();

        writer_spec.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_transpose_wh.cpp";
        writer_spec.runtime_arg_schema = {.runtime_arg_names = {"N", "Ht", "Wt", "HtWt"}};
        writer_run_args.runtime_arg_values = make_nwh_args();
    }

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/cumsum.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"Ht", compute_ht}, {"Wt", compute_wt}, {"NC", test_config.N}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "cumsum",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {reader_run_args, writer_run_args, {.kernel = COMPUTE}};
    params.tensor_args = {
        {IN_TENSOR, experimental::ProgramRunArgs::TensorArgument{in_tensor}},
        {OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{out_tensor}},
    };
    experimental::SetProgramRunArgs(program, params);

    // Fixed seed so a failure reproduces across runs.
    constexpr uint32_t kRandomSeed = 0x1234;
    std::vector<bfloat16> input =
        generate_uniform_random_vector<bfloat16>(-1.0f, 1.0f, dram_buffer_size / sizeof(bfloat16), kRandomSeed);

    std::vector<bfloat16> golden = gold_cumsum(input, {test_config.N, W, H}, test_config.rowwise);
    auto golden_packed = pack_vector<uint32_t, bfloat16>(golden);

    auto input_packed = pack_vector<uint32_t, bfloat16>(input);
    auto input_packed_tilized =
        ::unit_tests::compute::gold_standard_tilize(input_packed, {test_config.N * test_config.Ht, test_config.Wt});

    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input_packed_tilized);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output_packed_tilized;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output_packed_tilized);
    auto output_packed = ::unit_tests::compute::gold_standard_untilize(
        output_packed_tilized, {test_config.N * test_config.Ht, test_config.Wt});

    log_info(tt::LogTest, "Running test for N = {}, Wt = {}, Ht = {}", test_config.N, test_config.Wt, test_config.Ht);

    bool result = is_close_packed_vectors<bfloat16, uint32_t>(
        output_packed, golden_packed, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.01f); });
    ASSERT_TRUE(result);
}

// Every (N, Wt, Ht) in [1, 3]^3. The Quasar emulator is too slow for the full sweep, so there it runs a subset that
// still covers a single tile, the carry chain across Ht, several Wt columns and several batches.
std::vector<CumsumConfig> make_sweep(ARCH arch, bool rowwise) {
    std::vector<CumsumConfig> configs;
    if (arch == ARCH::QUASAR) {
        for (const auto& [n, wt, ht] :
             std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{1, 1, 1}, {1, 1, 3}, {1, 3, 1}, {2, 2, 2}}) {
            configs.push_back({.N = n, .Wt = wt, .Ht = ht, .rowwise = rowwise});
        }
        return configs;
    }
    for (uint32_t n = 1; n <= 3; n++) {
        for (uint32_t wt = 1; wt <= 3; wt++) {
            for (uint32_t ht = 1; ht <= 3; ht++) {
                configs.push_back({.N = n, .Wt = wt, .Ht = ht, .rowwise = rowwise});
            }
        }
    }
    return configs;
}

}  // namespace unit_tests::compute::cumsum

TEST_F(LLKMeshDeviceFixture, TensixComputeCumsumColumnwise) {
    for (const auto& test_config : unit_tests::compute::cumsum::make_sweep(this->arch_, /*rowwise=*/false)) {
        unit_tests::compute::cumsum::run_single_core_cumsum(*this->devices_.at(0), test_config);
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeCumsumRowwise) {
    for (const auto& test_config : unit_tests::compute::cumsum::make_sweep(this->arch_, /*rowwise=*/true)) {
        unit_tests::compute::cumsum::run_single_core_cumsum(*this->devices_.at(0), test_config);
    }
}

}  // namespace tt::tt_metal
