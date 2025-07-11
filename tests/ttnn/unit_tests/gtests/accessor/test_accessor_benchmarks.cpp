// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/ttnn/unit_tests/gtests/accessor/common.hpp"

#include <string>
#include <cstdint>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include <ttnn/tensor/tensor_accessor_args.hpp>

namespace accessor_benchmarks {

// TODO: Very similar to test_buffer_distribution_spec.cpp; refactor to common header?
struct InputBufferParams {
    std::string test_name;  // Used for generating unique files
    tt::tt_metal::Shape physical_tensor_shape;
    tt::tt_metal::Shape2D page_shape;
    float bytes_per_element;
    tt::DataFormat data_format;  // Used for setting up CBs

    struct DistributionSpecParams {
        tt::tt_metal::Shape physical_shard_shape;
        tt::tt_metal::CoreRangeSet grid;
        tt::tt_metal::ShardOrientation shard_orientation;
        tt::tt_metal::BufferType buffer_type;
    };
    DistributionSpecParams input_shard_spec;
};

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_replicated_input_mesh_buffer_from_inputs(
    const InputBufferParams& inputs, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    // These values would be passed from tensor correctly based on PageConfig
    const auto host_size_in_bytes = inputs.physical_tensor_shape.volume() * inputs.bytes_per_element;
    const auto page_size = inputs.page_shape.height() * inputs.page_shape.width() * inputs.bytes_per_element;

    // Mirrors allocate_mesh_buffer_on_device in ttnn
    const tt::tt_metal::distributed::ReplicatedBufferConfig mesh_buffer_config{.size = host_size_in_bytes};

    // Create input mesh buffer
    auto input_buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.input_shard_spec.physical_shard_shape,
        inputs.page_shape,
        inputs.input_shard_spec.grid,
        inputs.input_shard_spec.shard_orientation);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig input_device_local_config{
        .page_size = page_size,
        .buffer_type = inputs.input_shard_spec.buffer_type,
        .sharding_args = input_buffer_distribution_spec,
    };
    const auto input_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, input_device_local_config, mesh_device);

    return input_mesh_buffer;
}

}  // namespace accessor_benchmarks

using namespace accessor_benchmarks;
using namespace tt::tt_metal;

class AccessorBenchmarks : public GenericMeshDeviceFixture, public ::testing::WithParamInterface<InputBufferParams> {};

void benchmark_all_args_combinations_single_core(
    const InputBufferParams& params,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_,
    const std::string& res_path,
    const std::string& kernel_path) {
    // Create input and output replicated mesh buffers across generic mesh device; tests will only use first device
    const auto input_mesh_buffer = create_replicated_input_mesh_buffer_from_inputs(params, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto input_shard_view = input_mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto local_device = input_shard_view->device();

    tt::tt_metal::detail::SetDeviceProfilerDir(res_path + "/" + params.test_name);
    tt::tt_metal::detail::FreshProfilerDeviceLog();
    auto all_args_combinations = get_all_sharded_args_configs();
    for (const auto& arg_config : all_args_combinations) {
        auto args_bitmask = arg_config.raw();

        std::string crta_config_str = fmt::format("\"SHARDED_ACCESSOR_{:07b}\"", args_bitmask);
        log_info(
            tt::LogTest,
            "Creating single-core benchmarking program with the following args config: {}",
            crta_config_str);
        auto program = CreateProgram();

        constexpr CoreCoord grid = {0, 0};

        // Set up sharded accessor compile-time args for reader kernel
        const auto sharded_accessor_args = TensorAccessorArgs(*input_shard_view, arg_config);

        std::map<std::string, std::string> defines{{"ACCESSOR_CONFIG_NAME", crta_config_str}};
        // Create reader kernel
        KernelHandle reader_kernel_id = CreateKernel(
            program,
            kernel_path,
            grid,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = sharded_accessor_args.get_compile_time_args(),
                .defines = defines});

        // Set up runtime args for reader kernel
        SetCommonRuntimeArgs(program, reader_kernel_id, sharded_accessor_args.get_common_runtime_args());

        // Launch program
        auto mesh_work_load = tt::tt_metal::distributed::CreateMeshWorkload();
        AddProgramToMeshWorkload(
            mesh_work_load, std::move(program), (tt::tt_metal::distributed::MeshCoordinateRange)mesh_coordinate);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_work_load, false);

        // Wait for program to finish
        log_info(tt::LogTest, "Program launched!");
        Finish(mesh_device_->mesh_command_queue());
        log_info(tt::LogTest, "Program finished!");
    }
    tt::tt_metal::detail::DumpDeviceProfileResults(local_device);
}

TEST_P(AccessorBenchmarks, GetNocAddr) {
    benchmark_all_args_combinations_single_core(
        GetParam(),
        mesh_device_,
        "accessor_get_noc_addr_benchmarks",
        "tests/ttnn/unit_tests/gtests/accessor/kernels/accessor_get_noc_addr_page_id_benchmark.cpp");
}

TEST_P(AccessorBenchmarks, GetNocAddrPageCoord) {
    benchmark_all_args_combinations_single_core(
        GetParam(),
        mesh_device_,
        "accessor_get_noc_addr_page_coord_benchmarks",
        "tests/ttnn/unit_tests/gtests/accessor/kernels/accessor_get_noc_addr_page_coord_benchmark.cpp");
}

TEST_P(AccessorBenchmarks, Constructor) {
    benchmark_all_args_combinations_single_core(
        GetParam(),
        mesh_device_,
        "accessor_constructor_benchmarks",
        "tests/ttnn/unit_tests/gtests/accessor/kernels/accessor_constructor_benchmark.cpp");
}

INSTANTIATE_TEST_SUITE_P(
    AccessorTests,
    AccessorBenchmarks,
    ::testing::Values(
        // Sweep across shape ranks since TensorAccessor calculations scale with rank
        // TODO: Other interesting parameters to try:
        // - Page size: Possible that compiler optimizes division for power of 2 page sizes
        // - Shape dim: Possible that compiler optimizes division or modulo for certain values
        // - Try out other accessors as well, especially legacy ShardedAddrGen
        InputBufferParams{
            .test_name = "rank_2",
            .physical_tensor_shape = tt::tt_metal::Shape{320, 320},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        InputBufferParams{
            .test_name = "rank_3",
            .physical_tensor_shape = tt::tt_metal::Shape{5, 160, 320},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{3, 96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        InputBufferParams{
            .test_name = "rank_4",
            .physical_tensor_shape = tt::tt_metal::Shape{5, 5, 160, 160},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{3, 3, 96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        InputBufferParams{
            .test_name = "rank_5",
            .physical_tensor_shape = tt::tt_metal::Shape{5, 5, 5, 160, 160},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{3, 3, 3, 96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        InputBufferParams{
            .test_name = "rank_6",
            .physical_tensor_shape = tt::tt_metal::Shape{3, 3, 3, 3, 160, 160},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{2, 2, 2, 2, 96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        InputBufferParams{
            .test_name = "rank_7",
            .physical_tensor_shape = tt::tt_metal::Shape{3, 3, 3, 3, 3, 64, 64},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{2, 2, 2, 2, 2, 96, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        }));
