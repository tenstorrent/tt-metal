// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include "ttnn/cpp/ttnn/operations/sharding_utilities.hpp"

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
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_parameters = input_buffer_distribution_spec,
    };
    const auto input_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, input_device_local_config, mesh_device);

    return input_mesh_buffer;
}

}  // namespace accessor_benchmarks

using namespace accessor_benchmarks;
using namespace tt::tt_metal;

class AccessorBenchmarks : public GenericMeshDeviceFixture, public ::testing::WithParamInterface<InputBufferParams> {};

TEST_P(AccessorBenchmarks, Generic) {
    const auto& params = GetParam();

    // Create input and output replicated mesh buffers across generic mesh device; tests will only use first device
    const auto input_mesh_buffer = create_replicated_input_mesh_buffer_from_inputs(params, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto input_shard_view = input_mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto local_device = input_shard_view->device();

    const auto input_bank_base_address = input_mesh_buffer->address();

    /* CREATE AND LAUNCH PROGRAM ON DEVICE
     * - This program uses a single reader kernel to measure get_noc_addr overhead with different accessors
     * - It does not actually do any reads, so no CBs are needed
     * - Input buffers are set up with actual ND ShardSpec, but for interleaved we use same buffer
     *   -  ie. Calculations are definitely wrong/nonsense for interleaved, but it's ok to use for benchmarking
     *   - TODO: We can properly setup the input buffers and try reading from them as well
     */
    tt::tt_metal::detail::SetDeviceProfilerDir("accessor_benchmarks/" + params.test_name);
    tt::tt_metal::detail::FreshProfilerDeviceLog();
    {
        log_info(tt::LogTest, "Creating single-core benchmarking program");
        auto program = CreateProgram();

        constexpr CoreCoord grid = {0, 0};
        const auto data_format = params.data_format;
        const auto aligned_page_size = input_shard_view->aligned_page_size();

        // Set up sharded accessor compile-time args for reader kernel
        const auto& input_buffer_distribution_spec =
            std::get<BufferDistributionSpec>(input_mesh_buffer->device_local_config().shard_parameters.value());
        const auto input_sharded_accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
            *mesh_device_, input_buffer_distribution_spec, input_shard_view->core_type());
        std::vector<uint32_t> input_compile_time_args = {
            (uint32_t)data_format,
            aligned_page_size,
            input_sharded_accessor_args.rank,
            input_sharded_accessor_args.num_banks};
        input_compile_time_args.insert(
            input_compile_time_args.end(),
            input_sharded_accessor_args.shapes_and_bank_coords.cbegin(),
            input_sharded_accessor_args.shapes_and_bank_coords.cend());

        // Create reader kernel
        KernelHandle reader_kernel_id = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/accessor/kernels/accessor_benchmark.cpp",
            grid,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = input_compile_time_args});

        // Set up runtime args for reader kernel
        std::vector<uint32_t> input_runtime_args = {
            input_bank_base_address,
        };
        SetRuntimeArgs(program, reader_kernel_id, grid, input_runtime_args);

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

INSTANTIATE_TEST_SUITE_P(
    AccessorTests,
    AccessorBenchmarks,
    ::testing::Values(
        // Sweep across shape ranks since ShardedAccessor calculations scale with rank
        // TODO: Other interesting parameters to try:
        // - Page size: Possible that compiler optimizes division for power of 2 page sizes
        // - Shape dim: Possible that compiler optimizes division or modulo for certain values
        // - Try out other accessors as well, especially legacy ShardedAddrGen
        InputBufferParams{
            .test_name = "rank_2",
            .physical_tensor_shape = tt::tt_metal::Shape{160, 160},
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
            .physical_tensor_shape = tt::tt_metal::Shape{5, 160, 160},
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
