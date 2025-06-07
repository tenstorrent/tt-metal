// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fmt/format.h>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

#include "ttnn/cpp/ttnn/operations/sharding_utilities.hpp"

namespace sharded_accessor_device_tests {

struct InputOutputBufferParams {
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
    DistributionSpecParams output_shard_spec;
};

std::array<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>, 2>
create_replicated_input_and_output_mesh_buffers_from_inputs(
    const InputOutputBufferParams& inputs, tt::tt_metal::distributed::MeshDevice* mesh_device) {
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

    // Create output mesh buffer
    auto output_buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.input_shard_spec.physical_shard_shape,
        inputs.page_shape,
        inputs.output_shard_spec.grid,
        inputs.output_shard_spec.shard_orientation);
    const tt::tt_metal::distributed::DeviceLocalBufferConfig output_device_local_config{
        .page_size = page_size,
        .buffer_type = inputs.output_shard_spec.buffer_type,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_parameters = output_buffer_distribution_spec,
    };
    const auto output_mesh_buffer =
        tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, output_device_local_config, mesh_device);

    return {input_mesh_buffer, output_mesh_buffer};
}

}  // namespace sharded_accessor_device_tests

using namespace sharded_accessor_device_tests;
using namespace tt::tt_metal;

class ShardedAccessorTestsOnDevice : public GenericMeshDeviceFixture,
                                     public ::testing::WithParamInterface<InputOutputBufferParams> {};

TEST_P(ShardedAccessorTestsOnDevice, SingleCoreReshard) {
    const auto& params = GetParam();

    // Create input and output replicated mesh buffers across generic mesh device; tests will only use first device
    const auto [input_mesh_buffer, output_mesh_buffer] =
        create_replicated_input_and_output_mesh_buffers_from_inputs(params, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto input_shard_view = input_mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto output_shard_view = output_mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto local_device = input_shard_view->device();

    const auto host_size_in_bytes = input_mesh_buffer->device_local_size();
    ASSERT_EQ(host_size_in_bytes, output_mesh_buffer->device_local_size());

    const auto input_bank_base_address = input_mesh_buffer->address();
    const auto output_bank_base_address = output_mesh_buffer->address();
    ASSERT_NE(input_bank_base_address, output_bank_base_address);

    // Input and output buffers may not have the same aligned size per bank
    // Initialize input local device buffers to 0
    {
        std::vector<uint32_t> zeros_vector(input_shard_view->aligned_size_per_bank() / sizeof(uint32_t), 0);
        for (const auto& core : corerange_to_cores(params.input_shard_spec.grid)) {
            tt::tt_metal::detail::WriteToDeviceL1(
                local_device, core, input_bank_base_address, zeros_vector, input_shard_view->core_type());
        }
    }

    // Initialize output local device buffers to 0
    {
        std::vector<uint32_t> zeros_vector(output_shard_view->aligned_size_per_bank() / sizeof(uint32_t), 0);
        for (const auto& core : corerange_to_cores(params.output_shard_spec.grid)) {
            tt::tt_metal::detail::WriteToDeviceL1(
                local_device, core, output_bank_base_address, zeros_vector, output_shard_view->core_type());
        }
    }

    // Create src vector
    const auto src =
        tt::test_utils::generate_uniform_random_vector<uint8_t>(0, UINT8_MAX, host_size_in_bytes / sizeof(uint8_t));

    {
        log_info(tt::LogTest, "Writing input buffer to device");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(src.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_write_shards(
            input_mesh_buffer, shard_data_transfer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());
    }

    /* CREATE AND LAUNCH PROGRAM ON DEVICE
     * - This program uses reader and writer kernel to copy input buffer to output buffer using sharded accessors.
     * - Inside the reader and writer kernels, loop through total volume (in pages) of the tensor to complete the copy.
     * - This is essentially a single-core reshard OP.
     * - TODO: One major restriction is that page size must be the same for both input and output buffers.
     *   - For tile layout, can use UNPACK / PACK to convert between different data formats and page sizes.
     *   - For row major layout, need to handle shard shapes with different widths (ie. last dim) properly
     */
    {
        log_info(tt::LogTest, "Creating single-core reshard program");
        auto program = CreateProgram();

        constexpr CoreCoord grid = {0, 0};
        const auto data_format = params.data_format;

        // Setup circular buffer for reading and writing to buffers
        // TODO: Expose aligned page size to mesh buffer?
        TT_FATAL(
            input_shard_view->aligned_page_size() == output_shard_view->aligned_page_size(),
            "Input and output mesh buffers must have the same aligned page size!");
        const auto aligned_page_size = input_shard_view->aligned_page_size();
        constexpr auto num_tiles = 2;  // Double buffered for perf, but it doesn't really matter for this test
        CBHandle cb_in0_idx = tt::CBIndex::c_0;
        auto c_in0_config = CircularBufferConfig(aligned_page_size * num_tiles, {{cb_in0_idx, data_format}})
                                .set_page_size(cb_in0_idx, aligned_page_size);
        auto cb_in0_id = CreateCircularBuffer(program, grid, c_in0_config);

        // Set up compile-time args for reader kernel
        const auto& input_buffer_distribution_spec =
            std::get<BufferDistributionSpec>(input_mesh_buffer->device_local_config().shard_parameters.value());
        const auto input_sharded_accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
            *mesh_device_, input_buffer_distribution_spec, input_shard_view->core_type());
        std::vector<uint32_t> input_compile_time_args = {
            input_sharded_accessor_args.rank, input_sharded_accessor_args.num_banks};
        input_compile_time_args.insert(
            input_compile_time_args.end(),
            input_sharded_accessor_args.shapes_and_bank_coords.cbegin(),
            input_sharded_accessor_args.shapes_and_bank_coords.cend());
        input_compile_time_args.push_back(cb_in0_idx);
        input_compile_time_args.push_back(aligned_page_size);

        // Set up compile-time args for writer kernel
        const auto& output_buffer_distribution_spec =
            std::get<BufferDistributionSpec>(output_mesh_buffer->device_local_config().shard_parameters.value());
        const auto output_sharded_accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
            *mesh_device_, output_buffer_distribution_spec, output_shard_view->core_type());
        std::vector<uint32_t> output_compile_time_args = {
            output_sharded_accessor_args.rank, output_sharded_accessor_args.num_banks};
        output_compile_time_args.insert(
            output_compile_time_args.end(),
            output_sharded_accessor_args.shapes_and_bank_coords.cbegin(),
            output_sharded_accessor_args.shapes_and_bank_coords.cend());
        output_compile_time_args.push_back(cb_in0_idx);
        output_compile_time_args.push_back(aligned_page_size);

        // Create reader kernel
        KernelHandle reader_kernel_id = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/accessor/kernels/reader_reshard.cpp",
            grid,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = input_compile_time_args});

        // Create writer kernel
        KernelHandle writer_kernel_id = CreateKernel(
            program,
            "tests/ttnn/unit_tests/gtests/accessor/kernels/writer_reshard.cpp",
            grid,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = output_compile_time_args});

        // Set up runtime args for reader kernel
        std::vector<uint32_t> input_runtime_args = {
            input_bank_base_address,
        };
        SetRuntimeArgs(program, reader_kernel_id, grid, input_runtime_args);

        // Set up runtime args for writer kernel
        std::vector<uint32_t> output_runtime_args = {
            output_bank_base_address,
        };
        SetRuntimeArgs(program, writer_kernel_id, grid, output_runtime_args);

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

    // Initialize dst vector
    std::vector<uint8_t> dst(host_size_in_bytes / sizeof(uint8_t), 0);

    // Validate output buffer matches src vector
    {
        log_info(tt::LogTest, "Validating output buffer matches src vector");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(dst.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_read_shards(
            shard_data_transfer, output_mesh_buffer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());

        // Validate read results are correct
        EXPECT_EQ(src, dst);
    }

    // Validate input buffer matches src vector (ie. unmodified after kernel read/writes)
    {
        log_info(tt::LogTest, "Validating input buffer matches src vector (as a sanity check)");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(dst.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_read_shards(
            shard_data_transfer, input_mesh_buffer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());

        // Validate read results are correct
        EXPECT_EQ(src, dst);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAccessorTests,
    ShardedAccessorTestsOnDevice,
    // Test cases are similar to MeshBufferReadWriteTests in test_buffer_distribution_spec.cpp
    // - Output distribution spec is something different from input distribution spec
    ::testing::Values(
        // BLOCK sharding; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        InputOutputBufferParams{
            .physical_tensor_shape = tt::tt_metal::Shape{2, 64, 96},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::Float16,

            .input_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        // HEIGHT sharding with padding along shard width + random CoreRangeSet; tile layout
        // page size = 32 x 32 x 1.0625 = 1088 bytes (eg. bfloat8_b)
        InputOutputBufferParams{
            .physical_tensor_shape = tt::tt_metal::Shape{2, 128, 64},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 1.0625,  // Headers for block float amortized over elements
            .data_format = tt::DataFormat::Bfp8,

            .input_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 64, 96},
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {5, 5})),
                    .shard_orientation = ShardOrientation::COL_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        // WIDTH sharding with padding along shard height; row major layout with aligned page size
        // page size = 1 x 16 x 1 = 16 bytes (eg. uint8, int8, etc...)
        InputOutputBufferParams{
            .physical_tensor_shape = tt::tt_metal::Shape{2, 3, 32},
            .page_shape = tt::tt_metal::Shape2D{1, 16},
            .bytes_per_element = 1,
            .data_format = tt::DataFormat::UInt8,

            .input_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 3, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {2, 2})),
                    .shard_orientation = ShardOrientation::COL_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        // ND sharding with multiple shards per bank; row major layout with non-aligned page size
        // Coaslescing possible based on shard spec but must be noncoalesced due to non-aligned pages
        // page size = 1 x 4 x 1 = 4 bytes (eg. uint8, int8, etc...)
        InputOutputBufferParams{
            .physical_tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
            .page_shape = tt::tt_metal::Shape2D{1, 4},
            .bytes_per_element = 1,
            .data_format = tt::DataFormat::UInt8,

            .input_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 1, 2, 2, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                    .shard_orientation = ShardOrientation::COL_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{3, 3, 1, 1, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        },
        // ND sharding with multiple shards per bank; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        InputOutputBufferParams{
            .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 2, 64, 96},
            .page_shape = tt::tt_metal::Shape2D{32, 32},
            .bytes_per_element = 2,
            .data_format = tt::DataFormat::UInt16,

            .input_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{1, 1, 2, 64, 64},
                    .grid = CoreRangeSet(
                        tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                    .shard_orientation = ShardOrientation::COL_MAJOR,
                    .buffer_type = BufferType::L1,
                },
            .output_shard_spec =
                InputOutputBufferParams::DistributionSpecParams{
                    .physical_shard_shape = tt::tt_metal::Shape{5, 1, 1, 96, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
        }));
