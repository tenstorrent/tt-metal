// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "dispatch/system_memory_manager.hpp"

#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/allocator.hpp>

namespace distribution_spec_tests {
using tt::tt_metal::BufferDistributionSpec;

struct BufferDistributionSpecInputs {
    tt::tt_metal::Shape physical_tensor_shape;
    tt::tt_metal::Shape physical_shard_shape;
    tt::tt_metal::Shape2D page_shape;
    float bytes_per_element;
    tt::tt_metal::CoreRangeSet grid;
    tt::tt_metal::ShardOrientation shard_orientation;
    tt::tt_metal::BufferType buffer_type;
};

struct MeshBufferAllocationExpected {
    std::vector<CoreCoord> cores;
    size_t num_cores;
    size_t num_dev_pages;
    size_t aligned_size;
    size_t aligned_size_per_bank;
};

struct MeshBufferReadWriteExpected {
    using ExplicitCorePageMapping = std::vector<std::vector<uint32_t>>;
    ExplicitCorePageMapping explicit_core_page_mapping;
};

struct BufferAllocationParams {
    BufferDistributionSpecInputs inputs;
    MeshBufferAllocationExpected expected;
};

struct BufferReadWriteParams {
    BufferDistributionSpecInputs inputs;
    MeshBufferReadWriteExpected expected;
};

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_replicated_mesh_buffer_from_inputs(
    const BufferDistributionSpecInputs& inputs, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    auto buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.physical_shard_shape,
        inputs.page_shape,
        inputs.grid,
        inputs.shard_orientation);

    // These values would be passed from tensor correctly based on PageConfig
    const auto host_size_in_bytes = inputs.physical_tensor_shape.volume() * inputs.bytes_per_element;
    const auto page_size = inputs.page_shape.height() * inputs.page_shape.width() * inputs.bytes_per_element;

    // MeshBuffer params
    const tt::tt_metal::distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = inputs.buffer_type,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .shard_parameters = buffer_distribution_spec,
    };

    // Mirrors allocate_mesh_buffer_on_device in ttnn
    const tt::tt_metal::distributed::ReplicatedBufferConfig mesh_buffer_config{.size = host_size_in_bytes};
    return tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, device_local_config, mesh_device);
}

}  // namespace distribution_spec_tests
// namespace

using namespace distribution_spec_tests;
using namespace tt::tt_metal;

class MeshBufferAllocationTests : public GenericMeshDeviceFixture,
                                  public ::testing::WithParamInterface<BufferAllocationParams> {};

TEST_P(MeshBufferAllocationTests, Allocation) {
    const auto& params = GetParam();

    // Create a replicated mesh buffer across generic mesh device; tests will only use first device
    const auto mesh_buffer = create_replicated_mesh_buffer_from_inputs(params.inputs, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto shard_view = mesh_buffer->get_device_buffer(mesh_coordinate);

    // Check that the stored cores in local device buffer matches expected cores to be used
    auto page_mapping = shard_view->buffer_distribution_spec()->compute_page_mapping();
    EXPECT_EQ(page_mapping.all_cores, params.expected.cores);

    /* These are the params allocator cares about; check all of them */
    EXPECT_EQ(shard_view->num_cores().value(), params.expected.num_cores);
    // For BufferDistributionSpec, defined as: max number of pages per core * num_cores
    EXPECT_EQ(shard_view->num_dev_pages(), params.expected.num_dev_pages);

    // Alignment is handled internally, not testing that here
    // In local device buffer, defined as: num_dev_pages * aligned_page_size
    EXPECT_EQ(shard_view->aligned_size(), params.expected.aligned_size);
    // In local device buffer, calculated from: aligned_size, aligned_page_size, num_banks, alignment
    // TODO: Fix local device buffer implementation to use aligned_size / num_cores? They should be equal...
    // - Need to make shard_view->num_cores() not optional...
    EXPECT_EQ(shard_view->aligned_size_per_bank(), params.expected.aligned_size_per_bank);
    EXPECT_EQ(shard_view->aligned_size() % shard_view->num_cores().value(), 0);
    EXPECT_EQ(shard_view->aligned_size_per_bank(), shard_view->aligned_size() / shard_view->num_cores().value());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BufferDistributionSpec,
    MeshBufferAllocationTests,
    ::testing::Values(
        // Cut along last two dims; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        BufferAllocationParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                .physical_shard_shape = tt::tt_metal::Shape{5, 2, 32, 32},
                .page_shape = tt::tt_metal::Shape2D{32, 32},
                .bytes_per_element = 2,
                .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                .shard_orientation = ShardOrientation::ROW_MAJOR,
                .buffer_type = BufferType::L1,
            },
            MeshBufferAllocationExpected{
                .cores = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}, {0, 2}, {1, 2}, {2, 2}, {3, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 3}},
                .num_cores = 16,
                .num_dev_pages = 10 * 16,  // Shard shape is 10 pages
                .aligned_size = 2048 * 160,
                .aligned_size_per_bank = 2048 * 10,
            },
        },
        // Cut along batch and width; row major layout
        // page size = 1 x 64 x 4 = 256 bytes (eg. float32, uint32, etc...)
        BufferAllocationParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 128},
                .physical_shard_shape = tt::tt_metal::Shape{3, 2, 64, 64},
                .page_shape = tt::tt_metal::Shape2D{1, 64},
                .bytes_per_element = 4,
                .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                .shard_orientation = ShardOrientation::COL_MAJOR,
                .buffer_type = BufferType::L1,
            },
            MeshBufferAllocationExpected{
                .cores = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {3, 0}, {3, 1}, {3, 2}, {3, 3}},
                .num_cores = 16,
                .num_dev_pages = 384 * 16,  // Shard shape is 384 pages
                .aligned_size = 256 * 6144,
                .aligned_size_per_bank = 256 * 384,
            },
        },
        // Multiple shards per bank; tile layout
        // page size = 32 x 32 x 1.0625 = 1088 bytes (eg. bfloat8_b)
        BufferAllocationParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                .physical_shard_shape = tt::tt_metal::Shape{2, 1, 64, 64},
                .page_shape = tt::tt_metal::Shape2D{32, 32},
                .bytes_per_element = 1.0625,  // Headers for block float amortized over elements
                .grid = CoreRangeSet(std::set<CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                .shard_orientation = ShardOrientation::ROW_MAJOR,
                .buffer_type = BufferType::L1,
            },
            MeshBufferAllocationExpected{
                .cores = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}},
                .num_cores = 5,
                .num_dev_pages = 3 * 8 * 5,  // Shard shape is 8 pages; 3 shards max per bank (12 shards over 5 banks)
                .aligned_size = 1088 * 120,
                .aligned_size_per_bank = 1088 * 24,
            },
        },
        // Generic ND sharding example; row major layout
        // page size = 1 x 128 x 1 = 128 bytes (eg. uint8, int8, etc...)
        BufferAllocationParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{2, 3, 4, 5, 6, 128},
                .physical_shard_shape = tt::tt_metal::Shape{3, 1, 5, 4, 2, 128},
                .page_shape = tt::tt_metal::Shape2D{1, 128},
                .bytes_per_element = 1,
                .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                .shard_orientation = ShardOrientation::COL_MAJOR,
                .buffer_type = BufferType::L1,
            },
            MeshBufferAllocationExpected{
                .cores = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                .num_cores = 4,
                .num_dev_pages = 5 * 120 *
                                 4,  // Shard shape is 120 pages; 5 shards max per bank (18 shards over 4 banks)
                .aligned_size = 128 * 2400,
                .aligned_size_per_bank = 128 * 600,
            },
        })  // Values
);
// clang-format on

class MeshBufferReadWriteTests : public GenericMeshDeviceFixture,
                                 public ::testing::WithParamInterface<std::tuple<bool, bool, BufferReadWriteParams>> {};

TEST_P(MeshBufferReadWriteTests, WriteReadLoopback) {
    const auto& [cq_write, cq_read, params] = GetParam();

    // Create a replicated mesh buffer across generic mesh device; tests will only use first device
    const auto mesh_buffer = create_replicated_mesh_buffer_from_inputs(params.inputs, mesh_device_.get());

    // Extract local single-device buffer (ie. shard_view) concepts for testing
    const tt::tt_metal::distributed::MeshCoordinate mesh_coordinate{0, 0};
    const auto shard_view = mesh_buffer->get_device_buffer(mesh_coordinate);
    const auto local_device = shard_view->device();
    const auto host_size_in_bytes = mesh_buffer->device_local_size();
    const auto bank_base_address = mesh_buffer->address();
    const auto page_size = mesh_buffer->page_size();
    const auto aligned_page_size = mesh_buffer->get_backing_buffer()->aligned_page_size();

    // Double check that mesh buffer properties match local single-device buffer properties
    ASSERT_EQ(host_size_in_bytes, shard_view->size());
    ASSERT_EQ(bank_base_address, shard_view->address());

    /* Test is based off of: test_EnqueueWriteBuffer_and_EnqueueReadBuffer
     * Here, buffer refers to local device buffer or shard view
     * - Initialize buffer and command queue state to 0
     * - Initialize src vector
     * - Write to buffer (with either EnqueueWriteBuffer or WriteToBuffer)
     * - Validate written results are correct per core (using explicitly hard-coded core mapping)
     * - Read from buffer (with either EnqueueReadBuffer or ReadFromBuffer)
     */

    // Initialize local device buffer to 0
    {
        std::vector<uint32_t> zeros_vector(shard_view->aligned_size_per_bank() / sizeof(uint32_t), 0);
        for (const auto& core : corerange_to_cores(params.inputs.grid)) {
            tt::tt_metal::detail::WriteToDeviceL1(
                local_device, core, bank_base_address, zeros_vector, shard_view->core_type());
        }
    }

    // Clear out command queue
    {
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(local_device->id());
        chip_id_t mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(local_device->id());
        uint32_t cq_size = local_device->sysmem_manager().get_cq_size();
        uint32_t cq_start = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
            CommandQueueHostAddrType::UNRESERVED);

        std::vector<uint32_t> cq_zeros((cq_size - cq_start) / sizeof(uint32_t), 0);

        tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
            cq_zeros.data(),
            (cq_size - cq_start),
            get_absolute_cq_offset(channel, 0, cq_size) + cq_start,
            mmio_device_id,
            channel);
    }

    // Create src vector
    const auto src =
        tt::test_utils::generate_uniform_random_vector<uint8_t>(0, UINT8_MAX, host_size_in_bytes / sizeof(uint8_t));

    if (cq_write) {
        log_info(tt::LogTest, "Writing with: FDMeshCommandQueue enqueue_write_shards");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(src.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, shard_data_transfer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());
    } else {
        log_info(tt::LogTest, "Writing with: WriteToBuffer (equivalent to SDMeshCommandQueue enqueue_write_shards)");
        tt::tt_metal::detail::WriteToBuffer(*shard_view, src);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(local_device->id());
    }

    // Validate written results are correct per core
    {
        // result_per_core is reassigned inside ReadFromDeviceL1
        std::vector<uint32_t> result_per_core;
        // src.data() is sufficient since src is already a vector of uint8_t, but this is safer if src creation is
        // changed to another dtype
        const auto* src_ptr = static_cast<const uint8_t*>(src.data());

        auto buffer_page_mapping = shard_view->buffer_distribution_spec()->compute_page_mapping();
        const auto& cores = buffer_page_mapping.all_cores;
        const auto& page_mapping = buffer_page_mapping.core_host_page_indices;

        const auto& expected_page_mapping = params.expected.explicit_core_page_mapping;
        EXPECT_TRUE(expected_page_mapping.size() <= page_mapping.size());
        for (size_t expected_core_idx = 0; expected_core_idx < expected_page_mapping.size(); expected_core_idx++) {
            EXPECT_EQ(page_mapping[expected_core_idx], expected_page_mapping[expected_core_idx]);
        }
        for (size_t empty_core_idx = expected_page_mapping.size(); empty_core_idx < page_mapping.size();
             empty_core_idx++) {
            EXPECT_EQ(page_mapping[empty_core_idx], std::vector<uint32_t>(page_mapping[empty_core_idx].size()));
        }

        for (size_t i = 0; i < cores.size(); i++) {
            tt::tt_metal::detail::ReadFromDeviceL1(
                local_device,
                cores[i],
                bank_base_address,
                shard_view->aligned_size_per_bank(),
                result_per_core,
                shard_view->core_type());

            const auto* result_per_core_ptr = reinterpret_cast<const uint8_t*>(result_per_core.data());
            for (size_t core_page = 0; core_page < page_mapping[i].size(); core_page++) {
                if (!page_mapping[i][core_page]) {
                    continue;
                }
                const auto host_page = page_mapping[i][core_page];
                EXPECT_EQ(
                    std::memcmp(
                        src_ptr + host_page * page_size,
                        result_per_core_ptr + core_page * aligned_page_size,
                        page_size),
                    0);
            }
        }
    }

    // Initialize dst vector
    std::vector<uint8_t> dst(host_size_in_bytes / sizeof(uint8_t), 0);

    if (cq_read) {
        log_info(tt::LogTest, "Reading with: FDMeshCommandQueue enqueue_read_shards");
        std::vector<tt::tt_metal::distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfer{{
            .shard_coord = tt::tt_metal::distributed::MeshCoordinate{0, 0},
            .host_data = const_cast<void*>(reinterpret_cast<const void*>(dst.data())),
        }};
        mesh_device_->mesh_command_queue().enqueue_read_shards(shard_data_transfer, mesh_buffer, /*blocking=*/false);
        Finish(mesh_device_->mesh_command_queue());
    } else {
        log_info(tt::LogTest, "Reading with: ReadFromBuffer (equivalent to SDMeshCommandQueue enqueue_read_shards)");
        tt::tt_metal::detail::ReadFromBuffer(*shard_view, dst);
    }

    // Validate read results are correct
    EXPECT_EQ(src, dst);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    BufferDistributionSpec,
    MeshBufferReadWriteTests,
    ::testing::Combine(
        ::testing::Values(true, false),  // cq_write
        ::testing::Values(true, false),  // cq_read
        ::testing::Values(
            // BLOCK sharding; tile layout
            // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
            BufferReadWriteParams{
                BufferDistributionSpecInputs{
                    .physical_tensor_shape = tt::tt_metal::Shape{2, 64, 96},
                    .physical_shard_shape = tt::tt_metal::Shape{1, 32, 64},
                    .page_shape = tt::tt_metal::Shape2D{32, 32},
                    .bytes_per_element = 2,
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
                MeshBufferReadWriteExpected{
                    .explicit_core_page_mapping = {
                        {0, 1},
                        {2, 0},
                        {3, 4},
                        {5, 0},
                        {6, 7},
                        {8, 0},
                        {9, 10},
                        {11, 0},
                    },
                },
            },
            // HEIGHT sharding with padding along shard width + random CoreRangeSet; tile layout
            // page size = 32 x 32 x 1.0625 = 1088 bytes (eg. bfloat8_b)
            BufferReadWriteParams{
                BufferDistributionSpecInputs{
                    .physical_tensor_shape = tt::tt_metal::Shape{2, 128, 64},
                    .physical_shard_shape = tt::tt_metal::Shape{1, 64, 96},
                    .page_shape = tt::tt_metal::Shape2D{32, 32},
                    .bytes_per_element = 1.0625,  // Headers for block float amortized over elements
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
                MeshBufferReadWriteExpected{
                    .explicit_core_page_mapping = {
                        {0, 1, 0, 2, 3, 0},
                        {4, 5, 0, 6, 7, 0},
                        {8, 9, 0, 10, 11, 0},
                        {12, 13, 0, 14, 15, 0},
                    },
                },
            },
            // WIDTH sharding with padding along shard height; row major layout with aligned page size
            // page size = 1 x 16 x 1 = 16 bytes (eg. uint8, int8, etc...)
            BufferReadWriteParams{
                BufferDistributionSpecInputs{
                    .physical_tensor_shape = tt::tt_metal::Shape{2, 3, 32},
                    .physical_shard_shape = tt::tt_metal::Shape{2, 4, 16},
                    .page_shape = tt::tt_metal::Shape2D{1, 16},
                    .bytes_per_element = 1,
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
                MeshBufferReadWriteExpected{
                    .explicit_core_page_mapping = {
                        {0, 2, 4, 0, 6, 8, 10, 0},
                        {1, 3, 5, 0, 7, 9, 11, 0},
                    },
                },
            },
            // ND sharding with multiple shards per bank; row major layout with non-aligned page size
            // Coaslescing possible based on shard spec but must be noncoalesced due to non-aligned pages
            // page size = 1 x 4 x 1 = 4 bytes (eg. uint8, int8, etc...)
            BufferReadWriteParams{
                BufferDistributionSpecInputs{
                    .physical_tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
                    .physical_shard_shape = tt::tt_metal::Shape{1, 1, 2, 2, 4},
                    .page_shape = tt::tt_metal::Shape2D{1, 4},
                    .bytes_per_element = 1,
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                    .shard_orientation = ShardOrientation::COL_MAJOR,
                    .buffer_type = BufferType::L1,
                },
                MeshBufferReadWriteExpected{
                    .explicit_core_page_mapping = {
                        {0, 1, 3, 4, 30, 31, 33, 34},
                        {2, 0, 5, 0, 32, 0, 35, 0},
                        {6, 7, 9, 10, 0, 0, 0, 0},
                        {8, 0, 11, 0, 0, 0, 0, 0},
                        {12, 13, 15, 16, 0, 0, 0, 0},
                        {14, 0, 17, 0, 0, 0, 0, 0},
                        {18, 19, 21, 22, 0, 0, 0, 0},
                        {20, 0, 23, 0, 0, 0, 0, 0},
                        {24, 25, 27, 28, 0, 0, 0, 0},
                        {26, 0, 29, 0, 0, 0, 0, 0}
                    },
                },
            },
            // ND sharding with multiple shards per bank; tile layout
            // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
            BufferReadWriteParams{
                BufferDistributionSpecInputs{
                    .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                    .physical_shard_shape = tt::tt_metal::Shape{2, 1, 64, 64},
                    .page_shape = tt::tt_metal::Shape2D{32, 32},
                    .bytes_per_element = 2,
                    .grid = CoreRangeSet(
                        tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                    .shard_orientation = ShardOrientation::ROW_MAJOR,
                    .buffer_type = BufferType::L1,
                },
                MeshBufferReadWriteExpected{
                    .explicit_core_page_mapping = {
                        {0, 1, 3, 4, 12, 13, 15, 16, 26, 0, 29, 0, 38, 0, 41, 0, 54, 55, 57, 58, 0, 0, 0, 0},
                        {2, 0, 5, 0, 14, 0, 17, 0, 30, 31, 33, 34, 42, 43, 45, 46, 56, 0, 59, 0, 0, 0, 0, 0},
                        {6, 7, 9, 10, 18, 19, 21, 22, 32, 0, 35, 0, 44, 0, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {8, 0, 11, 0, 20, 0, 23, 0, 48, 49, 51, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {24, 25, 27, 28, 36, 37, 39, 40, 50, 0, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
                    },
                },
            })  // Values
        )       // Combine
);
// clang-format on
