// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "command_queue_fixture.hpp"

#include "tt_metal/test_utils/stimulus.hpp"
// #include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

namespace distribution_spec_tests {
using tt::tt_metal::BufferDistributionSpec;

struct BufferDistributionSpecInputs {
    tt::tt_metal::Shape physical_tensor_shape;
    tt::tt_metal::Shape physical_shard_shape;
    tt::tt_metal::Shape2D page_shape;
    float bytes_per_element;
    tt::tt_metal::CoreRangeSet grid;
    tt::tt_metal::ShardOrientation shard_orientation;
};

struct BufferDistributionSpecExpected {
    std::vector<CoreCoord> cores;
    size_t num_cores;
    size_t num_dev_pages;
    size_t aligned_size;
    size_t aligned_size_per_bank;
};

struct EnqueueReadWriteBufferExpected {
    std::vector<CoreCoord> cores;
};

struct BufferDistributionSpecParams {
    BufferDistributionSpecInputs inputs;
    BufferDistributionSpecExpected expected;
};

struct EnqueueReadWriteBufferParams {
    BufferDistributionSpecInputs inputs;
    EnqueueReadWriteBufferExpected expected;
};

std::shared_ptr<tt::tt_metal::Buffer> create_buffer_from_inputs(
    const BufferDistributionSpecInputs& inputs,
    const tt::tt_metal::BufferType& buffer_type,
    tt::tt_metal::IDevice* device) {
    auto buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        inputs.physical_tensor_shape,
        inputs.physical_shard_shape,
        inputs.page_shape,
        inputs.grid,
        inputs.shard_orientation);

    // These values would be passed from tensor correctly based on PageConfig
    const auto host_size = inputs.physical_tensor_shape.volume() * inputs.bytes_per_element;
    const auto page_size = inputs.page_shape.height() * inputs.page_shape.width() * inputs.bytes_per_element;
    return tt::tt_metal::Buffer::create(device, host_size, page_size, buffer_type, buffer_distribution_spec);
}

}  // namespace distribution_spec_tests
// namespace

using namespace distribution_spec_tests;
using namespace tt::tt_metal;

class BufferDistributionSpecTests : public CommandQueueSingleCardBufferFixture,
                                    public ::testing::WithParamInterface<BufferDistributionSpecParams> {};

TEST_P(BufferDistributionSpecTests, BufferCreation) {
    const auto& params = GetParam();

    auto buffer = create_buffer_from_inputs(params.inputs, BufferType::L1, devices_[0]);
    const auto& buffer_distribution_spec = buffer->get_modifiable_buffer_distribution_spec()->get();

    // Check that the stored cores in BufferDistributionSpec is exact cores used
    EXPECT_EQ(buffer_distribution_spec.get_cores(), params.expected.cores);

    /* These are the params allocator cares about; check all of them */
    EXPECT_EQ(buffer->num_cores().value(), params.expected.num_cores);
    // For BufferDistributionSpec, defined as: max number of pages per core * num_cores
    EXPECT_EQ(buffer->num_dev_pages(), params.expected.num_dev_pages);

    // Alignment is handled internally, not testing that here
    // In buffer, defined as: num_dev_pages * aligned_page_size
    EXPECT_EQ(buffer->aligned_size(), params.expected.aligned_size);
    // In buffer, calculated from: aligned_size, aligned_page_size, num_banks, alignment
    // TODO: Fix buffer implementation to use aligned_size / num_cores? They should be equal...
    // - Need to make buffer->num_cores() not optional...
    EXPECT_EQ(buffer->aligned_size_per_bank(), params.expected.aligned_size_per_bank);
    EXPECT_EQ(buffer->aligned_size() % buffer->num_cores().value(), 0);
    EXPECT_EQ(buffer->aligned_size_per_bank(), buffer->aligned_size() / buffer->num_cores().value());
}

INSTANTIATE_TEST_SUITE_P(
    BufferDistributionSpec,
    BufferDistributionSpecTests,
    ::testing::Values(
        // Cut along last two dims; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        BufferDistributionSpecParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                .physical_shard_shape = tt::tt_metal::Shape{5, 2, 32, 32},
                .page_shape = tt::tt_metal::Shape2D{32, 32},
                .bytes_per_element = 2,
                .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                .shard_orientation = ShardOrientation::ROW_MAJOR,
            },
            BufferDistributionSpecExpected{
                .cores = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}},
                .num_cores = 6,
                .num_dev_pages = 10 * 6,  // Shard shape is 10 pages
                .aligned_size = 2048 * 60,
                .aligned_size_per_bank = 2048 * 10,
            },
        },
        // Cut along batch and width; row major layout
        // page size = 1 x 64 x 4 = 256 bytes (eg. float32, uint32, etc...)
        BufferDistributionSpecParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 128},
                .physical_shard_shape = tt::tt_metal::Shape{3, 2, 64, 64},
                .page_shape = tt::tt_metal::Shape2D{1, 64},
                .bytes_per_element = 4,
                .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                .shard_orientation = ShardOrientation::COL_MAJOR,
            },
            BufferDistributionSpecExpected{
                .cores = {{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                .num_cores = 4,
                .num_dev_pages = 384 * 4,  // Shard shape is 384 pages
                .aligned_size = 256 * 1536,
                .aligned_size_per_bank = 256 * 384,
            },
        },
        // Multiple shards per bank; tile layout
        // page size = 32 x 32 x 1.0625 = 1088 bytes (eg. bfloat8_b)
        BufferDistributionSpecParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                .physical_shard_shape = tt::tt_metal::Shape{2, 1, 64, 64},
                .page_shape = tt::tt_metal::Shape2D{32, 32},
                .bytes_per_element = 1.0625,  // Headers for block float amortized over elements
                .grid = CoreRangeSet(std::set<CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                .shard_orientation = ShardOrientation::ROW_MAJOR,
            },
            BufferDistributionSpecExpected{
                .cores = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}},
                .num_cores = 5,
                .num_dev_pages = 3 * 8 * 5,  // Shard shape is 8 pages; 3 shards max per bank (12 shards over 5 banks)
                .aligned_size = 1088 * 120,
                .aligned_size_per_bank = 1088 * 24,
            },
        },
        // Generic ND sharding example; row major layout
        // page size = 1 x 128 x 1 = 128 bytes (eg. uint8, int8, etc...)
        BufferDistributionSpecParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{2, 3, 4, 5, 6, 128},
                .physical_shard_shape = tt::tt_metal::Shape{3, 1, 5, 4, 2, 128},
                .page_shape = tt::tt_metal::Shape2D{1, 128},
                .bytes_per_element = 1,
                .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                .shard_orientation = ShardOrientation::COL_MAJOR,
            },
            BufferDistributionSpecExpected{
                .cores = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                .num_cores = 4,
                .num_dev_pages = 5 * 120 *
                                 4,  // Shard shape is 120 pages; 5 shards max per bank (18 shards over 4 banks)
                .aligned_size = 128 * 2400,
                .aligned_size_per_bank = 128 * 600,
            },
        })  // Values
);

class EnqueueReadWriteBufferTests : public CommandQueueSingleCardBufferFixture,
                                    public ::testing::WithParamInterface<EnqueueReadWriteBufferParams> {};

TEST_P(EnqueueReadWriteBufferTests, EnqueueWriteBuffer) {
    const auto& params = GetParam();

    auto buffer = create_buffer_from_inputs(params.inputs, BufferType::L1, devices_[0]);
    auto device = buffer->device();

    // EnqueueWriteBuffer test
    // Reference: test_EnqueueWriteBuffer_and_EnqueueReadBuffer
    const auto src =
        tt::test_utils::generate_uniform_random_vector<uint8_t>(0, UINT8_MAX, buffer->size() / sizeof(uint8_t));
    auto& command_queue = device->command_queue();
    EnqueueWriteBuffer(command_queue, buffer, src.data(), /*blocking=*/false);
    Finish(command_queue);

    // Validate
    {
        std::vector<uint32_t> result_per_core;
        const auto* src_ptr = static_cast<const uint8_t*>(src.data());
        const auto& cores = params.expected.cores;
        const DeviceAddr base_address = buffer->address();
        // Guaranteed to have valid buffer distribution spec
        const auto& page_mapping = buffer->get_modifiable_buffer_distribution_spec()->get().get_page_mapping();
        for (size_t i = 0; i < cores.size(); i++) {
            const auto core = cores[i];
            std::cout << "------ Testing core: " << core.x << ", " << core.y << std::endl;

            // result_per_core is resized inside ReadFromDeviceL1
            tt::tt_metal::detail::ReadFromDeviceL1(
                device, core, base_address, buffer->aligned_size_per_bank(), result_per_core, buffer->core_type());
            tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());

            const auto* result_per_core_ptr = reinterpret_cast<const uint8_t*>(result_per_core.data());
            const auto& page_mapping_core = page_mapping[i];
            // EXPECT_EQ(src, result_per_core);
            for (auto& chunk_mapping : page_mapping_core) {
                const auto src_offset = chunk_mapping.src * buffer->page_size();
                const auto dst_offset = chunk_mapping.dst * buffer->aligned_page_size();

                // for (size_t j = 0; j < chunk_mapping.size * buffer->page_size(); j++) {
                //     if (src_ptr[src_offset + j] != result_per_core_ptr[dst_offset + j]) {
                //         std::cout << "Mismatch at index: " << j << std::endl;
                //         std::cout << "src_val: " << static_cast<int>(src_ptr[src_offset + j]) << std::endl;
                //         std::cout << "dst_val: " << static_cast<int>(result_per_core_ptr[dst_offset + j]) <<
                //         std::endl; break;
                //     }
                // }
                EXPECT_EQ(
                    std::memcmp(
                        src_ptr + src_offset,
                        result_per_core_ptr + dst_offset,
                        chunk_mapping.size * buffer->page_size()),
                    0);
            }
        }
    }

    // dram_channel_from_logical_core

    // ReadFromDeviceDRAMChannel
    // ReadFromDeviceL1
    // generate_uniform_random_vector
}

INSTANTIATE_TEST_SUITE_P(
    BufferDistributionSpec,
    EnqueueReadWriteBufferTests,
    ::testing::Values(
        // Cut along last two dims; tile layout
        // page size = 32 x 32 x 2 = 2048 bytes (eg. bfloat16, uint16, etc...)
        EnqueueReadWriteBufferParams{
            BufferDistributionSpecInputs{
                .physical_tensor_shape = tt::tt_metal::Shape{5, 2, 64, 96},
                .physical_shard_shape = tt::tt_metal::Shape{5, 2, 32, 32},
                .page_shape = tt::tt_metal::Shape2D{32, 32},
                .bytes_per_element = 2,
                .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                .shard_orientation = ShardOrientation::ROW_MAJOR,
            },
            EnqueueReadWriteBufferExpected{
                .cores = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}},
            },
        })  // Values
);
