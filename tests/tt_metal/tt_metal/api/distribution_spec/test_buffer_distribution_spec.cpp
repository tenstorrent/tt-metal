// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "command_queue_fixture.hpp"

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

struct BufferDistributionSpecParams {
    BufferDistributionSpecInputs inputs;
    BufferDistributionSpecExpected expected;
};
}  // namespace distribution_spec_tests
// namespace

using namespace distribution_spec_tests;
using namespace tt::tt_metal;

class BufferDistributionSpecTests : public CommandQueueSingleCardBufferFixture,
                                    public ::testing::WithParamInterface<BufferDistributionSpecParams> {};

TEST_P(BufferDistributionSpecTests, BufferCreation) {
    const auto& params = GetParam();
    auto device = devices_[0];

    auto physical_tensor_shape = params.inputs.physical_tensor_shape;
    auto physical_shard_shape = params.inputs.physical_shard_shape;
    auto page_shape = params.inputs.page_shape;
    auto bytes_per_element = params.inputs.bytes_per_element;
    auto corerangeset = params.inputs.grid;
    auto shard_orientation = params.inputs.shard_orientation;

    // Doesn't matter for this test
    const auto buffer_type = BufferType::L1;

    auto buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        physical_tensor_shape, physical_shard_shape, page_shape, corerangeset, shard_orientation);

    // Check that the stored cores in BufferDistributionSpec is exact cores used
    EXPECT_EQ(buffer_distribution_spec.get_cores(), params.expected.cores);

    // These values would be passed from tensor correctly based on PageConfig
    const auto host_size = physical_tensor_shape.volume() * bytes_per_element;
    const auto page_size = page_shape.height() * page_shape.width() * bytes_per_element;
    auto buffer = Buffer::create(device, host_size, page_size, buffer_type, buffer_distribution_spec);

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
