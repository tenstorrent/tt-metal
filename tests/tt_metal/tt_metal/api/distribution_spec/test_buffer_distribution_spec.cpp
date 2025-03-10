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
};

struct BufferDistributionSpecExpected {};

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
    auto physical_tensor_shape = params.inputs.physical_tensor_shape;
    auto physical_shard_shape = params.inputs.physical_shard_shape;
    const auto corerangeset = CoreRangeSet(CoreRange({0, 0}, {3, 3}));
    const auto shard_orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    const auto page_shape = tt::tt_metal::Shape2D{32, 32};
    const auto buffer_type = BufferType::L1;
    const size_t host_size = 8192;
    const size_t page_size = 2048;
    auto device = devices_[0];

    auto buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        physical_tensor_shape, physical_shard_shape, page_shape, corerangeset, shard_orientation);

    auto buffer = Buffer::create(device, host_size, page_size, buffer_type, buffer_distribution_spec);

    // These are the params allocator cares about
    auto aligned_size = buffer->aligned_size();
    auto aligned_page_size = buffer->aligned_page_size();
    auto allocator_buffer_type = buffer->buffer_type();
    auto bottom_up = buffer->bottom_up();
    auto num_cores = buffer->num_cores();
    std::cout << "size: " << aligned_size << std::endl;
    std::cout << "page size: " << aligned_page_size << std::endl;
    std::cout << "buffer is L1: " << (allocator_buffer_type == BufferType::L1) << std::endl;
    std::cout << "num_cores: " << num_cores.value() << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    BufferDistributionSpec,
    BufferDistributionSpecTests,
    ::testing::Values(BufferDistributionSpecParams{
        BufferDistributionSpecInputs{
            .physical_tensor_shape = tt::tt_metal::Shape{64, 64},
            .physical_shard_shape = tt::tt_metal::Shape{32, 32},
        },
        BufferDistributionSpecExpected{},
    })  // Values
);
