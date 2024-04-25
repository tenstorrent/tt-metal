// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tests/ttnn/unit_tests/ttnn_test_fixtures.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn {
namespace operations {
namespace core {
namespace test {

struct ToMemoryConfigParam {
    uint32_t h;
    uint32_t w;
};

class NoChangeToMemoryConfigFixture : public TTNNFixture, public testing::WithParamInterface<ToMemoryConfigParam> {};

TEST_P(NoChangeToMemoryConfigFixture, NoChangeToMemoryConfig) {
    auto param = GetParam();
    const auto device_id = 0;
    auto &device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    const auto input_tensor = ttnn::zeros(shape, ttnn::bfloat16, ttnn::ROW_MAJOR_LAYOUT, device);
    const auto expected_tensor = ttnn::zeros(shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, device);
    const auto l1_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::L1};
    const auto dram_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = tt::tt_metal::BufferType::DRAM};
    const auto output_tensor = ttnn::to_memory_config(input_tensor, dram_memory_config);
    TT_FATAL(tt::numpy::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)));
    ttnn::close_device(device);
}

INSTANTIATE_TEST_SUITE_P(
    ToMemoryConfigTests, NoChangeToMemoryConfigFixture, ::testing::Values(ToMemoryConfigParam{32, 64}));

class ChangeLayoutWithToMemoryConfigFixture : public TTNNFixture,
                                              public testing::WithParamInterface<ToMemoryConfigParam> {};

TEST_P(ChangeLayoutWithToMemoryConfigFixture, ChangeLayoutWithToMemoryConfig) {
    auto param = GetParam();
    const auto device_id = 0;
    auto &device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    const auto input_tensor = ttnn::zeros(shape, ttnn::bfloat16, ttnn::ROW_MAJOR_LAYOUT, device);
    const auto expected_tensor = ttnn::zeros(shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, device);
    CoreRangeSet core_range_set = CoreRangeSet({CoreRange({0, 0}, {7, 0})});

    const auto sharded_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .shard_spec = ShardSpec(core_range_set, {param.h, param.w}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const auto temp_tensor = ttnn::to_memory_config(input_tensor, sharded_memory_config);

    const auto next_sharded_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .shard_spec = ShardSpec(core_range_set, {param.h, param.w}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};

    const auto next_temp_tensor = ttnn::to_memory_config(temp_tensor, next_sharded_memory_config);

    const auto another_sharded_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .shard_spec = ShardSpec(core_range_set, {param.w, param.h}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};

    const auto another_next_temp_tensor = ttnn::to_memory_config(next_temp_tensor, another_sharded_memory_config);

    TT_FATAL(tt::numpy::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(another_next_temp_tensor)));

    const auto reshard_sharded_memory_config = tt::tt_metal::MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .shard_spec = ShardSpec(core_range_set, {param.w, param.h}, tt::tt_metal::ShardOrientation::ROW_MAJOR)};

    // Invalid shard spec...
    // const auto output_tensor = ttnn::to_memory_config(another_next_temp_tensor, reshard_sharded_memory_config);
    // TT_FATAL(tt::numpy::allclose<::bfloat16>(ttnn::from_device(expected_tensor), ttnn::from_device(output_tensor)));

    ttnn::close_device(device);
}

INSTANTIATE_TEST_SUITE_P(
    ToMemoryConfigTests, ChangeLayoutWithToMemoryConfigFixture, ::testing::Values(ToMemoryConfigParam{32, 64}));

}  // namespace test
}  // namespace core
}  // namespace operations
}  // namespace ttnn
