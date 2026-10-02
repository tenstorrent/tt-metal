// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::prim::test {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using tt::tt_metal::BufferType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::ShardSpec;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;

class EmbeddingBackwardMemoryLayoutValidationFixture
    : public TTNNFixtureWithSuiteDevice<EmbeddingBackwardMemoryLayoutValidationFixture> {};

const CoreRangeSet kSingleCoreGrid(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));

MemoryConfig sharded_memory_config(std::array<uint32_t, 2> shard_shape) {
    return MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(kSingleCoreGrid, shard_shape, ShardOrientation::ROW_MAJOR));
}

Tensor make_index_tensor(tt::tt_metal::distributed::MeshDevice* device, const MemoryConfig& memory_config = {}) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 1, 32}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), memory_config)),
        device);
}

Tensor make_gradient_tensor(tt::tt_metal::distributed::MeshDevice* device, const MemoryConfig& memory_config = {}) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 32, 32}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config)),
        device);
}

Tensor make_output_tensor(tt::tt_metal::distributed::MeshDevice* device, const MemoryConfig& memory_config = {}) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 64, 32}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config)),
        device);
}

using Operation = EmbeddingBackwardDeviceOperation;
using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;

Operation::operation_attributes_t make_attributes(const MemoryConfig& output_memory_config = {}) {
    return Operation::operation_attributes_t{
        .output_mem_config = output_memory_config, .output_dtype = DataType::BFLOAT16, .num_embeddings = 64};
}

void expect_miss_and_hit_accept(
    const Operation::operation_attributes_t& attributes, const Operation::tensor_args_t& args) {
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(attributes, args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(attributes, args));
}

void expect_miss_and_hit_reject(
    const Operation::operation_attributes_t& attributes, const Operation::tensor_args_t& args) {
    constexpr auto message = "does not currently support sharding";
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_miss(attributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_hit(attributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
}

}  // namespace

TEST_F(EmbeddingBackwardMemoryLayoutValidationFixture, RejectsEveryMixedShardedOperandBeforeColdOrCachedDispatch) {
    const auto interleaved_attributes = make_attributes();
    const Operation::tensor_args_t interleaved_args{
        .index_tensor = make_index_tensor(device_),
        .grad_tensor = make_gradient_tensor(device_),
        .preallocated_output = std::nullopt,
    };
    expect_miss_and_hit_accept(interleaved_attributes, interleaved_args);

    auto sharded_index_args = interleaved_args;
    sharded_index_args.index_tensor = make_index_tensor(device_, sharded_memory_config({1, 32}));
    expect_miss_and_hit_reject(interleaved_attributes, sharded_index_args);

    auto sharded_gradient_args = interleaved_args;
    sharded_gradient_args.grad_tensor = make_gradient_tensor(device_, sharded_memory_config({32, 32}));
    expect_miss_and_hit_reject(interleaved_attributes, sharded_gradient_args);

    const auto sharded_output_attributes = make_attributes(sharded_memory_config({64, 32}));
    expect_miss_and_hit_reject(sharded_output_attributes, interleaved_args);

    auto sharded_preallocated_output_args = interleaved_args;
    sharded_preallocated_output_args.preallocated_output = make_output_tensor(device_, sharded_memory_config({64, 32}));
    expect_miss_and_hit_reject(interleaved_attributes, sharded_preallocated_output_args);
}

}  // namespace ttnn::prim::test
