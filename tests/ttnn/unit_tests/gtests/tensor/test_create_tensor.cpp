// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string_view>
#include <tuple>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace {

void run_create_tensor_test(tt::tt_metal::distributed::MeshDevice* device, const ttnn::Shape& input_shape) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    const ttnn::QueueId io_cq = ttnn::QueueId(0);
    constexpr DataType dtype = DataType::BFLOAT16;
    constexpr uint32_t datum_size_bytes = 2;

    auto input_buf_size_datums = input_shape.volume();

    auto host_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);

    for (int i = 0; i < input_buf_size_datums; i++) {
        host_data[i] = 1;
    }

    TensorSpec tensor_spec(input_shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    ASSERT_EQ(input_buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
    auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(device, tensor_spec);

    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {tt::tt_metal::distributed::MeshCoordinate{0, 0}}};

    Tensor input_tensor = Tensor(input_storage, tensor_spec, TensorTopology{});

    ttnn::write_buffer(io_cq, input_tensor, {host_data});

    ttnn::read_buffer(io_cq, input_tensor, {readback_data});

    for (int i = 0; i < input_buf_size_datums; i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
    }

    input_tensor.deallocate();
}

struct CreateTensorParams {
    ttnn::Shape shape;
};

}  // namespace

class CreateTensorTest : public ttnn::TTNNFixtureWithSuiteDevice<CreateTensorTest>,
                         public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorTest, Tile) {
    const CreateTensorParams& params = GetParam();
    run_create_tensor_test(device_, params.shape);
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorTestWithShape,
    CreateTensorTest,
    ::testing::Values(
        CreateTensorParams{.shape = ttnn::Shape({1, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({2, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({0, 0, 0, 0})},
        CreateTensorParams{.shape = ttnn::Shape({0, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({0})}));

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& value) {
    os << enchantum::to_string(value);
    return os;
}

using CombinationInputParams =
    std::tuple<ttnn::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;
class EmptyTensorTest : public ttnn::TTNNFixtureWithSuiteDevice<EmptyTensorTest>,
                        public ::testing::WithParamInterface<CombinationInputParams> {};

TEST_P(EmptyTensorTest, Combinations) {
    auto params = GetParam();
    auto shape = std::get<0>(params);
    auto dtype = std::get<1>(params);
    auto layout = std::get<2>(params);
    auto memory_config = std::get<3>(params);
    log_info(
        tt::LogTest,
        "Running test with shape={}, dtype={}, layout={}, memory_config={}",
        shape,
        dtype,
        layout,
        memory_config);

    if (layout == tt::tt_metal::Layout::ROW_MAJOR && dtype == tt::tt_metal::DataType::BFLOAT8_B) {
        GTEST_SKIP() << "Skipping test with ROW_MAJOR layout and BFLOAT8_B dtype!";
    }

    auto tensor_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        dtype, PageConfig(layout), memory_config, /* logical */ shape, /* padded */ shape);

    auto tensor = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        device_);
    EXPECT_EQ(tensor.logical_shape(), shape);

    test_utils::test_tensor_on_device(shape, tensor_layout, device_);
}

INSTANTIATE_TEST_SUITE_P(
    EmptyTensorTestWithShape,
    EmptyTensorTest,
    ::testing::Combine(
        ::testing::Values(
            ttnn::Shape({}),
            ttnn::Shape({0}),
            ttnn::Shape({1}),
            ttnn::Shape({1, 2}),
            ttnn::Shape({1, 2, 3}),
            ttnn::Shape({1, 2, 3, 4}),
            // ttnn::Shape({0, 0, 0, 0}), fails with width sharded case
            ttnn::Shape({1, 1, 1, 1}),
            // ttnn::Shape({0, 1, 32, 32}), fails with width sharded case
            ttnn::Shape({1, 1, 32, 32}),
            ttnn::Shape({2, 1, 32, 32}),
            ttnn::Shape({64, 1, 256, 1}),
            ttnn::Shape({1, 1, 21120, 16}),
            ttnn::Shape({1, 2, 3, 4, 5})),

        ::testing::Values(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::DataType::BFLOAT8_B),

        ::testing::Values(tt::tt_metal::Layout::TILE, tt::tt_metal::Layout::ROW_MAJOR),

        ::testing::Values(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1},

            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}

            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 1}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 1}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },

            // ttnn::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },

            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
            //         {64, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // }
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
            //         {64, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // }
            )));

// =============================================================================
// Workaround test for https://github.com/tenstorrent/tt-metal/issues/40716
//
// This test verifies that constructing a Tensor from a deallocated DeviceStorage
// still preserves the mesh_device_ pointer. This is a WORKAROUND for models that
// incorrectly operate on deallocated tensors (e.g., reshape after deallocate in
// llama_attention.py).
//
// TODO: Remove this test once the underlying model bugs are fixed. The models
// should not be passing deallocated tensors to operations like reshape. Once
// models are fixed to properly manage tensor lifetimes, this workaround in the
// Tensor constructor (using get_device_bypass_deallocate_check()) should be
// reverted to use get_device() with proper is_allocated() checks.
// =============================================================================
class TensorFromDeallocatedStorageTest : public ttnn::TTNNFixtureWithSuiteDevice<TensorFromDeallocatedStorageTest> {};

TEST_F(TensorFromDeallocatedStorageTest, ConstructingTensorFromDeallocatedStoragePreservesMeshDevice) {
    // This test verifies the workaround for issue #40716:
    // When a new Tensor is constructed from a DeviceStorage that has been deallocated,
    // the mesh_device_ should still be set (not nullptr).
    //
    // The bug scenario (from Llama 70B Galaxy model):
    // 1. Tensor A is created on device
    // 2. view(A) creates Tensor B - B's DeviceStorage has a copy of shared_ptr<MeshBuffer>
    // 3. A is deallocated - A's mesh_buffer is reset to nullptr, but B still has the ptr
    // 4. B's MeshBuffer is now in DeallocatedState (is_allocated() == false)
    // 5. view(B) creates Tensor C from B's DeviceStorage
    // 6. C's device() returns nullptr because is_allocated() was false -> segfault

    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    ttnn::Shape shape({1, 1, 32, 32});
    TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    // Step 1: Create initial tensor on device
    auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(device_, tensor_spec);
    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {tt::tt_metal::distributed::MeshCoordinate{0, 0}}};
    Tensor tensor_a = Tensor(input_storage, tensor_spec, TensorTopology{});

    ASSERT_NE(tensor_a.device(), nullptr) << "Tensor A should have valid device";
    ASSERT_TRUE(tensor_a.is_allocated()) << "Tensor A should be allocated";

    // Step 2: Copy the DeviceStorage (simulating what view() does internally)
    // This creates a separate DeviceStorage with a copy of the shared_ptr<MeshBuffer>
    auto storage_copy = tensor_a.device_storage();  // Makes a copy, not a reference
    Tensor tensor_b = Tensor(storage_copy, tensor_spec, TensorTopology{});

    ASSERT_NE(tensor_b.device(), nullptr) << "Tensor B should have valid device";
    ASSERT_TRUE(tensor_b.is_allocated()) << "Tensor B should be allocated";

    // Step 3: Deallocate tensor A with force=true
    // This calls mesh_buffer->deallocate() putting it in DeallocatedState,
    // then calls mesh_buffer.reset() on A's storage (setting A's ptr to nullptr)
    // But tensor_b's storage still holds the shared_ptr to the now-deallocated MeshBuffer!
    tensor_a.deallocate(/*force=*/true);

    // Verify B's storage state: mesh_buffer exists but is deallocated
    const auto& storage_b = tensor_b.device_storage();
    EXPECT_FALSE(storage_b.is_allocated()) << "is_allocated() should return false (MeshBuffer is in DeallocatedState)";
    EXPECT_NE(storage_b.get_device_bypass_deallocate_check(), nullptr)
        << "get_device_bypass_deallocate_check() should return device even when deallocated";

    // Step 4: Create tensor C from B's storage (simulating another view() call)
    // This is where the bug manifests - if Tensor constructor uses is_allocated(),
    // mesh_device_ won't be set because is_allocated() returns false
    Tensor tensor_c = Tensor(storage_b, tensor_spec, TensorTopology{});

    // THE KEY ASSERTION: tensor C must have valid device pointer
    // This is the workaround - without it, device() returns nullptr and causes segfaults
    ASSERT_NE(tensor_c.device(), nullptr)
        << "REGRESSION (issue #40716): Tensor constructed from deallocated storage has null device! "
           "The Tensor constructor should use get_device_bypass_deallocate_check() to preserve mesh_device_.";

    // Verify it's the same device
    EXPECT_EQ(tensor_c.device(), device_) << "Tensor C should have same device as original";
}
