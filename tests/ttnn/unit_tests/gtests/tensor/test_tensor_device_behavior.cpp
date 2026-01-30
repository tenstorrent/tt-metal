// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn_test_fixtures.hpp"

/**
 * @file test_tensor_device_behavior.cpp
 * @brief Unit tests for Tensor class behavior when allocated on device.
 *
 * This test suite verifies the behavior of device tensor member functions as outlined in the
 * Metal Tensor Proposal (tech_reports/runtime-tensor/Metal Tensor Proposal.md).
 *
 * The intent is to establish parity between the current TTNN::Tensor (with device storage)
 * and the future tt_metal::DeviceTensor implementation.
 *
 * Realisitically, this is just teseting no data lost when HostTensor becomes DeviceTensor,
 * This is more of a sanity check and noop for ttnn::Tensor.
 *
 * =============================================================================
 * DeviceTensor Member Functions (from Metal Tensor Proposal)
 * =============================================================================
 *
 * SPECIAL MEMBER FUNCTIONS:
 *   - Default ctor                    [TESTED] (creates nullptr-like state, no resource ownership)
 *   - Copy ctor / Copy assignment     [N/A]    (DeviceTensor is non-copyable)
 *   - Move ctor / Move assignment     [TESTED]
 *
 * DEVICE-SPECIFIC METHODS:
 *   - device                          [TESTED] (returns the device pointer)
 *   - deallocate                      [TESTED] (releases resource, tensor becomes default-like)
 *   - is_allocated                    [TESTED] (checks if device memory is allocated)
 *
 * GETTERS:
 *   - logical_shape / padded_shape    [TESTED]
 *   - dtype / layout                  [TESTED]
 *   - tensor_spec                     [TESTED]
 *   - logical_volume                  [TESTED]
 *   - physical_volume                 [TESTED]
 *   - memory_config                   [TESTED]
 *   - strides                         [TESTED]
 *   - element_size                    [TESTED]
 *   - shard_spec                      [UNTESTED]
 *   - nd_shard_spec                   [UNTESTED]
 *
 * BOOLEAN QUERIES:
 *   - is_sharded                      [TESTED] (false for interleaved tensors)
 *
 * =============================================================================
 * Methods NOT applicable to DeviceTensor (per proposal):
 * =============================================================================
 *   - Copy ctor / Copy assignment     (DeviceTensor is non-copyable, unique ownership)
 *   - "HostBuffer" ctors              (HostTensor only)
 *   - from_borrowed_data              (HostTensor only)
 *   - from_span / from_vector         (HostTensor only for host creation; device uses CQ APIs)
 *   - to_vector / item                (HostTensor only; no implicit data movement)
 *   - to_layout, pad, unpad, etc.     (should be free functions)
 *   - storage_type                    (doesn't make sense to ask)
 *   - host_storage                    (HostTensor only)
 */

namespace ttnn {
namespace {

// =============================================================================
// Test Setup
// =============================================================================

using TensorUnderTest = tt::tt_metal::Tensor;

using PageConfig = tt::tt_metal::PageConfig;
using MemoryConfig = tt::tt_metal::MemoryConfig;
using TensorLayout = tt::tt_metal::TensorLayout;
using DataType = tt::tt_metal::DataType;
using Layout = tt::tt_metal::Layout;
using Shape = tt::tt_metal::Shape;
using HostBuffer = tt::tt_metal::HostBuffer;
using TensorMemoryLayout = tt::tt_metal::TensorMemoryLayout;
using BufferType = tt::tt_metal::BufferType;
using TensorTopology = tt::tt_metal::TensorTopology;
using TensorSpec = tt::tt_metal::TensorSpec;

template <typename T>
std::vector<T> create_test_vector(std::size_t size) {
    std::vector<T> vec(size);
    for (std::size_t i = 0; i < size; i++) {
        vec[i] = static_cast<T>(i);
    }
    return vec;
}

// Helper to create a device tensor from host data
// This is setup so in the future, we can switch it between DeviceTensor and TTNN::Tensor to test both.
TensorUnderTest create_device_tensor(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::vector<float>& host_data,
    const Shape& shape,
    DataType dtype = DataType::BFLOAT16,
    Layout layout = Layout::TILE,
    TensorMemoryLayout mem_layout = TensorMemoryLayout::INTERLEAVED,
    BufferType buffer_type = BufferType::DRAM) {
    MemoryConfig mem_cfg{mem_layout, buffer_type};
    TensorLayout tensor_layout(dtype, PageConfig(layout), mem_cfg);
    TensorSpec spec(shape, tensor_layout);
    return TensorUnderTest::from_vector(host_data, spec, device);
}

// Test fixture for device tensor tests
class TensorDeviceBehaviorTest : public TTNNFixtureWithDevice {};

// =============================================================================
// Special Member Functions
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, DefaultConstruction) {
    // Default constructed DeviceTensor is like a nullptr - no resource ownership
    // This is useful for STL containers, move semantics, and two-phase init patterns
    TensorUnderTest tensor;
    (void)tensor;
}

TEST_F(TensorDeviceBehaviorTest, MoveConstruction) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    auto original_shape = tensor.logical_shape();

    TensorUnderTest tensor_moved(std::move(tensor));

    EXPECT_EQ(tensor_moved.logical_shape(), original_shape);
    EXPECT_EQ(tensor_moved.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor_moved.layout(), Layout::TILE);

    // Verify moved-from object is still assignable (valid state)
    auto vec2 = create_test_vector<float>(32 * 32);
    tensor = create_device_tensor(device_, vec2, Shape({32, 32}));
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
}

TEST_F(TensorDeviceBehaviorTest, MoveAssignment) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));

    TensorUnderTest tensor_moved;
    tensor_moved = std::move(tensor);

    EXPECT_EQ(tensor_moved.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor_moved.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor_moved.layout(), Layout::TILE);

    // Verify moved-from object is still assignable (valid state)
    auto vec2 = create_test_vector<float>(32 * 32);
    tensor = create_device_tensor(device_, vec2, Shape({32, 32}));
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
}

TEST_F(TensorDeviceBehaviorTest, MoveConstruction_DefaultConstructed) {
    TensorUnderTest tensor;
    TensorUnderTest tensor_moved(std::move(tensor));
    (void)tensor_moved;

    // Verify moved-from object is still assignable (valid state)
    auto vec = create_test_vector<float>(32 * 32);
    tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
}

// =============================================================================
// Device-Specific Methods: device / is_allocated / deallocate
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, device) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.device(), device_);
}

TEST_F(TensorDeviceBehaviorTest, is_allocated) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_TRUE(tensor.is_allocated());
}

TEST_F(TensorDeviceBehaviorTest, deallocate) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_TRUE(tensor.is_allocated());

    tensor.deallocate();
    EXPECT_FALSE(tensor.is_allocated());
}

TEST_F(TensorDeviceBehaviorTest, deallocate_ThenReassign) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    tensor.deallocate();
    EXPECT_FALSE(tensor.is_allocated());

    // Should be able to reassign after deallocation
    auto vec2 = create_test_vector<float>(32 * 32);
    tensor = create_device_tensor(device_, vec2, Shape({32, 32}));
    EXPECT_TRUE(tensor.is_allocated());
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
}

// =============================================================================
// Getters: Shape / DType / Layout / TensorSpec
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, logical_shape) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
}

TEST_F(TensorDeviceBehaviorTest, padded_shape) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    // For tile layout, padded shape should be tile-aligned
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 32}));
}

TEST_F(TensorDeviceBehaviorTest, dtype) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}), DataType::BFLOAT16);
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
}

TEST_F(TensorDeviceBehaviorTest, layout) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}), DataType::BFLOAT16, Layout::TILE);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
}

TEST_F(TensorDeviceBehaviorTest, tensor_spec) {
    auto vec = create_test_vector<float>(32 * 32);
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec, device_);
    EXPECT_EQ(tensor.tensor_spec(), spec);
}

// =============================================================================
// Getters: logical_volume / physical_volume
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, logical_volume) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.logical_volume(), 32 * 32);
}

TEST_F(TensorDeviceBehaviorTest, logical_volume_HigherRank) {
    auto vec = create_test_vector<float>(1 * 2 * 32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({1, 2, 32, 32}));
    EXPECT_EQ(tensor.logical_volume(), 1 * 2 * 32 * 32);
}

TEST_F(TensorDeviceBehaviorTest, physical_volume) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.physical_volume(), 32 * 32);
}

// =============================================================================
// Getters: memory_config / strides
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, memory_config_DRAM) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    EXPECT_EQ(tensor.memory_config().memory_layout(), TensorMemoryLayout::INTERLEAVED);
    EXPECT_EQ(tensor.memory_config().buffer_type(), BufferType::DRAM);
}

TEST_F(TensorDeviceBehaviorTest, memory_config_L1) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(
        device_,
        vec,
        Shape({32, 32}),
        DataType::BFLOAT16,
        Layout::TILE,
        TensorMemoryLayout::INTERLEAVED,
        BufferType::L1);
    EXPECT_EQ(tensor.memory_config().memory_layout(), TensorMemoryLayout::INTERLEAVED);
    EXPECT_EQ(tensor.memory_config().buffer_type(), BufferType::L1);
}

TEST_F(TensorDeviceBehaviorTest, strides) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    auto strides = tensor.strides();
    EXPECT_EQ(strides.rank(), 2);
}

// =============================================================================
// Getters: element_size
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, element_size_BFloat16) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}), DataType::BFLOAT16);
    EXPECT_EQ(tensor.element_size(), 2);  // bfloat16 is 2 bytes
}

TEST_F(TensorDeviceBehaviorTest, element_size_Float32) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}), DataType::FLOAT32);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
}

// =============================================================================
// Getters: shard_spec / nd_shard_spec
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, shard_spec_InterleavedReturnsNullopt) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    // Interleaved tensors don't have shard spec
    EXPECT_FALSE(tensor.shard_spec().has_value());
}

TEST_F(TensorDeviceBehaviorTest, nd_shard_spec_InterleavedReturnsNullopt) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    // Interleaved tensors don't have nd shard spec
    EXPECT_FALSE(tensor.nd_shard_spec().has_value());
}

// TODO: Add tests for sharded tensors (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED)
// These require proper shard spec configuration with CoreRangeSet

// =============================================================================
// Boolean Queries: is_sharded
// =============================================================================

TEST_F(TensorDeviceBehaviorTest, is_sharded_Interleaved) {
    auto vec = create_test_vector<float>(32 * 32);
    auto tensor = create_device_tensor(device_, vec, Shape({32, 32}));
    // Interleaved tensor is not sharded
    EXPECT_FALSE(tensor.is_sharded());
}

}  // namespace
}  // namespace ttnn
