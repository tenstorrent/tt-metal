// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/buffer_types.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

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
    Tensor tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device_);

    ASSERT_NE(tensor_a.device(), nullptr) << "Tensor A should have valid device";
    ASSERT_TRUE(tensor_a.is_allocated()) << "Tensor A should be allocated";

    // Step 2: Copy the DeviceStorage (simulating what view() does internally)
    // This creates a separate DeviceStorage with a copy of the shared_ptr<MeshBuffer>
    auto storage_copy = tensor_a.device_storage();  // Makes a copy, not a reference
    Tensor tensor_b = Tensor(storage_copy);

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
    Tensor tensor_c = Tensor(storage_b);

    // THE KEY ASSERTION: tensor C must have valid device pointer
    // This is the workaround - without it, device() returns nullptr and causes segfaults
    ASSERT_NE(tensor_c.device(), nullptr)
        << "REGRESSION (issue #40716): Tensor constructed from deallocated storage has null device! "
           "The Tensor constructor should use get_device_bypass_deallocate_check() to preserve mesh_device_.";

    // Verify it's the same device
    EXPECT_EQ(tensor_c.device(), device_) << "Tensor C should have same device as original";
}
