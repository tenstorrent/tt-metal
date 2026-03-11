// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {

using namespace tt::tt_metal;

class TensorDeallocateTest : public ttnn::TTNNFixtureWithDevice {};

TEST_F(TensorDeallocateTest, DoubleDeallocateDoesNotCrash) {
    // This test verifies that calling deallocate twice on the same tensor
    // does not cause a crash (segfault). This is a regression test for a bug
    // where double-deallocation would cause a null pointer dereference.
    //
    // The bug was introduced when refactoring mesh_buffer() to return a reference
    // instead of shared_ptr, and the deallocate path didn't properly handle
    // already-deallocated tensors.

    auto* device = device_;
    const ttnn::Shape shape({1, 1, 32, 32});
    MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    constexpr DataType dtype = DataType::BFLOAT16;

    TensorSpec tensor_spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    auto buffer = tensor_impl::allocate_device_buffer(device, tensor_spec);
    DeviceStorage storage(buffer);
    Tensor tensor(storage, tensor_spec, TensorTopology{});

    // First deallocation - should work fine
    tensor.deallocate(/*force=*/true);

    // Second deallocation - should NOT crash (should be a no-op)
    EXPECT_NO_THROW(tensor.deallocate(/*force=*/true));
}

TEST_F(TensorDeallocateTest, DeallocateAfterBufferReset) {
    // Test that deallocate works correctly after the underlying buffer
    // has already been reset/cleared.

    auto* device = device_;
    const ttnn::Shape shape({1, 1, 64, 64});
    MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    constexpr DataType dtype = DataType::BFLOAT16;

    TensorSpec tensor_spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    auto buffer = tensor_impl::allocate_device_buffer(device, tensor_spec);
    DeviceStorage storage(buffer);
    Tensor tensor(storage, tensor_spec, TensorTopology{});

    // Verify tensor is allocated
    EXPECT_TRUE(tensor.is_allocated());

    // Deallocate
    tensor.deallocate(/*force=*/true);

    // Tensor should no longer be allocated
    EXPECT_FALSE(tensor.is_allocated());

    // Deallocating again should be safe
    EXPECT_NO_THROW(tensor.deallocate(/*force=*/true));
    EXPECT_NO_THROW(tensor.deallocate(/*force=*/false));
}

TEST_F(TensorDeallocateTest, MultipleTensorsShareBuffer_DeallocateOne) {
    // Test the scenario where multiple tensors might share an underlying buffer
    // (via shared_ptr semantics) and one is deallocated.

    auto* device = device_;
    const ttnn::Shape shape({1, 1, 32, 32});
    MemoryConfig mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    constexpr DataType dtype = DataType::BFLOAT16;

    TensorSpec tensor_spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    auto buffer = tensor_impl::allocate_device_buffer(device, tensor_spec);
    DeviceStorage storage(buffer);

    // Create first tensor
    Tensor tensor1(storage, tensor_spec, TensorTopology{});
    EXPECT_TRUE(tensor1.is_allocated());

    // Create second tensor sharing the same storage
    Tensor tensor2(storage, tensor_spec, TensorTopology{});
    EXPECT_TRUE(tensor2.is_allocated());

    // Deallocate first tensor with force=false (shouldn't deallocate shared buffer)
    tensor1.deallocate(/*force=*/false);

    // Second tensor should still be usable (buffer not deallocated due to refcount)
    // Note: After deallocate with force=false, the tensor's storage is reset but
    // the underlying buffer may still exist if shared.

    // Force deallocate second tensor
    tensor2.deallocate(/*force=*/true);

    // Both should now be safe to deallocate again
    EXPECT_NO_THROW(tensor1.deallocate(/*force=*/true));
    EXPECT_NO_THROW(tensor2.deallocate(/*force=*/true));
}

}  // namespace
