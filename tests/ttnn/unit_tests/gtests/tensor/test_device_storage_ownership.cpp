// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::distributed::test {
namespace {

using ::testing::SizeIs;

using tt::tt_metal::DataType;
using tt::tt_metal::GenericMeshDeviceFixture;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::MeshDevice1x2Fixture;
using tt::tt_metal::MeshTensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorTopology;
using ttnn::DeviceStorage;
using ttnn::Tensor;

// Most ownership tests only need a single device.
using DeviceStorageOwnershipTest = GenericMeshDeviceFixture;
// Tests that explicitly exercise multi-device behaviour (views, shards).
using DeviceStorageMultiDeviceTest = MeshDevice1x2Fixture;

TensorSpec make_test_tensor_spec() {
    return TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
}

// ======================================================================================
// DeviceStorage Ownership Tests
//
// These tests verify the ownership semantics of DeviceStorage directly.
// DeviceStorage tracks ownership via mesh_buffer shared_ptr use_count.
// ======================================================================================

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_DefaultConstructedState) {
    DeviceStorage storage;

    EXPECT_FALSE(storage.is_allocated());
    EXPECT_THROW(storage.is_uniform_storage(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_ThrowsWhenConstructedFromMovedFromMeshTensor) {
    auto source_mesh_tensor = MeshTensor::allocate_on_device(*mesh_device_, make_test_tensor_spec(), TensorTopology{});
    MeshTensor moved_mesh_tensor(std::move(source_mesh_tensor));

    EXPECT_TRUE(source_mesh_tensor.is_valueless_after_move());  // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(moved_mesh_tensor.is_valueless_after_move());

    auto construct_storage_from_moved_from = [&]() {
        DeviceStorage storage(std::move(source_mesh_tensor));  // NOLINT(bugprone-use-after-move)
        (void)storage;
    };
    EXPECT_THROW(construct_storage_from_moved_from(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_SoleOwnerAfterCreation) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& storage = tensor.device_storage();

    EXPECT_TRUE(storage.is_allocated());
    EXPECT_TRUE(storage.is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_CopySharesOwnership) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& original_storage = tensor.device_storage();

    // Copying DeviceStorage shares the underlying mesh_buffer
    DeviceStorage storage_copy = original_storage;  // NOLINT(performance-unnecessary-copy-initialization)

    // Both now share ownership (mesh_buffer.use_count() > 1)
    EXPECT_FALSE(original_storage.is_sole_owner_of_device_memory());
    EXPECT_FALSE(storage_copy.is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_SoleOwnerRestoredAfterCopyDestroyed) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& original_storage = tensor.device_storage();
    EXPECT_TRUE(original_storage.is_sole_owner_of_device_memory());

    {
        // Copying DeviceStorage shares ownership
        DeviceStorage storage_copy = original_storage;  // NOLINT(performance-unnecessary-copy-initialization)
        EXPECT_FALSE(original_storage.is_sole_owner_of_device_memory());
        EXPECT_FALSE(storage_copy.is_sole_owner_of_device_memory());
    }

    // After copy is destroyed, sole ownership is restored
    EXPECT_TRUE(original_storage.is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_MoveConstructorTransfersOwnership) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    DeviceStorage original = tensor.device_storage();
    EXPECT_TRUE(original.is_allocated());

    DeviceStorage moved_into(std::move(original));

    // Moved-into storage has the memory; moved-from is in DeallocatedDefaultConstructed state.
    EXPECT_TRUE(moved_into.is_allocated());
    EXPECT_FALSE(original.is_allocated());  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_MoveAssignmentTransfersOwnership) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    DeviceStorage original = tensor.device_storage();
    EXPECT_TRUE(original.is_allocated());

    DeviceStorage moved_into;
    moved_into = std::move(original);

    // Moved-into storage has the memory; moved-from is in DeallocatedDefaultConstructed state.
    EXPECT_TRUE(moved_into.is_allocated());
    EXPECT_FALSE(original.is_allocated());  // NOLINT(bugprone-use-after-move)
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_MoveDoesNotAddSharedReference) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    ASSERT_TRUE(tensor.device_storage().is_sole_owner_of_device_memory());

    // Copying the storage increments the use-count — tensor is no longer sole owner.
    {
        DeviceStorage copy = tensor.device_storage();  // NOLINT(performance-unnecessary-copy-initialization)
        EXPECT_FALSE(tensor.device_storage().is_sole_owner_of_device_memory());
    }
    // Copy destroyed — sole ownership restored.
    EXPECT_TRUE(tensor.device_storage().is_sole_owner_of_device_memory());

    // Moving the storage does NOT increment the use-count: it transfers the
    // existing slot.  After both the moved-from (temp) and moved-into objects
    // are destroyed, tensor is the sole owner again — proving no extra reference
    // was added by the move.
    {
        DeviceStorage temp = tensor.device_storage();  // copy: use_count = 2
        DeviceStorage moved_into(std::move(temp));     // move: use_count stays 2, temp deallocated
        EXPECT_FALSE(temp.is_allocated());  // NOLINT(bugprone-use-after-move)
    }
    // temp and moved_into both gone — sole ownership restored.
    EXPECT_TRUE(tensor.device_storage().is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageMultiDeviceTest, DeviceStorage_ViewSharesOwnership) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& storage = tensor.device_storage();

    auto coords = storage.get_coords();
    ASSERT_THAT(coords, SizeIs(2));

    // Create a view with subset of coords
    std::vector<distributed::MeshCoordinate> subset_coords = {coords[0]};
    DeviceStorage view_storage(storage, subset_coords);

    EXPECT_TRUE(view_storage.is_allocated());
    EXPECT_FALSE(storage.is_sole_owner_of_device_memory());
    EXPECT_FALSE(view_storage.is_sole_owner_of_device_memory());
    ASSERT_THAT(view_storage.get_coords(), SizeIs(1));
}

TEST_F(DeviceStorageMultiDeviceTest, DeviceStorage_ViewDeallocateAffectsOwner) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& storage = tensor.device_storage();

    std::vector<distributed::MeshCoordinate> subset_coords = {storage.get_coords()[0]};
    DeviceStorage view_storage(storage, subset_coords);

    // Deallocate through view affects original
    view_storage.deallocate();
    EXPECT_FALSE(view_storage.is_allocated());
    EXPECT_FALSE(storage.is_allocated());
}

TEST_F(DeviceStorageMultiDeviceTest, DeviceStorage_OwnerDeallocateAffectsView) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    DeviceStorage owner_storage = tensor.device_storage();

    std::vector<distributed::MeshCoordinate> subset_coords = {owner_storage.get_coords()[0]};
    DeviceStorage view_storage(owner_storage, subset_coords);

    // Deallocate through owner affects view
    owner_storage.deallocate();
    EXPECT_FALSE(owner_storage.is_allocated());
    EXPECT_FALSE(view_storage.is_allocated());
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_BufferGettersThrowWhenDeallocated) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force=*/true);

    EXPECT_THROW(tensor.device_storage().get_buffer(), std::exception);
    EXPECT_THROW(tensor.device_storage().get_mesh_buffer(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_ReleaseMeshTensorMovesOutUnderlyingMemory) {
    auto mesh_tensor = MeshTensor::allocate_on_device(*mesh_device_, make_test_tensor_spec(), TensorTopology{});
    DeviceStorage storage(std::move(mesh_tensor));

    // Capture identity of the underlying device memory before releasing.
    const auto address = storage.get_mesh_tensor().mesh_buffer().address();

    MeshTensor released = storage.release_mesh_tensor();

    // The released tensor is valid and owns the very same MeshBuffer that the storage held —
    // proving the memory was moved out, not copied or reallocated.
    EXPECT_FALSE(released.is_valueless_after_move());
    EXPECT_EQ(released.address(), address);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_ReleaseMeshTensorLeavesDefaultConstructedState) {
    auto mesh_tensor = MeshTensor::allocate_on_device(*mesh_device_, make_test_tensor_spec(), TensorTopology{});
    DeviceStorage storage(std::move(mesh_tensor));
    ASSERT_TRUE(storage.is_allocated());

    MeshTensor released = storage.release_mesh_tensor();

    // Post-condition: storage is equivalent to a default-constructed DeviceStorage.
    EXPECT_FALSE(storage.is_allocated());
    EXPECT_THROW(storage.get_mesh_tensor(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_ReleaseMeshTensorThrowsWhenDefaultConstructed) {
    DeviceStorage storage;
    ASSERT_FALSE(storage.is_allocated());

    EXPECT_THROW(storage.release_mesh_tensor(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, DeviceStorage_ReleaseMeshTensorThrowsWhenDeallocated) {
    auto mesh_tensor = MeshTensor::allocate_on_device(*mesh_device_, make_test_tensor_spec(), TensorTopology{});
    DeviceStorage storage(std::move(mesh_tensor));
    storage.deallocate();
    ASSERT_FALSE(storage.is_allocated());

    EXPECT_THROW(storage.release_mesh_tensor(), std::exception);
}

// ======================================================================================
// Tensor Deallocation Tests
//
// These tests verify Tensor-level deallocation behavior.
// Tensor tracks ownership via tensor_attributes shared_ptr, which is separate from
// DeviceStorage's mesh_buffer ownership tracking.
// ======================================================================================

TEST_F(DeviceStorageOwnershipTest, Tensor_MoveTransfersOwnership) {
    Tensor tensor1 = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    // Move constructor transfers ownership
    Tensor tensor2 = std::move(tensor1);
    EXPECT_TRUE(tensor2.device_storage().is_sole_owner_of_device_memory());

    // Move assignment transfers ownership
    Tensor tensor3 = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor3 = std::move(tensor2);
    EXPECT_TRUE(tensor3.device_storage().is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, Tensor_DeallocateForceAlwaysDeallocates) {
    Tensor tensor1 = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;  // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_TRUE(tensor1.is_allocated());
    EXPECT_TRUE(tensor2.is_allocated());

    tensor1.deallocate(/*force=*/true);

    EXPECT_FALSE(tensor1.is_allocated());
    EXPECT_FALSE(tensor2.is_allocated());
}

TEST_F(DeviceStorageOwnershipTest, Tensor_DeallocateNonForceOnlyWhenSoleOwner) {
    Tensor tensor1 = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;  // NOLINT(performance-unnecessary-copy-initialization)

    // Non-force deallocate is no-op when ownership is shared
    tensor1.deallocate(/*force=*/false);
    EXPECT_TRUE(tensor1.is_allocated());
    EXPECT_TRUE(tensor2.is_allocated());

    // After copy is gone, non-force deallocate works
    tensor2 = Tensor{};
    tensor1.deallocate(/*force=*/false);
    EXPECT_FALSE(tensor1.is_allocated());
}

TEST_F(DeviceStorageOwnershipTest, Tensor_DeallocateIsIdempotent) {
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    tensor.deallocate(/*force=*/true);
    EXPECT_FALSE(tensor.is_allocated());

    EXPECT_NO_THROW(tensor.deallocate(/*force=*/true));
    EXPECT_NO_THROW(tensor.deallocate(/*force=*/false));
}

TEST_F(DeviceStorageMultiDeviceTest, Tensor_ShardsShareDeviceStorageOwnership) {
    const auto num_devices = mesh_device_->num_devices();
    Tensor tensor = ttnn::create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    auto shards = get_device_tensors(tensor);

    ASSERT_THAT(shards, SizeIs(num_devices));

    // Original and all shards share ownership
    EXPECT_FALSE(tensor.device_storage().is_sole_owner_of_device_memory());
    for (const auto& shard : shards) {
        EXPECT_TRUE(shard.device_storage().is_allocated());
        EXPECT_FALSE(shard.device_storage().is_sole_owner_of_device_memory());
    }

    // Force deallocate through shard affects all
    shards[0].deallocate(/*force=*/true);
    EXPECT_FALSE(tensor.is_allocated());
    for (size_t i = 1; i < shards.size(); ++i) {
        EXPECT_FALSE(shards[i].is_allocated());
    }
}

}  // namespace
}  // namespace ttnn::distributed::test
