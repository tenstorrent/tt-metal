// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
using tt::tt_metal::DeviceStorage;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::MeshDevice1x2Fixture;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;

using DeviceStorageOwnershipTest = MeshDevice1x2Fixture;

TensorSpec make_test_tensor_spec() {
    return TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
}

TEST_F(DeviceStorageOwnershipTest, SoleOwnerAfterCreation) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    const auto& storage = tensor.device_storage();

    EXPECT_TRUE(storage.is_allocated());
    EXPECT_TRUE(storage.is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, CopySharesOwnership) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    EXPECT_TRUE(tensor1.device_storage().is_sole_owner_of_device_memory());

    // Copy constructor shares ownership
    Tensor tensor2 = tensor1;  // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT_FALSE(tensor1.device_storage().is_sole_owner_of_device_memory());
    EXPECT_FALSE(tensor2.device_storage().is_sole_owner_of_device_memory());

    // Copy assignment also shares ownership
    Tensor tensor3 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor3 = tensor1;
    EXPECT_FALSE(tensor3.device_storage().is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, SoleOwnerRestoredAfterCopyDestroyed) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    {
        Tensor tensor2 = tensor1;  // NOLINT(performance-unnecessary-copy-initialization)
        EXPECT_FALSE(tensor1.device_storage().is_sole_owner_of_device_memory());
    }

    EXPECT_TRUE(tensor1.device_storage().is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, MoveTransfersOwnership) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    // Move constructor transfers ownership
    Tensor tensor2 = std::move(tensor1);
    EXPECT_TRUE(tensor2.device_storage().is_sole_owner_of_device_memory());

    // Move assignment transfers ownership
    Tensor tensor3 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor3 = std::move(tensor2);
    EXPECT_TRUE(tensor3.device_storage().is_sole_owner_of_device_memory());
}

TEST_F(DeviceStorageOwnershipTest, DeallocateForce_AlwaysDeallocates) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;

    EXPECT_TRUE(tensor1.is_allocated());
    EXPECT_TRUE(tensor2.is_allocated());

    tensor1.deallocate(/*force=*/true);

    EXPECT_FALSE(tensor1.is_allocated());
    EXPECT_FALSE(tensor2.is_allocated());
}

TEST_F(DeviceStorageOwnershipTest, DeallocateNonForce_OnlyWhenSoleOwner) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;

    // Non-force deallocate is no-op when ownership is shared
    tensor1.deallocate(/*force=*/false);
    EXPECT_TRUE(tensor1.is_allocated());
    EXPECT_TRUE(tensor2.is_allocated());

    // After copy is gone, non-force deallocate works
    tensor2 = Tensor{};
    EXPECT_TRUE(tensor1.device_storage().is_sole_owner_of_device_memory());
    tensor1.deallocate(/*force=*/false);
    EXPECT_FALSE(tensor1.is_allocated());
}

TEST_F(DeviceStorageOwnershipTest, DeallocateIsIdempotent) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());

    tensor.deallocate(/*force=*/true);
    EXPECT_FALSE(tensor.is_allocated());

    EXPECT_NO_THROW(tensor.deallocate(/*force=*/true));
    EXPECT_NO_THROW(tensor.deallocate(/*force=*/false));
}

TEST_F(DeviceStorageOwnershipTest, DefaultConstructedStorageState) {
    DeviceStorage storage;

    EXPECT_FALSE(storage.is_allocated());
    EXPECT_TRUE(storage.is_uniform_storage());
}

TEST_F(DeviceStorageOwnershipTest, GettersThrowWhenDeallocated) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force=*/true);

    EXPECT_THROW(tensor.device_storage().get_buffer(), std::exception);
    EXPECT_THROW(tensor.device_storage().get_mesh_buffer(), std::exception);
    EXPECT_THROW(tensor.device_storage().get_device(), std::exception);
}

TEST_F(DeviceStorageOwnershipTest, TensorShardsShareOwnership) {
    const auto num_devices = mesh_device_->num_devices();
    if (num_devices < 2) {
        GTEST_SKIP() << "Test requires at least 2 devices";
    }

    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
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

TEST_F(DeviceStorageOwnershipTest, DeviceStorageViewSharesOwnership) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
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

    // Force deallocate through view affects original
    view_storage.deallocate(/*force=*/true);
    EXPECT_FALSE(storage.is_allocated());
}

}  // namespace
}  // namespace ttnn::distributed::test
