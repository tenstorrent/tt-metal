// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace tt::tt_metal;

TensorSpec make_test_tensor_spec() {
    return TensorSpec(ttnn::Shape{1, 1, 32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
}

using DeallocateTest = GenericMeshDeviceFixture;

TEST_F(DeallocateTest, HostTensorDeallocate) {
    Tensor tensor = allocate_tensor_on_host(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force = */ true);
    EXPECT_TRUE(tensor.is_allocated()) << "Host tensor should not be able to be deallocated";
}

TEST_F(DeallocateTest, SingleTensorDeallocate) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    // This tensor is the only instance, so deallocate will succeed.
    tensor.deallocate(/*force = */ false);
    EXPECT_FALSE(tensor.is_allocated()) << "Single tensor should be able to be deallocated";
}

TEST_F(DeallocateTest, SharedTensorDeallocate) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;
    Tensor tensor3(tensor1.device_storage());
    tensor1.deallocate(/*force = */ false);
    EXPECT_TRUE(tensor1.is_allocated() && tensor2.is_allocated() && tensor3.is_allocated())
        << "Shared tensor should not be able to be deallocated when deallocated with force=false";
    tensor2.deallocate(/*force = */ true);
    EXPECT_FALSE(tensor1.is_allocated() || tensor2.is_allocated() || tensor3.is_allocated())
        << "Shared tensor should be able to be deallocated when deallocated with force=true";
}

TEST_F(DeallocateTest, SharedTensorDeallocateForce) {
    Tensor tensor1 = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    Tensor tensor2 = tensor1;
    Tensor tensor3(tensor1.device_storage());
    tensor1.deallocate(/*force = */ true);
    EXPECT_FALSE(tensor1.is_allocated() || tensor2.is_allocated() || tensor3.is_allocated())
        << "Shared tensor should be able to be deallocated when deallocated with force=true";
}

TEST_F(DeallocateTest, DeallocatedTensorHasDevice) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force = */ true);
    EXPECT_FALSE(tensor.is_allocated());

    EXPECT_NE(tensor.device(), nullptr) << "Deallocated tensor should have valid device";

    Tensor tensor2 = tensor;
    EXPECT_NE(tensor2.device(), nullptr) << "Copy of deallocated tensor should have valid device";

    Tensor tensor3(tensor.device_storage());
    EXPECT_NE(tensor3.device(), nullptr) << "Tensor constructed from deallocated storage should have valid device";
}

TEST_F(DeallocateTest, DeallocatedTensorDoesNOTHaveMeshTensor) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force = */ true);
    EXPECT_FALSE(tensor.is_allocated());

    EXPECT_ANY_THROW({ tensor.mesh_tensor(); }) << "Deallocated tensor should not have mesh tensor";

    Tensor tensor1 = tensor;
    EXPECT_ANY_THROW({ tensor1.mesh_tensor(); }) << "Copy of deallocated tensor should not have mesh tensor";

    Tensor tensor2(tensor.device_storage());
    EXPECT_ANY_THROW({ tensor2.mesh_tensor(); })
        << "Tensor constructed from deallocated storage should not have mesh tensor";
}

TEST_F(DeallocateTest, DeallocatedTensorTensorSpec) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    tensor.deallocate(/*force = */ true);
    EXPECT_FALSE(tensor.is_allocated());

    EXPECT_EQ(tensor.tensor_spec(), make_test_tensor_spec()) << "Deallocated tensor should have valid tensor spec";

    Tensor tensor2 = tensor;
    EXPECT_EQ(tensor2.tensor_spec(), make_test_tensor_spec())
        << "Copy of deallocated tensor should have valid tensor spec";
    EXPECT_EQ(tensor2.tensor_spec(), tensor.tensor_spec())
        << "Copy of deallocated tensor should have same tensor spec as original";
}

TEST_F(DeallocateTest, DefaultConstructedThrowsForSpecTopologyAndMeshTensor) {
    DeviceStorage storage;

    EXPECT_THROW(storage.get_tensor_spec(), std::exception);
    EXPECT_THROW(storage.get_tensor_topology(), std::exception);
    EXPECT_THROW(storage.get_mesh_tensor(), std::exception);
}

TEST_F(DeallocateTest, SpecAndTopologyAccessibleAfterDeallocate) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    DeviceStorage storage = tensor.device_storage();

    storage.deallocate();
    ASSERT_FALSE(storage.is_allocated());

    EXPECT_NO_THROW(storage.get_tensor_spec());
    EXPECT_NO_THROW(storage.get_tensor_topology());
}

TEST_F(DeallocateTest, MeshTensorGetterThrowsWhenDeallocated) {
    Tensor tensor = create_device_tensor(make_test_tensor_spec(), mesh_device_.get());
    DeviceStorage storage = tensor.device_storage();

    storage.deallocate();
    ASSERT_FALSE(storage.is_allocated());

    EXPECT_THROW(storage.get_mesh_tensor(), std::exception);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::distributed::test
