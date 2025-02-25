// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {

using ::testing::ElementsAre;

using TensorDistributionTest = T3kMultiDeviceFixture;

TensorSpec get_tensor_spec(const ttnn::Shape& shape, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, Layout::ROW_MAJOR, MemoryConfig{}));
}

TEST_F(TensorDistributionTest, DistributeToDevice) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);

    // If no device is provided, the tensor is kept on host.
    EXPECT_TRUE(distribute_tensor(input_tensor, *mapper).storage_type() == StorageType::MULTI_DEVICE_HOST);
    EXPECT_TRUE(distribute_tensor(input_tensor, *mapper, *mesh_device_).storage_type() == StorageType::MULTI_DEVICE);
}

TEST_F(TensorDistributionTest, Replication) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(replicated_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (const auto& device_tensor : device_tensors) {
        EXPECT_THAT(device_tensor.to_vector<float>(), ElementsAre(42.F, 13.F, -99.F));
    }
}

TEST_F(TensorDistributionTest, Shard1DInvalidDim) {
    const int num_devices = mesh_device_->num_devices();
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>(num_devices, 0), get_tensor_spec(ttnn::Shape{1, 1, 1, num_devices}, DataType::FLOAT32));

    EXPECT_ANY_THROW({
        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, -1);
        Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    });

    EXPECT_ANY_THROW({
        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 4);
        Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    });
}

TEST_F(TensorDistributionTest, Shard1DTooFewShards) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_LT(3, num_devices);
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    EXPECT_ANY_THROW({
        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 3);
        Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    });
}

TEST_F(TensorDistributionTest, Shard1D) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_devices, 3, 1}, DataType::FLOAT32));

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i * 1.F, i * 2.F, i * 3.F));
    }

    auto composer = concat_mesh_to_tensor_composer(/*dim=*/0);
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_devices, 1, 3, 1}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

TEST_F(TensorDistributionTest, Shard2DInvalidMeshShape) {
    ASSERT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    EXPECT_ANY_THROW(
        shard_tensor_to_2d_mesh_mapper(*mesh_device_, MeshShape{3, 1}, Shard2dConfig{.row_dim = 1, .col_dim = 2}));

    EXPECT_ANY_THROW(
        shard_tensor_to_2d_mesh_mapper(*mesh_device_, MeshShape{2, 5}, Shard2dConfig{.row_dim = 1, .col_dim = 2}));

    EXPECT_ANY_THROW(
        shard_tensor_to_2d_mesh_mapper(*mesh_device_, MeshShape{1, 1, 2}, Shard2dConfig{.row_dim = 1, .col_dim = 2}));
}

TEST_F(TensorDistributionTest, Shard2DInvalidShardConfig) {
    EXPECT_ANY_THROW(shard_tensor_to_2d_mesh_mapper(*mesh_device_, MeshShape{2, 4}, Shard2dConfig{}));
}

TEST_F(TensorDistributionTest, Concat2DInvalidConfig) {
    EXPECT_ANY_THROW(concat_2d_mesh_to_tensor_composer(*mesh_device_, Concat2dConfig{.row_dim = 2, .col_dim = 2}));
}

TEST_F(TensorDistributionTest, Shard2DReplicateDim) {
    constexpr size_t kNumRows = 2;
    constexpr size_t kNumCols = 4;
    ASSERT_EQ(mesh_device_->shape(), MeshShape(kNumRows, kNumCols));

    std::vector<float> test_data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, 1}, DataType::FLOAT32));
    input_tensor.print();

    auto mapper = shard_tensor_to_2d_mesh_mapper(
        *mesh_device_,
        MeshShape{kNumRows, kNumCols},
        Shard2dConfig{
            .row_dim = 1,
        });
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    sharded_tensor.print();

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());

    int i = 0;
    for (; i < 4; i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(0.0, 1.0, 2.0, 3.0));
    }
    for (; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(4.0, 5.0, 6.0, 7.0));
    }
}

TEST_F(TensorDistributionTest, Shard2D) {
    constexpr size_t kNumRows = 2;
    constexpr size_t kNumCols = 4;
    ASSERT_EQ(mesh_device_->shape(), MeshShape(kNumRows, kNumCols));
    const int num_devices = kNumRows * kNumCols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols, 3}, DataType::FLOAT32));

    auto mapper = shard_tensor_to_2d_mesh_mapper(
        *mesh_device_,
        MeshShape{kNumRows, kNumCols},
        Shard2dConfig{
            .row_dim = 1,
            .col_dim = 2,
        });
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i * 1.F, i * 2.F, i * 3.F));
    }

    auto composer = concat_2d_mesh_to_tensor_composer(
        *mesh_device_,
        Concat2dConfig{
            .row_dim = 0,
            .col_dim = 2,
        });
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{kNumRows, 1, kNumCols, 3}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

}  // namespace ttnn::distributed::test
