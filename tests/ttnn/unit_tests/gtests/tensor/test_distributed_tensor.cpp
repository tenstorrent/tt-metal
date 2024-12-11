// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "common/bfloat16.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn_test_fixtures.hpp"
#include <exception>
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {

using ::ttnn::experimental::xtensor::from_vector;

using TensorDistributionTest = T3kMultiDeviceFixture;

TEST_F(TensorDistributionTest, Replication) {
    Tensor input_tensor = from_vector(std::vector<float>{42.F, 13.F, -99.F}, ttnn::Shape{1, 1, 1, 3});

    auto mapper = api::replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);

    std::vector<Tensor> device_tensors = api::get_device_tensors(replicated_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (const auto& device_tensor : device_tensors) {
        EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(device_tensor.cpu(), input_tensor));
    }
}

TEST_F(TensorDistributionTest, Shard1DInvalidDim) {
    const int num_devices = mesh_device_->num_devices();
    Tensor input_tensor = from_vector(std::vector<float>(num_devices, 0), ttnn::Shape{1, 1, 1, num_devices});

    EXPECT_THROW(
        {
            auto mapper = api::shard_tensor_to_mesh_mapper(*mesh_device_, -1);
            Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);
        },
        std::exception);

    EXPECT_THROW(
        {
            auto mapper = api::shard_tensor_to_mesh_mapper(*mesh_device_, 4);
            Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);
        },
        std::exception);
}

TEST_F(TensorDistributionTest, Shard1DTooFewShards) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_LT(3, num_devices);
    Tensor input_tensor = from_vector(std::vector<float>{42.F, 13.F, -99.F}, ttnn::Shape{1, 1, 1, 3});

    EXPECT_THROW(
        {
            auto mapper = api::shard_tensor_to_mesh_mapper(*mesh_device_, 3);
            Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);
        },
        std::exception);
}

TEST_F(TensorDistributionTest, Shard1D) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor = from_vector(test_data, ttnn::Shape{1, num_devices, 3, 1});

    auto mapper = api::shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);

    std::vector<Tensor> device_tensors = api::get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (int i = 0; i < device_tensors.size(); i++) {
        auto expected = from_vector(std::vector<float>{i * 1.F, i * 2.F, i * 3.F}, ttnn::Shape{1, 1, 3, 1});
        EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(device_tensors[i].cpu(), expected));
    }

    auto composer = api::concat_mesh_to_tensor_composer(/*dim=*/0);
    Tensor concatenated_tensor = api::aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor = from_vector(test_data, ttnn::Shape{num_devices, 1, 3, 1});
    EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(concatenated_tensor, expected_tensor));
}

TEST_F(TensorDistributionTest, Shard2DInvalidMeshShape) {
    const auto [num_rows, num_cols] = mesh_device_->shape();
    ASSERT_EQ(num_rows, 2);
    ASSERT_EQ(num_cols, 4);

    EXPECT_THROW(
        api::shard_tensor_2d_to_mesh_mapper(*mesh_device_, MeshShape{3, 1}, Shard2dConfig{.row_dim = 1, .col_dim = 2}),
        std::exception);

    EXPECT_THROW(
        api::shard_tensor_2d_to_mesh_mapper(*mesh_device_, MeshShape{2, 5}, Shard2dConfig{.row_dim = 1, .col_dim = 2}),
        std::exception);
}

TEST_F(TensorDistributionTest, Shard2DInvalidShardConfig) {
    EXPECT_THROW(api::shard_tensor_2d_to_mesh_mapper(*mesh_device_, MeshShape{2, 4}, Shard2dConfig{}), std::exception);
}

TEST_F(TensorDistributionTest, Concat2DInvalidConfig) {
    EXPECT_THROW(
        api::concat_mesh_2d_to_tensor_composer(*mesh_device_, Concat2dConfig{.row_dim = 2, .col_dim = 2}),
        std::exception);
}

TEST_F(TensorDistributionTest, Shard2DReplicateDim) {
    const auto [num_rows, num_cols] = mesh_device_->shape();
    ASSERT_EQ(num_rows, 2);
    ASSERT_EQ(num_cols, 4);
    const int num_devices = num_rows * num_cols;

    std::vector<float> test_data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tensor input_tensor = from_vector(test_data, ttnn::Shape{1, num_rows, num_cols, 1});
    input_tensor.print();

    auto mapper = api::shard_tensor_2d_to_mesh_mapper(
        *mesh_device_,
        MeshShape{num_rows, num_cols},
        Shard2dConfig{
            .row_dim = 1,
        });
    Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);
    sharded_tensor.print();

    std::vector<Tensor> device_tensors = api::get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());

    int i = 0;
    for (; i < 4; i++) {
        auto expected = from_vector(std::vector<float>{0.0, 1.0, 2.0, 3.0}, ttnn::Shape{1, 1, 4, 1});
        EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(device_tensors[i].cpu(), expected));
    }
    for (; i < device_tensors.size(); i++) {
        auto expected = from_vector(std::vector<float>{4.0, 5.0, 6.0, 7.0}, ttnn::Shape{1, 1, 4, 1});
        EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(device_tensors[i].cpu(), expected));
    }
}

TEST_F(TensorDistributionTest, Shard2D) {
    const auto [num_rows, num_cols] = mesh_device_->shape();
    ASSERT_EQ(num_rows, 2);
    ASSERT_EQ(num_cols, 4);
    const int num_devices = num_rows * num_cols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor = from_vector(test_data, ttnn::Shape{1, num_rows, num_cols, 3});

    auto mapper = api::shard_tensor_2d_to_mesh_mapper(
        *mesh_device_,
        MeshShape{num_rows, num_cols},
        Shard2dConfig{
            .row_dim = 1,
            .col_dim = 2,
        });
    Tensor sharded_tensor = api::distribute_tensor(input_tensor, *mesh_device_, *mapper);

    std::vector<Tensor> device_tensors = api::get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (int i = 0; i < device_tensors.size(); i++) {
        auto expected = from_vector(std::vector<float>{i * 1.F, i * 2.F, i * 3.F}, ttnn::Shape{1, 1, 1, 3});
        EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(device_tensors[i].cpu(), expected));
    }

    auto composer = api::concat_mesh_2d_to_tensor_composer(
        *mesh_device_,
        Concat2dConfig{
            .row_dim = 0,
            .col_dim = 2,
        });
    Tensor concatenated_tensor = api::aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor = from_vector(test_data, ttnn::Shape{num_rows, 1, num_cols, 3});
    EXPECT_TRUE(ttnn::numpy::allclose<bfloat16>(concatenated_tensor, expected_tensor));
}

}  // namespace ttnn::distributed::test
