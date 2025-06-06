// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;

TensorSpec get_tensor_spec(const ttnn::Shape& shape, DataType dtype) {
    return TensorSpec(shape, TensorLayout(dtype, Layout::ROW_MAJOR, MemoryConfig{}));
}

class AggregateTensorTest : public GenericMeshDeviceFixture,
                            public ::testing::WithParamInterface</*use_borrowed_storage*/ bool> {};

TEST_P(AggregateTensorTest, Roundtrip) {
    const bool use_borrowed_storage = GetParam();
    const int num_devices = mesh_device_->num_devices();
    std::vector<std::vector<float>> test_data(num_devices);
    for (int i = 0; i < num_devices; i++) {
        test_data[i] = std::vector<float>{i * 1.F, i * 2.F, i * 3.F};
    }

    std::vector<Tensor> tensors;
    tensors.reserve(num_devices);

    for (int i = 0; i < num_devices; i++) {
        if (use_borrowed_storage) {
            tensors.push_back(Tensor::from_borrowed_data(
                tt::stl::Span(test_data[i]),
                ttnn::Shape{1, 1, 3, 1},
                /*on_creation_callback=*/[]() {},
                /*on_destruction_callback=*/[]() {}));
        } else {
            tensors.push_back(
                Tensor::from_vector(test_data[i], get_tensor_spec(ttnn::Shape{1, 1, 3, 1}, DataType::FLOAT32)));
        }
    }

    Tensor aggregated_tensor = aggregate_as_tensor(tensors, AllGatherTensor{});
    EXPECT_TRUE(aggregated_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST);

    const auto tensor_shards = get_device_tensors(aggregated_tensor);
    ASSERT_EQ(tensor_shards.size(), test_data.size());

    size_t i = 0;
    for (const auto& tensor_shard : tensor_shards) {
        EXPECT_EQ(tensor_shard.to_vector<float>(), test_data[i++]);
    }
}

INSTANTIATE_TEST_SUITE_P(AggregateTensorTest, AggregateTensorTest, ::testing::Values(true, false));

using TensorDistributionTest = GenericMeshDeviceFixture;

TEST_F(TensorDistributionTest, DistributeToDevice) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);

    // If no device is provided, the tensor is kept on host.
    EXPECT_TRUE(distribute_tensor(input_tensor, *mapper).storage_type() == StorageType::MULTI_DEVICE_HOST);
    EXPECT_TRUE(
        distribute_tensor(input_tensor, *mapper, *mesh_device_).storage_type() != StorageType::MULTI_DEVICE_HOST);
}

TEST_F(TensorDistributionTest, SingleDeviceTensorReplication) {
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

using TensorDistributionT3000Test = T3000MeshDeviceFixture;

TEST_F(TensorDistributionT3000Test, Shard1DInvalidDim) {
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

TEST_F(TensorDistributionT3000Test, Shard1DFewerShardsThanDevices) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data;
    for (int i = 0; i < num_devices - 1; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_devices - 1, 3, 1}, DataType::FLOAT32));

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices() - 1);
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i * 1.F, i * 2.F, i * 3.F));
    }

    auto composer = concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_devices - 1, 1, 3, 1}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

TEST_F(TensorDistributionT3000Test, Shard1D) {
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

    auto composer = concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_devices, 1, 3, 1}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

class TensorDistributionT3000Test2D : public TensorDistributionT3000Test,
                                      public ::testing::WithParamInterface<MeshShape> {};

TEST_P(TensorDistributionT3000Test2D, FullyReplicated) {
    const auto num_rows = GetParam()[0];
    const auto num_cols = GetParam()[1];
    const auto num_devices = num_rows * num_cols;

    std::vector<float> test_data(num_devices);
    std::iota(test_data.begin(), test_data.end(), 0.0);

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_rows, num_cols, 1}, DataType::FLOAT32));

    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Replicate{}},
        },
        MeshShape(num_rows, num_cols));
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), num_devices);

    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), Pointwise(FloatEq(), test_data));
    }
}

TEST_P(TensorDistributionT3000Test2D, ReplicateDim) {
    const auto num_rows = GetParam()[0];
    const auto num_cols = GetParam()[1];
    const auto num_devices = num_rows * num_cols;

    std::vector<float> test_data;
    {
        int i = 0;
        for (; i < num_cols; i++) {
            test_data.push_back(0);
        }
        for (; i < num_devices; i++) {
            test_data.push_back(1);
        }
    }

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_rows, num_cols, 1}, DataType::FLOAT32));

    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Replicate{}},
        },
        MeshShape(num_rows, num_cols));
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), num_devices);

    {
        int i = 0;
        for (; i < num_cols; i++) {
            EXPECT_THAT(device_tensors[i].to_vector<float>(), Each(FloatEq(0.0f)));
        }
        for (; i < device_tensors.size(); i++) {
            EXPECT_THAT(device_tensors[i].to_vector<float>(), Each(FloatEq(1.0f)));
        }
    }
}

TEST_P(TensorDistributionT3000Test2D, ShardDims) {
    const auto num_rows = GetParam()[0];
    const auto num_cols = GetParam()[1];
    const int num_devices = num_rows * num_cols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_rows, num_cols, 3}, DataType::FLOAT32));

    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Shard{2}},
        },
        MeshShape(num_rows, num_cols));
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), num_devices);
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i * 1.F, i * 2.F, i * 3.F));
    }

    auto composer = create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 2},
        },
        MeshShape(num_rows, num_cols));
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_rows, 1, num_cols, 3}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

INSTANTIATE_TEST_SUITE_P(
    TensorDistributionT3000Test2D,
    TensorDistributionT3000Test2D,
    ::testing::Values(MeshShape(1, 1), MeshShape(2, 2), MeshShape(2, 4), MeshShape(1, 3)));

TEST_F(TensorDistributionT3000Test, NdMapperInvalidShape) {
    EXPECT_ANY_THROW(create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}},
        },
        MeshShape{1, 9}));
}

TEST_F(TensorDistributionT3000Test, NdMapperUnevenSharding) {
    constexpr int kNumRows = 2;
    constexpr int kNumCols = 4;

    std::vector<float> test_data;
    for (int i = 0; i < kNumRows * (kNumCols - 1); i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    // Not enough shards for the second dimension.
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, kNumCols - 1, 3}, DataType::FLOAT32));

    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Shard{2}},
        });

    EXPECT_ANY_THROW({ Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_); });
}

TEST_F(TensorDistributionT3000Test, NdMapperInvalidPlacements) {
    // Too few placements.
    EXPECT_ANY_THROW(create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}},
        }));

    // Too many placements.
    EXPECT_ANY_THROW(create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{1}, MeshMapperConfig::Shard{2}},
        }));
}

TEST_F(TensorDistributionT3000Test, NdComposerInvalidShape) {
    EXPECT_ANY_THROW(create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 1, 2},
        },
        MeshShape{1, 9}));
}

TEST_F(TensorDistributionT3000Test, NdComposerInvalidDims) {
    EXPECT_ANY_THROW(create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 1, 2, 3},
        }));
}

TEST_F(TensorDistributionT3000Test, NdComposerUnevenComposition) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));
    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(replicated_tensor);

    auto composer = create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 1},
        },
        MeshShape(2, 3));

    EXPECT_ANY_THROW({ Tensor aggregated_tensor = aggregate_tensor(replicated_tensor, *composer); });
}

TEST_F(TensorDistributionT3000Test, NdMapperShard3D) {
    constexpr size_t kNumRows = 2;
    constexpr size_t kNumCols = 4;
    constexpr size_t kInnerDim = 7;
    constexpr size_t kOuterDim = 4;
    ASSERT_EQ(mesh_device_->shape(), MeshShape(kNumRows, kNumCols));

    std::vector<float> test_data;
    for (int i = 0; i < 4 * kNumRows * kNumCols * 7; ++i) {
        test_data.push_back(static_cast<float>(i));
    }
    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{kOuterDim, kNumRows, kNumCols, kInnerDim}, DataType::FLOAT32));

    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements =
                {
                    MeshMapperConfig::Replicate{},
                    MeshMapperConfig::Shard{2},
                    MeshMapperConfig::Shard{1},
                },
        },
        MeshShape(2, 2, 2));
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (const auto& tensor : device_tensors) {
        EXPECT_EQ(tensor.logical_shape(), ttnn::Shape({kOuterDim, 1, 2, kInnerDim}));
    }

    // Expect the first dim to be replicated.
    std::vector<float> expected_data;
    expected_data.reserve(2 * test_data.size());
    std::copy(test_data.begin(), test_data.end(), std::back_inserter(expected_data));
    std::copy(test_data.begin(), test_data.end(), std::back_inserter(expected_data));

    auto composer = create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 2, 1},
        },
        MeshShape(2, 2, 2));
    Tensor aggregated_tensor = aggregate_tensor(sharded_tensor, *composer);
    EXPECT_EQ(aggregated_tensor.logical_shape(), ttnn::Shape({2 * kOuterDim, kNumRows, kNumCols, kInnerDim}));
    EXPECT_THAT(aggregated_tensor.to_vector<float>(), Pointwise(FloatEq(), expected_data));
}

}  // namespace ttnn::distributed::test
