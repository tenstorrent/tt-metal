// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "host_buffer.hpp"
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

// Returns the number of unique buffers in host-side multi-device tensor.
int count_unique_buffers(const Tensor& tensor) {
    std::unordered_set<const void*> buffer_addresses;
    tensor.host_storage().buffer().apply(
        [&buffer_addresses](const HostBuffer& buffer) { buffer_addresses.insert(buffer.view_bytes().data()); });
    return buffer_addresses.size();
}

using TensorDistributionTest = GenericMeshDeviceFixture;
using TensorDistribution2x4Test = MeshDevice2x4Fixture;

TEST_F(TensorDistributionTest, DistributeToDevice) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);

    // If no device is provided, the tensor is kept on host.
    EXPECT_TRUE(distribute_tensor(input_tensor, *mapper).storage_type() == StorageType::HOST);
    EXPECT_TRUE(distribute_tensor(input_tensor, *mapper, *mesh_device_).storage_type() != StorageType::HOST);

    // Tensor topology is a single device
    const auto& tensor_topology = input_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(1));
    EXPECT_EQ(
        tensor_topology.placements(),
        (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Replicate{}}));
}

TEST_F(TensorDistributionTest, SingleDeviceTensorReplication) {
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>{42.F, 13.F, -99.F}, get_tensor_spec(ttnn::Shape{1, 1, 1, 3}, DataType::FLOAT32));

    auto mapper = replicate_tensor_to_mesh_mapper(*mesh_device_);
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor replicated across entire mesh should be 1D shape with number of devices
    const auto& tensor_topology = replicated_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(mesh_device_->num_devices()));
    EXPECT_EQ(
        tensor_topology.placements(),
        (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Replicate{}}));

    std::vector<Tensor> device_tensors = get_device_tensors(replicated_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (const auto& device_tensor : device_tensors) {
        EXPECT_THAT(device_tensor.to_vector<float>(), ElementsAre(42.F, 13.F, -99.F));
    }
}

TEST_F(TensorDistribution2x4Test, Shard1DInvalidDim) {
    const int num_devices = mesh_device_->num_devices();
    Tensor input_tensor = Tensor::from_vector(
        std::vector<float>(num_devices, 0), get_tensor_spec(ttnn::Shape{1, 1, 1, num_devices}, DataType::FLOAT32));

    EXPECT_ANY_THROW({
        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, -10);
        Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    });

    EXPECT_ANY_THROW({
        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 4);
        Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    });
}

TEST_F(TensorDistribution2x4Test, Shard1DFewerShardsThanDevices) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data;
    for (int i = 0; i < num_devices - 1; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_devices - 1, 3, 1}, DataType::FLOAT32));

    const int shard_dim = 1;
    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, shard_dim);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded across 1 dimension should be 1D shape with number of actual shards (ie.
    // num_devices - 1)
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(mesh_device_->num_devices() - 1));
    EXPECT_EQ(
        tensor_topology.placements(),
        (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Shard{shard_dim}}));

    EXPECT_EQ(count_unique_buffers(sharded_tensor), mesh_device_->num_devices() - 1);

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

TEST_F(TensorDistribution2x4Test, Shard1DNegativeDim) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data(num_devices, 0);
    std::iota(test_data.begin(), test_data.end(), 0);
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, 1, 1, num_devices}, DataType::FLOAT32));

    const int shard_dim = -1;
    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, shard_dim);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor sharded across 1 dimension should be 1D shape with number of actual shards (ie.
    // num_devices)
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(mesh_device_->num_devices()));
    EXPECT_EQ(
        tensor_topology.placements(),
        (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Shard{shard_dim}}));

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), mesh_device_->num_devices());
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i));
    }
}

TEST_F(TensorDistribution2x4Test, Shard1D) {
    const int num_devices = mesh_device_->num_devices();
    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_devices, 3, 1}, DataType::FLOAT32));

    const int shard_dim = 1;
    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, shard_dim);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded across 1 dimension should be 1D shape with number of actual shards (ie.
    // num_devices)
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(mesh_device_->num_devices()));
    EXPECT_EQ(
        tensor_topology.placements(),
        (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Shard{shard_dim}}));

    EXPECT_EQ(count_unique_buffers(sharded_tensor), mesh_device_->num_devices());

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

TEST_F(TensorDistribution2x4Test, PartialConcat) {
    constexpr int kNumRows = 2;
    std::vector<float> test_data;
    for (int i = 0; i < kNumRows; i++) {
        test_data.insert(test_data.end(), {(i * 10) + 0, (i * 10) + 1, (i * 10) + 2});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, kNumRows, 1, 3}, DataType::FLOAT32));

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Replicate{}},
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded/replicated across 2D should be mesh device shape
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_device_->shape());
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

    EXPECT_EQ(count_unique_buffers(sharded_tensor), kNumRows);

    // Full concat.
    EXPECT_THAT(
        aggregate_tensor(
            sharded_tensor,
            *create_mesh_composer(
                *mesh_device_,
                MeshComposerConfig{
                    .dims = {-1, 0},
                }))
            .to_vector<float>(),
        // `0, 1, 2` resides on rows; `10, 11, 12` resides on columns.
        ElementsAre(0, 1, 2, 10, 11, 12, 0, 1, 2, 10, 11, 12, 0, 1, 2, 10, 11, 12, 0, 1, 2, 10, 11, 12));

    // Partial concat over first (2, 1) submesh.
    EXPECT_THAT(
        aggregate_tensor(
            sharded_tensor,
            *create_mesh_composer(
                *mesh_device_,
                MeshComposerConfig{
                    .dims = {-1, 0},
                    .mesh_shape_override = MeshShape(2, 1),
                }))
            .to_vector<float>(),
        ElementsAre(0, 1, 2, 10, 11, 12));
}

class TensorDistribution2x4Test2D : public TensorDistribution2x4Test,
                                    public ::testing::WithParamInterface<MeshShape> {};

TEST_P(TensorDistribution2x4Test2D, FullyReplicated) {
    const auto num_rows = GetParam()[0];
    const auto num_cols = GetParam()[1];
    const auto num_devices = num_rows * num_cols;

    std::vector<float> test_data(num_devices);
    std::iota(test_data.begin(), test_data.end(), 0.0);

    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_rows, num_cols, 1}, DataType::FLOAT32));

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(num_rows, num_cols),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor replicated across 2D (with override) should be same as mesh mapper config
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

    EXPECT_EQ(count_unique_buffers(sharded_tensor), 1);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), num_devices);

    for (const auto& device_tensor : device_tensors) {
        EXPECT_THAT(device_tensor.to_vector<float>(), Pointwise(FloatEq(), test_data));
    }
}

TEST_P(TensorDistribution2x4Test2D, ReplicateDim) {
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

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(num_rows, num_cols),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded/replicated across 2D (with override) should be same as mesh mapper config
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

    EXPECT_EQ(count_unique_buffers(sharded_tensor), num_rows);

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

TEST_P(TensorDistribution2x4Test2D, ShardDims) {
    const auto num_rows = GetParam()[0];
    const auto num_cols = GetParam()[1];
    const int num_devices = num_rows * num_cols;

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{1, num_rows, num_cols, 3}, DataType::FLOAT32));

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Shard{2}},
        .mesh_shape_override = MeshShape(num_rows, num_cols),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded across 2D (with override) should be same as mesh mapper config
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

    EXPECT_EQ(count_unique_buffers(sharded_tensor), num_rows * num_cols);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    EXPECT_EQ(device_tensors.size(), num_devices);
    for (int i = 0; i < device_tensors.size(); i++) {
        EXPECT_THAT(device_tensors[i].to_vector<float>(), ElementsAre(i * 1.F, i * 2.F, i * 3.F));
    }

    auto composer = create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 2},
            .mesh_shape_override = MeshShape(num_rows, num_cols),
        });
    Tensor concatenated_tensor = aggregate_tensor(sharded_tensor, *composer);

    Tensor expected_tensor =
        Tensor::from_vector(test_data, get_tensor_spec(ttnn::Shape{num_rows, 1, num_cols, 3}, DataType::FLOAT32));
    EXPECT_TRUE(ttnn::allclose<float>(concatenated_tensor, expected_tensor));
}

INSTANTIATE_TEST_SUITE_P(
    TensorDistribution2x4Test2D,
    TensorDistribution2x4Test2D,
    ::testing::Values(MeshShape(1, 1), MeshShape(2, 2), MeshShape(2, 4), MeshShape(1, 3)));

TEST_F(TensorDistribution2x4Test, NdMapperInvalidShape) {
    EXPECT_ANY_THROW(create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}},
            .mesh_shape_override = MeshShape{1, 9},
        }));
}

TEST_F(TensorDistribution2x4Test, NdMapperUnevenSharding) {
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

TEST_F(TensorDistribution2x4Test, NdMapperInvalidPlacements) {
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

TEST_F(TensorDistribution2x4Test, NdComposerInvalidShape) {
    EXPECT_ANY_THROW(create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 1, 2},
            .mesh_shape_override = MeshShape{1, 9},
        }));
}

TEST_F(TensorDistribution2x4Test, NdComposerInvalidDims) {
    EXPECT_ANY_THROW(create_mesh_composer(
        *mesh_device_,
        MeshComposerConfig{
            .dims = {0, 1, 2, 3},
        }));
}

TEST_F(TensorDistribution2x4Test, NdMapperShard3D) {
    constexpr size_t kNumRows = 2;
    constexpr size_t kNumCols = 4;
    constexpr size_t kInnerDim = 7;
    constexpr size_t kOuterDim = 4;
    ASSERT_EQ(mesh_device_->shape(), MeshShape(kNumRows, kNumCols));

    std::vector<float> test_data;
    test_data.reserve(4 * kNumRows * kNumCols * 7);
    for (int i = 0; i < 4 * kNumRows * kNumCols * 7; ++i) {
        test_data.push_back(static_cast<float>(i));
    }
    Tensor input_tensor = Tensor::from_vector(
        test_data, get_tensor_spec(ttnn::Shape{kOuterDim, kNumRows, kNumCols, kInnerDim}, DataType::FLOAT32));

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Shard{2}, MeshMapperConfig::Shard{1}},
        .mesh_shape_override = MeshShape(2, 2, 2),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded across 3D (with override) should be same as mesh mapper config
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

    EXPECT_EQ(count_unique_buffers(sharded_tensor), 2 * 2);

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
            .mesh_shape_override = MeshShape(2, 2, 2),
        });
    Tensor aggregated_tensor = aggregate_tensor(sharded_tensor, *composer);
    EXPECT_EQ(aggregated_tensor.logical_shape(), ttnn::Shape({2 * kOuterDim, kNumRows, kNumCols, kInnerDim}));
    EXPECT_THAT(aggregated_tensor.to_vector<float>(), Pointwise(FloatEq(), expected_data));
}

TEST_F(TensorDistribution2x4Test, BorrowedDistributedTensors) {
    const int num_devices = mesh_device_->num_devices();
    ASSERT_EQ(num_devices, 8);

    // Returns true when every shard's underlying data pointer falls inside [src, src + count).
    auto check_borrow = []<typename T>(const Tensor& distributed, const T* src, size_t count) {
        auto shards = get_device_tensors(distributed);
        for (const auto& shard : shards) {
            auto buf = tt::tt_metal::host_buffer::get_host_buffer(shard);
            auto span = buf.view_as<T>();
            if (span.data() < src || span.data() + span.size() > src + count) {
                return false;
            }
        }
        return true;
    };

    // Verifies that each shard contains the expected iota-contiguous float data.
    auto verify_iota_shards = []<typename T = float>(const Tensor& distributed, int num_devices, int elems_per_shard) {
        auto shards = get_device_tensors(distributed);
        ASSERT_EQ(static_cast<int>(shards.size()), num_devices);
        for (int i = 0; i < num_devices; i++) {
            auto v = shards[i].to_vector<T>();
            ASSERT_EQ(static_cast<int>(v.size()), elems_per_shard);
            for (int j = 0; j < elems_per_shard; j++) {
                if constexpr (std::is_same_v<T, float>) {
                    EXPECT_FLOAT_EQ(v[j], static_cast<T>(i * elems_per_shard + j));
                } else {
                    EXPECT_EQ(v[j], static_cast<T>(i * elems_per_shard + j));
                }
            }
        }
    };

    const TensorLayout tensor_layout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{});

    // ========================================================================
    // Positive cases — can_borrow should be TRUE.
    //
    // Requirements for borrowing:
    //   1. Single shard dimension
    //   2. buffer_pin != nullptr
    //   3. ROW_MAJOR layout + no padding (logical_matches_physical)
    //   4. Buffer element type matches shard spec dtype
    //   5. All dimensions preceding the shard dimension have size 1
    // ========================================================================

    constexpr size_t elements_per_shard = 3;
    // Shard dim 0: shape {8, 1, 3, 1} — no preceding dims to check.
    {
        SCOPED_TRACE("Positive: shard dim 0");
        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 0);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{static_cast<uint32_t>(num_devices), 1, elements_per_shard, 1},
            pin,
            tensor_layout,
            *mapper);

        EXPECT_TRUE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        verify_iota_shards(dist, num_devices, elements_per_shard);
    }

    // Shard dim 1: shape {1, 8, 3, 1} — dim 0 = 1.
    {
        SCOPED_TRACE("Positive: shard dim 1");
        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{1, static_cast<uint32_t>(num_devices), elements_per_shard, 1},
            pin,
            tensor_layout,
            *mapper);

        EXPECT_TRUE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        verify_iota_shards(dist, num_devices, elements_per_shard);
    }

    // Shard dim 2: shape {1, 1, 24, 1} — dims 0, 1 = 1.
    {
        SCOPED_TRACE("Positive: shard dim 2");
        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 2);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{1, 1, static_cast<uint32_t>(num_devices * elements_per_shard), 1},
            pin,
            tensor_layout,
            *mapper);

        EXPECT_TRUE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        verify_iota_shards(dist, num_devices, elements_per_shard);
    }

    // Shard dim 3 (last): shape {1, 1, 1, 24} — dims 0, 1, 2 = 1.
    {
        SCOPED_TRACE("Positive: shard dim 3");

        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 3);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{1, 1, 1, static_cast<uint32_t>(num_devices * elements_per_shard)},
            pin,
            tensor_layout,
            *mapper);

        EXPECT_TRUE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        verify_iota_shards(dist, num_devices, elements_per_shard);
    }

    // uint32_t dtype: shape {1, 8, 3, 1}, shard dim 1.
    // Verifies that the dtype matching condition works for non-float types.
    {
        SCOPED_TRACE("Positive: uint32_t dtype");
        auto data = std::make_shared<std::vector<uint32_t>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0u);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<uint32_t> span(data->data(), data->size());
        const TensorLayout rm_uint32_tensor_layout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{});

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{1, static_cast<uint32_t>(num_devices), elements_per_shard, 1},
            pin,
            rm_uint32_tensor_layout,
            *mapper);

        EXPECT_TRUE(check_borrow.operator()<uint32_t>(dist, data->data(), data->size()));
        verify_iota_shards.operator()<uint32_t>(dist, num_devices, elements_per_shard);
    }

    // ========================================================================
    // Negative cases — can_borrow should be FALSE.
    // Each test violates exactly one of the borrowing requirements.
    // ========================================================================

    // Multi-dim sharding: two Shard placements → shard_dims.size() == 2 (violates req 1).
    {
        SCOPED_TRACE("Negative: multi-dim sharding");
        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = create_mesh_mapper(
            *mesh_device_,
            MeshMapperConfig{
                .placements = {MeshMapperConfig::Shard{1}, MeshMapperConfig::Shard{2}},
                .mesh_shape_override = MeshShape(2, 4),
            });
        Tensor dist =
            create_distributed_tensor(span, ttnn::Shape{1, 2, 4, elements_per_shard}, pin, tensor_layout, *mapper);

        EXPECT_FALSE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        auto shards = get_device_tensors(dist);
        ASSERT_EQ(static_cast<int>(shards.size()), num_devices);
        for (const auto& shard : shards) {
            EXPECT_EQ(shard.to_vector<float>().size(), elements_per_shard);
        }
    }

    // Null buffer_pin: const-span overload passes MemoryPin() internally (violates req 2).
    {
        SCOPED_TRACE("Negative: null buffer_pin (const span)");
        auto data = std::make_shared<std::vector<float>>(num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::stl::Span<const float> const_span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
        Tensor dist = create_distributed_tensor(
            const_span,
            ttnn::Shape{1, static_cast<uint32_t>(num_devices), elements_per_shard, 1},
            tensor_layout,
            *mapper);

        EXPECT_FALSE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        verify_iota_shards(dist, num_devices, elements_per_shard);
    }

    // TILE layout: logical_matches_physical returns false for non-ROW_MAJOR (violates req 3).
    {
        SCOPED_TRACE("Negative: TILE layout");
        constexpr int kTileHW = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        auto data = std::make_shared<std::vector<float>>(num_devices * kTileHW);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());
        const TensorLayout tile_float_tensor_layout(DataType::FLOAT32, Layout::TILE, MemoryConfig{});

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{1, static_cast<uint32_t>(num_devices), tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            pin,
            tile_float_tensor_layout,
            *mapper);

        EXPECT_FALSE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        auto shards = get_device_tensors(dist);
        ASSERT_EQ(static_cast<int>(shards.size()), num_devices);
        for (int i = 0; i < num_devices; i++) {
            auto v = shards[i].to_vector<float>();
            ASSERT_EQ(v.size(), static_cast<size_t>(kTileHW));
            EXPECT_FLOAT_EQ(v[0], static_cast<float>(i * kTileHW));
            EXPECT_FLOAT_EQ(v[kTileHW - 1], static_cast<float>((i + 1) * kTileHW - 1));
        }
    }

    // Preceding dim > 1: shape {2, 8, 3, 1}, shard dim 1 — dim 0 = 2 (violates req 5).
    // Shard data is non-contiguous in the source buffer so borrowing is not possible.
    {
        SCOPED_TRACE("Negative: preceding dim > 1");
        auto data = std::make_shared<std::vector<float>>(2 * num_devices * elements_per_shard);
        std::iota(data->begin(), data->end(), 0.0f);
        tt::tt_metal::MemoryPin pin(data);
        tt::stl::Span<float> span(data->data(), data->size());

        auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
        Tensor dist = create_distributed_tensor(
            span,
            ttnn::Shape{2, static_cast<uint32_t>(num_devices), elements_per_shard, 1},
            pin,
            tensor_layout,
            *mapper);

        EXPECT_FALSE(check_borrow.operator()<float>(dist, data->data(), data->size()));
        auto shards = get_device_tensors(dist);
        ASSERT_EQ(static_cast<int>(shards.size()), num_devices);
        for (const auto& shard : shards) {
            EXPECT_EQ(shard.to_vector<float>().size(), 2 * elements_per_shard);
        }
    }
}

}  // namespace ttnn::distributed::test
