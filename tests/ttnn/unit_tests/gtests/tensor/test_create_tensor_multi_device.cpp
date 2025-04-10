// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::distributed::test {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::StorageType;
using ::tt::tt_metal::TensorMemoryLayout;

class MultiDeviceTensorCreationTest : public GenericMeshDeviceFixture, public ::testing::WithParamInterface<bool> {};

TEST_P(MultiDeviceTensorCreationTest, Empty) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    const Tensor mesh_replicated_tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_P(MultiDeviceTensorCreationTest, EmptyLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    const Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);
    EXPECT_THAT(tensor.get_workers(), SizeIs(1));

    const Tensor mesh_replicated_tensor = ttnn::empty_like(
        tensor,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        *mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_P(MultiDeviceTensorCreationTest, Full) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    const Tensor mesh_replicated_tensor = ttnn::full(
        ttnn::Shape({32, 32}),
        /*fill_value=*/42,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        std::ref(*mesh_device),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.get_logical_shape(), ttnn::Shape({32, 32}));
    EXPECT_EQ(mesh_replicated_tensor.get_dtype(), DataType::FLOAT32);
    EXPECT_EQ(mesh_replicated_tensor.get_layout(), Layout::ROW_MAJOR);

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_P(MultiDeviceTensorCreationTest, FullLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    Tensor mesh_replicated_tensor = ttnn::full_like(
        tensor,
        /*fill_value=*/42,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        std::ref(*mesh_device));

    EXPECT_EQ(mesh_replicated_tensor.get_logical_shape(), tensor.get_logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.get_padded_shape(), tensor.get_padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.get_dtype(), tensor.get_dtype());
    EXPECT_EQ(mesh_replicated_tensor.get_layout(), tensor.get_layout());

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_P(MultiDeviceTensorCreationTest, FullLikeWithOptTensor) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);
    EXPECT_EQ(tensor.get_workers().size(), 1);

    Tensor opt_output = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    Tensor mesh_replicated_tensor = ttnn::full_like(
        tensor,
        /*fill_value=*/42,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        opt_output);

    EXPECT_EQ(mesh_replicated_tensor.get_logical_shape(), tensor.get_logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.get_padded_shape(), tensor.get_padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.get_dtype(), tensor.get_dtype());
    EXPECT_EQ(mesh_replicated_tensor.get_layout(), tensor.get_layout());

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_P(MultiDeviceTensorCreationTest, Arange) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    Tensor tensor = ttnn::arange(
        /*start=*/0,
        /*end=*/1024,
        /*step=*/1,
        ttnn::DataType::FLOAT32,
        std::ref(*mesh_device));

    EXPECT_EQ(tensor.get_logical_shape(), ttnn::Shape({1024}));

    EXPECT_THAT(get_device_tensors(tensor), SizeIs(mesh_device->num_devices()));

    std::vector<float> expected(1024);
    std::iota(expected.begin(), expected.end(), 0.0f);
    for (const auto& device_tensor : get_device_tensors(tensor)) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(1024));
        EXPECT_THAT(values, Pointwise(FloatEq(), expected));
    }
}

INSTANTIATE_TEST_SUITE_P(AllTests, MultiDeviceTensorCreationTest, ::testing::Bool());

}  // namespace
}  // namespace ttnn::distributed::test
