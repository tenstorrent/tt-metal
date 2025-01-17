
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>

#include <tt-metalium/buffer_constants.hpp>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::distributed::test {
namespace {

using ::testing::SizeIs;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::StorageType;
using ::tt::tt_metal::TensorMemoryLayout;

class MultiDeviceTensorCreationTest : public T3kMultiDeviceFixture, public ::testing::WithParamInterface<bool> {};

TEST_P(MultiDeviceTensorCreationTest, Empty) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    const Tensor mesh_replicated_tensor = ttnn::empty(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_EQ(mesh_replicated_tensor.get_workers().size(), mesh_device->num_devices());

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(mesh_replicated_tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

TEST_P(MultiDeviceTensorCreationTest, EmptyLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    const Tensor tensor = ttnn::empty(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);
    EXPECT_THAT(tensor.get_workers(), SizeIs(1));

    const Tensor mesh_replicated_tensor = ttnn::empty_like(
        tensor,
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        *mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_THAT(mesh_replicated_tensor.get_workers(), SizeIs(mesh_device->num_devices()));

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(mesh_replicated_tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

TEST_P(MultiDeviceTensorCreationTest, Full) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    const Tensor mesh_replicated_tensor = ttnn::full(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        /*fill_value=*/42,
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        std::ref(*mesh_device),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_THAT(mesh_replicated_tensor.get_workers(), SizeIs(mesh_device->num_devices()));
    EXPECT_EQ(mesh_replicated_tensor.shape(), ttnn::SimpleShape({32, 32}));
    EXPECT_EQ(mesh_replicated_tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(mesh_replicated_tensor.layout(), Layout::ROW_MAJOR);

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(mesh_replicated_tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

TEST_P(MultiDeviceTensorCreationTest, FullLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);
    EXPECT_THAT(tensor.get_workers(), SizeIs(1));

    Tensor mesh_replicated_tensor = ttnn::full_like(
        tensor,
        /*fill_value=*/42,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        std::ref(*mesh_device));

    EXPECT_EQ(mesh_replicated_tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_THAT(mesh_replicated_tensor.get_workers(), SizeIs(mesh_device->num_devices()));
    EXPECT_EQ(mesh_replicated_tensor.shape(), tensor.shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(mesh_replicated_tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

TEST_P(MultiDeviceTensorCreationTest, FullLikeWithOptTensor) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        mesh_device->get_devices().at(0),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);
    EXPECT_EQ(tensor.get_workers().size(), 1);

    Tensor opt_output = ttnn::empty(
        ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
        DataType::BFLOAT16,
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

    EXPECT_EQ(mesh_replicated_tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_THAT(mesh_replicated_tensor.get_workers(), SizeIs(mesh_device->num_devices()));
    EXPECT_EQ(mesh_replicated_tensor.shape(), tensor.shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(mesh_replicated_tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

TEST_P(MultiDeviceTensorCreationTest, Arange) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    Tensor tensor = ttnn::arange(
        /*start=*/0,
        /*end=*/1024,
        /*step=*/1,
        ttnn::DataType::BFLOAT16,
        std::ref(*mesh_device));

    EXPECT_EQ(tensor.storage_type(), StorageType::MULTI_DEVICE);
    EXPECT_EQ(tensor.get_workers().size(), mesh_device->num_devices());
    EXPECT_EQ(tensor.shape(), ttnn::SimpleShape({1, 1, 1, 1024}));

    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(tensor);
    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

INSTANTIATE_TEST_SUITE_P(AllTests, MultiDeviceTensorCreationTest, ::testing::Bool());

}  // namespace
}  // namespace ttnn::distributed::test
