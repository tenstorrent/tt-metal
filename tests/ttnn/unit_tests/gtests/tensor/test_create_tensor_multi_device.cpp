// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::distributed::test {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::StorageType;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::distributed::H2DMode;
using ::tt::tt_metal::distributed::H2DSocket;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoreCoord;

using MultiDeviceTensorCreationTest = GenericMeshDeviceFixture;

TEST_F(MultiDeviceTensorCreationTest, Empty) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    const Tensor mesh_replicated_tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, EmptyLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    const Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);

    const Tensor mesh_replicated_tensor = ttnn::empty_like(
        tensor,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        *mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, Full) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    const Tensor mesh_replicated_tensor = ttnn::full(
        ttnn::Shape({32, 32}),
        /*fill_value=*/42,
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        std::ref(*mesh_device),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), ttnn::Shape({32, 32}));
    EXPECT_EQ(mesh_replicated_tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(mesh_replicated_tensor.layout(), Layout::ROW_MAJOR);

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_F(MultiDeviceTensorCreationTest, FullLike) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
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
        std::ref(*mesh_device));

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), tensor.logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.padded_shape(), tensor.padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());

    auto device_tensors = get_device_tensors(mesh_replicated_tensor);
    EXPECT_THAT(device_tensors, SizeIs(mesh_device->num_devices()));
    for (const auto& device_tensor : device_tensors) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(32 * 32));
        EXPECT_THAT(values, Each(Eq(42.0f)));
    }
}

TEST_F(MultiDeviceTensorCreationTest, FullLikeWithOptTensor) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    ASSERT_FALSE(mesh_device->get_devices().empty());

    Tensor tensor = ttnn::empty(
        ttnn::Shape({32, 32}),
        DataType::FLOAT32,
        Layout::ROW_MAJOR,
        mesh_device,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});

    EXPECT_EQ(tensor.storage_type(), StorageType::DEVICE);

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

    EXPECT_EQ(mesh_replicated_tensor.logical_shape(), tensor.logical_shape());
    EXPECT_EQ(mesh_replicated_tensor.padded_shape(), tensor.padded_shape());
    EXPECT_EQ(mesh_replicated_tensor.dtype(), tensor.dtype());
    EXPECT_EQ(mesh_replicated_tensor.layout(), tensor.layout());
    EXPECT_THAT(get_device_tensors(mesh_replicated_tensor), SizeIs(mesh_device->num_devices()));
}

TEST_F(MultiDeviceTensorCreationTest, Arange) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    Tensor tensor = ttnn::arange(
        /*start=*/0,
        /*stop=*/1024,
        /*step=*/1,
        ttnn::DataType::FLOAT32,
        std::ref(*mesh_device));

    EXPECT_EQ(tensor.logical_shape(), ttnn::Shape({1024}));
    EXPECT_THAT(get_device_tensors(tensor), SizeIs(mesh_device->num_devices()));

    std::vector<float> expected(1024);
    std::iota(expected.begin(), expected.end(), 0.0f);
    for (const auto& device_tensor : get_device_tensors(tensor)) {
        auto values = device_tensor.to_vector<float>();
        EXPECT_THAT(values, SizeIs(1024));
        EXPECT_THAT(values, Pointwise(FloatEq(), expected));
    }
}

TEST_F(MultiDeviceTensorCreationTest, CopyTensorOverH2DSocket_Uint32_RowMajor_Dram) {
    MeshDevice* mesh_device = this->mesh_device_.get();

    const ttnn::Shape logical_shape({1, 1, 1, 640});
    auto tensor_layout = TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    auto spec = TensorSpec(logical_shape, tensor_layout);

    std::vector<uint32_t> src(logical_shape.volume());
    std::iota(src.begin(), src.end(), 0u);
    Tensor host_tensor = Tensor::from_vector<uint32_t>(src, spec);

    Tensor device_tensor = tt::tt_metal::create_device_tensor(spec, mesh_device);

    ASSERT_NE(device_tensor.buffer(), nullptr);
    ASSERT_EQ(device_tensor.dtype(), DataType::UINT32);
    ASSERT_EQ(device_tensor.layout(), Layout::ROW_MAJOR);
    ASSERT_EQ(device_tensor.memory_config().buffer_type(), BufferType::DRAM);

    const uint32_t xfer_bytes = device_tensor.buffer()->num_pages() * device_tensor.buffer()->page_size();

    std::cout << "Transfer Bytes in test: " << xfer_bytes << std::endl;
    ASSERT_EQ(xfer_bytes, src.size() * sizeof(uint32_t));

    const MeshCoreCoord recv_core{MeshCoordinate(0, 0), CoreCoord(0, 0)};
    H2DSocket socket(
        this->mesh_device_,
        recv_core,
        BufferType::L1,
        /*fifo_size=*/xfer_bytes,
        H2DMode::DEVICE_PULL);

    tt::tt_metal::copy_tensor_over_socket(host_tensor, device_tensor, {&socket});

    socket.barrier();

    auto readback = device_tensor.to_vector<uint32_t>();
    ASSERT_THAT(readback, SizeIs(src.size()));
    EXPECT_EQ(readback, src);
}

}  // namespace
}  // namespace ttnn::distributed::test
