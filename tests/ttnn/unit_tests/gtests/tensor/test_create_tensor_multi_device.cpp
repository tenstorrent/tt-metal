
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "buffers/buffer_constants.hpp"
#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::distributed::test {
namespace {

using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::StorageType;
using ::tt::tt_metal::TensorMemoryLayout;

class MultiDeviceTensorCreationTest : public T3kMultiDeviceFixture, public ::testing::WithParamInterface<bool> {};

TEST_P(MultiDeviceTensorCreationTest, CreateEmpty) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    mesh_device->enable_async(GetParam());

    const auto mesh_replicated_tensor = ttnn::empty(
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

INSTANTIATE_TEST_SUITE_P(AllTests, MultiDeviceTensorCreationTest, ::testing::Bool());

}  // namespace
}  // namespace ttnn::distributed::test
