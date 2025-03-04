// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"

namespace ttnn::distributed::test {

using namespace tt::tt_metal;

Tensor create_host_multi_device_tensor(
    const Tensor& tensor, int replication_factor, const std::optional<MeshShape>& distribution_shape) {
    std::vector<OwnedBuffer> owned_buffers;
    std::vector<ttnn::TensorSpec> specs;

    for (int i = 0; i < replication_factor; i++) {
        owned_buffers.push_back(std::get<OwnedStorage>(tensor.get_storage()).buffer);
        specs.push_back(tensor.get_tensor_spec());
    }

    return Tensor{MultiDeviceHostStorage(distribution_shape, owned_buffers, specs), tensor.get_tensor_spec()};
}

TEST_F(T3kMultiDeviceFixture, TestGetTensorsFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape({32, 32}), DataType::BFLOAT16);
    const auto replicated_tensor =
        create_host_multi_device_tensor(input_tensor, /*replication_factor=*/8, std::nullopt);
    const auto device_tensors = get_tensors_from_multi_device_storage(replicated_tensor);

    EXPECT_EQ(device_tensors.size(), 8);
}

TEST_F(T3kMultiDeviceFixture, TestGetDistributionShapeFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape({32, 32}), DataType::BFLOAT16);
    const auto replicated_tensor =
        create_host_multi_device_tensor(input_tensor, /*replication_factor=*/8, MeshShape(1, 8));

    const auto shape = get_distribution_shape_from_tensor(replicated_tensor);
    ASSERT_TRUE(shape.has_value());
    EXPECT_EQ(*shape, MeshShape(1, 8));
}

}  // namespace ttnn::distributed::test
