// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"

namespace ttnn::multi_device::test {

using namespace tt::tt_metal;

Tensor create_host_multi_device_tensor(const Tensor& tensor, const ReplicateTensor& strategy) {
    std::vector<OwnedBuffer> owned_buffers;
    std::vector<tt::tt_metal::Shape> shapes;

    for (int i = 0; i < strategy.replication_factor; i++) {
        owned_buffers.push_back(std::get<OwnedStorage>(tensor.get_storage()).buffer);
        shapes.push_back(tensor.get_legacy_shape());
    }

    return Tensor{
        MultiDeviceHostStorage(strategy, owned_buffers, shapes),
        tensor.get_legacy_shape(),
        tensor.get_dtype(),
        tensor.get_layout()};
}

TEST_F(T3kMultiDeviceFixture, TestGetTensorsFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape(std::array<uint32_t, 2>{32, 32}), ttnn::bfloat16);
    const auto replicated_tensor = create_host_multi_device_tensor(input_tensor, ReplicateTensor(8));
    const auto device_tensors = get_tensors_from_multi_device_storage(replicated_tensor);

    EXPECT_EQ(device_tensors.size(), 8);
}

TEST_F(T3kMultiDeviceFixture, TestGetDistributedTensorConfigFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape(std::array<uint32_t, 2>{32, 32}), ttnn::bfloat16);
    const auto replicated_tensor = create_host_multi_device_tensor(input_tensor, ReplicateTensor(8));
    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(replicated_tensor);

    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

}  // namespace ttnn::multi_device::test
