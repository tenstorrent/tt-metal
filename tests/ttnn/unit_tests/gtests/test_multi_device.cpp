// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::distributed::test {

using namespace tt::tt_metal;

Tensor create_host_multi_device_tensor(const Tensor& tensor, const ReplicateTensor& strategy) {
    std::vector<HostBuffer> owned_buffers;
    std::vector<ttnn::TensorSpec> specs;

    for (int i = 0; i < strategy.replication_factor; i++) {
        owned_buffers.push_back(std::get<HostStorage>(tensor.get_storage()).buffer);
        specs.push_back(tensor.get_tensor_spec());
    }

    return Tensor{MultiDeviceHostStorage(strategy, owned_buffers, specs), tensor.get_tensor_spec()};
}

TEST_F(GenericMeshDeviceFixture, TestGetTensorsFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape({32, 32}), DataType::BFLOAT16);
    const auto replicated_tensor =
        create_host_multi_device_tensor(input_tensor, ReplicateTensor(mesh_device_->num_devices()));
    const auto device_tensors = get_device_tensors(replicated_tensor);

    EXPECT_EQ(device_tensors.size(), 8);
}

TEST_F(GenericMeshDeviceFixture, TestGetDistributedTensorConfigFromMultiDeviceStorage) {
    MeshDevice* mesh_device = this->mesh_device_.get();
    const auto input_tensor = ttnn::ones(ttnn::Shape({32, 32}), DataType::BFLOAT16);
    const auto replicated_tensor =
        create_host_multi_device_tensor(input_tensor, ReplicateTensor(mesh_device_->num_devices()));
    const auto distributed_tensor_config = get_distributed_tensor_config_from_tensor(replicated_tensor);

    EXPECT_TRUE(std::holds_alternative<ReplicateTensor>(distributed_tensor_config));
}

}  // namespace ttnn::distributed::test
