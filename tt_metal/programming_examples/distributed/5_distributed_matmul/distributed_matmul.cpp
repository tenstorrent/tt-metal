// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/mesh_config.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "tt-metalium/program.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "tt_stl/small_vector.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/distributed_configs.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/to_string.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include <ttnn/tensor/tensor.hpp>

using ttnn::distributed::MeshMapperConfig;
int main() {
    int M = 1024;
    int N = 128;
    int K = 512;
    auto mesh_shape = tt::tt_metal::distributed::MeshShape({1, 2});
    auto mesh_device =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape));
    std::vector<float> inputA_data(M * K, 1);
    std::vector<float> inputB_data(K * N, 1);
    ttnn::Shape shapeA = ttnn::Shape({M, K});
    ttnn::Shape shapeB = ttnn::Shape({K, N});
    auto inputA_host_buffer = tt::tt_metal::HostBuffer(std::move(inputA_data));
    auto inputB_host_buffer = tt::tt_metal::HostBuffer(std::move(inputB_data));

    auto inputA = ttnn::Tensor(
        inputA_host_buffer, shapeA, shapeA, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);
    auto inputB = ttnn::Tensor(
        inputB_host_buffer, shapeB, shapeB, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);

    // Split the input tensors across the mesh devices in such a way to minimize communication during matmul.
    // Split inputA along M dimension across the mesh devices.
    auto inputA_mesh_mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        MeshMapperConfig{
            .placements = {
                MeshMapperConfig::Replicate(),
                MeshMapperConfig::Shard(0),
            }});

    // Split inputB along N dimension across the mesh devices
    auto inputB_mesh_mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        MeshMapperConfig{
            .placements = {
                MeshMapperConfig::Replicate(),
                MeshMapperConfig::Shard(1),
            }});
    inputA = ttnn::distributed::distribute_tensor(inputA, *inputA_mesh_mapper, *mesh_device);
    inputB = ttnn::distributed::distribute_tensor(inputB, *inputB_mesh_mapper, *mesh_device);

    log_info(
        tt::LogAlways,
        "\n Input A : {}, Memory Config : {}, Storage : {} ",
        ttnn::to_string(inputA),
        inputA.memory_config(),
        inputA.storage());
    log_info(
        tt::LogAlways,
        "\n Input B : {}, Memory Config : {}, Storage : {} ",
        ttnn::to_string(inputB),
        inputB.memory_config(),
        inputB.storage());

    auto output = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec{
            tt::tt_metal::Shape({M / 2, N}),  // Shape per device, as we shard M across devices.
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::types::DRAM_MEMORY_CONFIG)

        },
        mesh_device.get());

    log_info(
        tt::LogAlways,
        "\n Output : {}, Memory Config : {}, Storage : {} ",
        ttnn::to_string(output),
        output.memory_config(),
        output.storage());
}
