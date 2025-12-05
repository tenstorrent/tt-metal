// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal.hpp>
#include "impl/context/metal_context.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"
#include "ttnn/api/ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/udm/tensor_builder.hpp"
#include "tt_metal/udm/block_program.hpp"
#include "tt_metal/udm/block_kernel.hpp"
#include "tt_metal/udm/block_circular_buffer.hpp"
#include "tt_metal/udm/block_utils.hpp"

namespace udm_program_tests {

using namespace ttnn;
using namespace tt::tt_metal;

class UDMProgramFixture : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 1) {
            GTEST_SKIP() << "UDM program tests require at least 1 device";
        }
    }
};

/**
 * @brief Helper to get mesh coordinates from a distributed tensor
 */
std::vector<ttnn::MeshCoordinate> get_mesh_coordinates_from_tensor(const Tensor& input_tensor) {
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& mesh_coords = tensor_topology.mesh_coords();

    log_info(tt::LogTest, "Extracted {} mesh coordinates from tensor topology", mesh_coords.size());
    for (size_t i = 0; i < mesh_coords.size(); ++i) {
        log_info(tt::LogTest, "  Coordinate {}: [{}, {}]", i, mesh_coords[i][0], mesh_coords[i][1]);
    }

    return mesh_coords;
}

/**
 * @brief Helper to get mesh shape and fabric node IDs from tensor topology
 */
std::pair<MeshShape, std::unordered_map<ttnn::MeshCoordinate, tt::tt_fabric::FabricNodeId>> get_mesh_info_from_tensor(
    const Tensor& input_tensor, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& mesh_shape = tensor_topology.distribution_shape();
    const auto& mesh_coordinates = get_mesh_coordinates_from_tensor(input_tensor);

    std::unordered_map<ttnn::MeshCoordinate, tt::tt_fabric::FabricNodeId> fabric_node_ids;
    for (const auto& coord : mesh_coordinates) {
        auto fabric_node_id = mesh_device->get_fabric_node_id(coord);
        fabric_node_ids.emplace(coord, fabric_node_id);
        log_info(
            tt::LogTest,
            "Fabric node ID for coordinate [{}, {}]: mesh_id={}, chip_id={}",
            coord[0],
            coord[1],
            *fabric_node_id.mesh_id,
            fabric_node_id.chip_id);
    }

    return {mesh_shape, fabric_node_ids};
}

/**
 * @brief Reconstruct full global tensor shape from a sharded distributed tensor
 *
 * When a tensor is sharded across devices, each device holds a portion of the data.
 * This function reconstructs what the full global tensor shape would be.
 *
 * @param input_tensor A distributed tensor where each device has a shard
 * @return The reconstructed global padded shape
 */
Shape reconstruct_tensor_shape(const Tensor& input_tensor) {
    const auto& per_device_shape = input_tensor.padded_shape();
    const auto& tensor_topology = input_tensor.tensor_topology();
    const auto& placements = tensor_topology.placements();
    const auto& mesh_shape = tensor_topology.distribution_shape();

    // Start with per-device padded shape and scale up sharded dimensions
    ttsl::SmallVector<uint32_t> global_shape(per_device_shape.cbegin(), per_device_shape.cend());

    // For each mesh dimension, check if a tensor dimension is sharded across it
    for (size_t mesh_dim = 0; mesh_dim < placements.size(); ++mesh_dim) {
        if (std::holds_alternative<ttnn::distributed::MeshMapperConfig::Shard>(placements[mesh_dim])) {
            const auto& shard = std::get<ttnn::distributed::MeshMapperConfig::Shard>(placements[mesh_dim]);
            // This tensor dimension is sharded, so scale it by the mesh size in that dimension
            global_shape[shard.dim] *= mesh_shape[mesh_dim];
        }
    }

    Shape full_shape(global_shape);
    log_info(tt::LogTest, "Per-device padded shape: {}", per_device_shape);
    log_info(tt::LogTest, "Reconstructed global padded shape: {}", full_shape);
    return full_shape;
}

/**
 * @brief Create a UDM program for the given tensor using UDM APIs
 */
tt::tt_metal::udm::BlockProgram create_program(const Tensor& input_tensor) {
    // Create TensorBuilder - the top-level entry point
    auto tensor_builder = tt::tt_metal::udm::CreateTensorBuilder(input_tensor);

    // Create BlockProgram from TensorBuilder
    auto block_program = tt::tt_metal::udm::CreateBlockProgram(tensor_builder);

    // Map tensor to global cores
    auto gcores_info = tt::tt_metal::udm::map_tensor_to_gcores(tensor_builder, input_tensor);

    // Get compile-time arguments from BlockTensorAccessor (instead of TensorAccessorArgs)
    auto compile_time_args = tensor_builder.tensor_accessor().get_compile_time_args();

    // Create block kernel across all gcores
    tt::tt_metal::udm::BlockKernelHandle kernel_id = tt::tt_metal::udm::CreateBlockKernel(
        block_program,
        "tests/ttnn/unit_tests/gtests/udm/kernels/udm_program.cpp",
        gcores_info.gcores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args,
        });

    // Set runtime args for each gcore
    uint32_t page_start_id = 0;
    for (uint32_t i = 0; i < gcores_info.num_cores; i++) {
        std::vector<uint32_t> runtime_args = {
            static_cast<uint32_t>(tensor_builder.tensor_accessor().get_buffer_address()),
            gcores_info.pages_per_gcore,
            page_start_id};

        tt::tt_metal::udm::SetBlockKernelRuntimeArgs(block_program, kernel_id, gcores_info.gcores[i], runtime_args);

        page_start_id += gcores_info.pages_per_gcore;
    }

    return block_program;
}

/**
 * @brief Run a block program on all mesh coordinates
 *
 * The program will execute on all device coordinates where the tensor is distributed.
 */
void run_program(
    const Tensor& sharded_tensor,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::udm::BlockProgram& block_program) {
    // Get the mesh shape from the tensor topology
    const auto& mesh_shape = sharded_tensor.tensor_topology().distribution_shape();

    // Create a mesh coordinate range that covers all coordinates in the mesh
    auto mesh_coord_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape);

    auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
    // BlockProgram wraps the underlying Program
    mesh_workload.add_program(mesh_coord_range, std::move(block_program.program()));

    tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);
}

/**
 * @brief Create a sharded tensor distributed across the mesh device
 */
Tensor create_sharded_tensor(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const Shape& tensor_shape,
    const Shape& shard_shape,
    const std::vector<int>& shard_dims) {
    auto mesh_shape = mesh_device->shape();

    // Create placements based on specified shard dimensions
    ttsl::SmallVector<ttnn::distributed::MeshMapperConfig::Placement> placements;
    for (size_t mesh_dim = 0; mesh_dim < mesh_shape.dims(); ++mesh_dim) {
        int tensor_dim = shard_dims[mesh_dim];
        placements.push_back(ttnn::distributed::MeshMapperConfig::Shard{tensor_dim});
    }

    // Create shard spec and distribute tensor
    auto grid = CoreRangeSet(CoreRange({0, 0}, {1, 1}));  // 2x2 = 4 cores
    auto shard_spec = NdShardSpec{
        .shard_shape = shard_shape,
        .grid = grid,
        .orientation = ShardOrientation::ROW_MAJOR,
    };

    MemoryConfig mem_config(BufferType::L1, shard_spec);
    TensorSpec tensor_spec(tensor_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), mem_config));

    uint32_t volume = 1;
    for (size_t i = 0; i < tensor_shape.rank(); ++i) {
        volume *= tensor_shape[i];
    }
    std::vector<uint16_t> src_data(volume);
    std::iota(src_data.begin(), src_data.end(), 0);

    auto host_tensor = Tensor::from_vector(src_data, tensor_spec);

    // Use create_mesh_mapper with custom placements
    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        ttnn::distributed::MeshMapperConfig{.placements = std::move(placements), .mesh_shape_override = mesh_shape});

    auto sharded_tensor = ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
    return sharded_tensor;
}

/**
 * @brief Main test function for UDM program
 */
void run_udm_program_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const Shape& tensor_shape,
    const Shape& shard_shape,
    const std::vector<int>& shard_dims) {
    auto sharded_tensor = create_sharded_tensor(mesh_device, tensor_shape, shard_shape, shard_dims);

    // Create a UDM block program using the new UDM APIs
    auto block_program = create_program(sharded_tensor);

    // Run the program across all coordinates where the tensor is distributed
    auto* tensor_mesh_device = sharded_tensor.device();
    TT_FATAL(tensor_mesh_device != nullptr, "Tensor must be on device");
    run_program(sharded_tensor, tensor_mesh_device, block_program);
}

/**
 * @brief Test page ordering for a 2D tensor sharded across multiple devices
 */
TEST_F(UDMProgramFixture, Test2DMultiDeviceShardedTensor) {
    // 2D tensor: [128, 128] sharded across mesh dimensions
    // Mesh dim 0 shards tensor dim 0, mesh dim 1 shards tensor dim 1
    Shape tensor_shape({128, 128});
    Shape shard_shape({32, 64});  // 1 tile x 2 tiles per core
    run_udm_program_test(
        mesh_device_.get(), tensor_shape, shard_shape, {0, 1}  // Mesh[0] -> Tensor[0], Mesh[1] -> Tensor[1]
    );
}

}  // namespace udm_program_tests
