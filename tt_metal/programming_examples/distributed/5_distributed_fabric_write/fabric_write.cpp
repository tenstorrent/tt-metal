// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
#include "tt-metalium/experimental/fabric/fabric_edm_types.hpp"
#include "tt-metalium/experimental/fabric/fabric_types.hpp"
#include "tt-metalium/global_semaphore.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/to_string.hpp"
#include "ttnn/types.hpp"
#include <ttnn/distributed/distributed_tensor.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/device.hpp>
#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

using ttnn::distributed::MeshMapperConfig;
tt::tt_metal::distributed::MeshWorkload get_workload(
    const ttnn::Tensor& inputA, tt::tt_metal::GlobalSemaphore& semaphore);
int main() {
    int M = 1024;
    int N = 128;
    auto mesh_shape = tt::tt_metal::distributed::MeshShape({1, 2});
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    auto mesh_device =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape));
    auto& command_queue = mesh_device->mesh_command_queue();
    std::vector<float> inputA_data(M * N, 1);
    ttnn::Shape shapeA = ttnn::Shape({M, N});
    auto inputA_host_buffer = tt::tt_metal::HostBuffer(std::move(inputA_data));

    auto inputA = ttnn::Tensor(
        inputA_host_buffer, shapeA, shapeA, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);

    // Split the input tensors across the mesh devices in such a way to minimize communication during matmul.
    // Split inputA along M dimension across the mesh devices.
    auto inputA_mesh_mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        MeshMapperConfig{
            .placements = {
                MeshMapperConfig::Replicate(),
                MeshMapperConfig::Shard(0),
            }});

    inputA = ttnn::distributed::distribute_tensor(inputA, *inputA_mesh_mapper, *mesh_device);

    const auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device.get(), available_cores, 0);
    try {
        auto workload = get_workload(inputA, semaphore);
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, true);
    } catch (const std::exception& e) {
        log_error(tt::LogAlways, "Exception during workload execution: {}", e.what());
    }
    mesh_device->close();
}

tt::tt_metal::distributed::MeshWorkload get_workload(
    const ttnn::Tensor& inputA, tt::tt_metal::GlobalSemaphore& semaphore) {
    auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
    const uint32_t link = 0;
    auto* mesh_device = inputA.device();
    auto mesh_shape = mesh_device->shape();
    uint32_t device_id = 0;

    const tt::tt_metal::CoreCoord core({0, 0});
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        std::optional<ttnn::MeshCoordinate> forward_coord =
            ttnn::ccl::get_physical_neighbor_from_physical_coord(inputA, coord, 1, tt::tt_fabric::Topology::Linear, 1);

        std::optional<ttnn::MeshCoordinate> backward_coord =
            ttnn::ccl::get_physical_neighbor_from_physical_coord(inputA, coord, -1, tt::tt_fabric::Topology::Linear, 1);
        const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);

        log_info(
            tt::LogAlways,
            "Creating program for coord {}, {}. Forward = {}, Backward = {}",
            coord,
            target_fabric_node_id,
            forward_coord,
            backward_coord);

        auto program = tt::tt_metal::CreateProgram();
        auto reader_kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/distributed/5_distributed_fabric_write/kernels/fabric_write_kernel.cpp",
            tt::tt_metal::CoreCoord({0, 0}),
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
            });

        std::vector<uint32_t> reader_rt_args = {
            device_id,
            semaphore.address(),
        };
        reader_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto forward_device_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, reader_rt_args);
        }

        reader_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);
            const auto backward_device_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args(
                target_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, reader_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, tt::tt_metal::CoreRange{core, core}, reader_rt_args);

        auto packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        log_info(tt::LogOp, "Packet header size = {}, Runtime args = {}", packet_header_size, reader_rt_args);
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(coord), std::move(program));
        device_id++;
    }
    return mesh_workload;
}
