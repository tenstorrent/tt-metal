// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>

#include <iostream>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
#include "tt-metalium/experimental/fabric/fabric_edm_types.hpp"
#include "tt-metalium/experimental/fabric/fabric_types.hpp"
#include "tt-metalium/global_semaphore.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/to_string.hpp"
#include "ttnn/tensor/types.hpp"
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
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// This programming example demonstrates a simple multi-chip matrix multiplication using fabric communication.
// This example uses two devices, with one Tensix core per device, with inputs & outputs in DRAM.

// The two input matrices of shape [M, K] and [K, N] are split across two devices in a 1D mesh.
// InputA is divided sharded along the M dimension and InputB is sharded along the N dimension.
// The output matrix of shape [M, N] is also sharded along the M dimension.
// As each device has the entire N dimension of the output, InputB needs to be shared between the devices via fabric.

// Each device has a semaphore that is incremented by the other device via fabric communication.
// The kernel waits for the semaphore to be incremented before proceeding.
// This programming example is for a N300 device with two devices in a 1D mesh.
// If you have more than one device, run export TT_VISIBLE_DEVICES="0" to make sure only one device is visible.

using ttnn::distributed::MeshMapperConfig;
tt::tt_metal::distributed::MeshWorkload get_workload(
    const ttnn::Tensor& inputA,
    const ttnn::Tensor& inputB,
    const ttnn::Tensor& output,
    tt::tt_metal::GlobalSemaphore& semaphore);

int main() {
    // Input Tensor Shape
    int M = 1024;
    int N = 128;
    int K = 512;

    auto mesh_shape = tt::tt_metal::distributed::MeshShape({1, 2});

    // Enable fabric for kernels in a 1D Mesh. Fabric is disabled by default.
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    auto mesh_device =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape));
    auto& command_queue = mesh_device->mesh_command_queue();

    // Create Input Tensor
    std::vector<float> inputA_data(M * K, 1);
    std::vector<float> inputB_data(K * N, 1);
    for (int index_k = 0; index_k < K; index_k++) {
        for (int index_n = 0; index_n < N; index_n++) {
            inputB_data[index_k * N + index_n] = static_cast<float>(index_n) / 512;
        }
    }

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

    inputA = ttnn::distributed::distribute_tensor(inputA, *inputA_mesh_mapper, *mesh_device);
    inputA = ttnn::to_layout(inputA, tt::tt_metal::Layout::TILE);

    // Split inputB along N dimension across the mesh devices.
    auto inputB_mesh_mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        MeshMapperConfig{
            .placements = {
                MeshMapperConfig::Replicate(),
                MeshMapperConfig::Shard(1),
            }});
    inputB = ttnn::distributed::distribute_tensor(inputB, *inputB_mesh_mapper, *mesh_device);
    inputB = ttnn::to_layout(inputB, tt::tt_metal::Layout::TILE);

    // output tensor is split along the M dimension.
    auto output = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec{
            tt::tt_metal::Shape({M / 2, N}),  // Shape per device, as we shard M across devices.
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::types::DRAM_MEMORY_CONFIG)

        },
        mesh_device.get());

    log_info(tt::LogOp, "\n Input A= {}\n Input B= {}\n", ttnn::to_string(inputA), ttnn::to_string(inputB));

    const auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));

    // Create a semaphore that is shared across all devices in the mesh.
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device.get(), available_cores, 0);

    try {
        auto workload = get_workload(inputA, inputB, output, semaphore);
        tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, true);
    } catch (const std::exception& e) {
        log_error(tt::LogAlways, "Exception during workload execution: {}", e.what());
    }

    // Aggregate the output tensor from the mesh devices back to a single tensor.
    // The output tensor is sharded along M dimension across the mesh.
    // We aggregate it back to a single tensor by combining along the M dimension.
    output = ttnn::distributed::aggregate_tensor(
        output,
        ttnn::distributed::MeshToTensor::create(
            *mesh_device,
            tt::tt_metal::distributed::MeshComposerConfig{
                .dims = {-1, 0}
                // Each value in dims corresponds to a dimension of mesh shape.
                // Mesh Shape is [1, 2] for N300.
                // For the first dimension of the mesh, we don't aggregate, so -1.
                // For the second dimension of the mesh, we aggregate along dimension 0 (M dimension of the tensor).
            }));

    log_info(tt::LogAlways, "\n Output = {}", ttnn::to_string(output));
    mesh_device->close();
}

tt::tt_metal::distributed::MeshWorkload get_workload(
    const ttnn::Tensor& inputA,
    const ttnn::Tensor& inputB,
    const ttnn::Tensor& output,
    tt::tt_metal::GlobalSemaphore& semaphore) {
    // Per device tensor shape
    const ttnn::Shape& inputA_shape = inputA.logical_shape();
    const ttnn::Shape& inputB_shape = inputB.logical_shape();
    const ttnn::Shape& output_shape = output.logical_shape();

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    log_info(
        tt::LogAlways,
        " InputA shape: {}, InputB shape: {}, Output shape: {}",
        inputA_shape,
        inputB_shape,
        output_shape);

    int M = inputA_shape[0];
    int K = inputA_shape[1];
    int N = inputB_shape[1];

    // output_N is N * 2 because inputB is sharded along N dimension across two devices.
    int output_N = output_shape[1];

    int M_tiles = M / tt::constants::TILE_HEIGHT;
    int K_tiles = K / tt::constants::TILE_HEIGHT;
    int N_tiles = N / tt::constants::TILE_WIDTH;
    int outN_tiles = output_N / tt::constants::TILE_WIDTH;

    uint32_t tile_size_bytes = tt::tile_size(datatype_to_dataformat_converter(inputA.dtype()));

    // The current code sends the entire tile in one fabric packet.
    // So we need to make sure that tile size is less than fabric max payload size.
    TT_FATAL(
        tile_size_bytes <= fabric_max_packet_size,
        "Tile size {} bytes exceeds fabric max payload size {} bytes",
        tile_size_bytes,
        fabric_max_packet_size);
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(inputA.dtype());

    auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
    const uint32_t link = 0;
    auto* mesh_device = inputA.device();
    auto mesh_shape = mesh_device->shape();
    uint32_t device_id = 0;

    const uint32_t inputA_cb_index = 0;
    const uint32_t inputB_cb_index = 1;
    const uint32_t output_cb_index = 2;
    tt::tt_metal::CircularBufferConfig inputA_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * K_tiles * tile_size_bytes, {{inputA_cb_index, data_format}})
            .set_page_size(inputA_cb_index, tile_size_bytes);

    tt::tt_metal::CircularBufferConfig inputB_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * K_tiles * tile_size_bytes, {{inputB_cb_index, data_format}})
            .set_page_size(inputB_cb_index, tile_size_bytes);

    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * outN_tiles * tile_size_bytes, {{output_cb_index, data_format}})
            .set_page_size(output_cb_index, tile_size_bytes);
    const tt::tt_metal::CoreCoord core({0, 0});

    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        std::optional<ttnn::MeshCoordinate> forward_coord =
            ttnn::ccl::get_physical_neighbor_from_physical_coord(inputA, coord, 1, tt::tt_fabric::Topology::Linear, 1);

        std::optional<ttnn::MeshCoordinate> backward_coord =
            ttnn::ccl::get_physical_neighbor_from_physical_coord(inputA, coord, -1, tt::tt_fabric::Topology::Linear, 1);
        const auto target_fabric_node_id = mesh_device->get_fabric_node_id(coord);

        log_debug(
            tt::LogAlways,
            "Creating program for coord {}, {}. Forward = {}, Backward = {}",
            coord,
            target_fabric_node_id,
            forward_coord,
            backward_coord);

        auto program = tt::tt_metal::CreateProgram();
        tt::tt_metal::CreateCircularBuffer(program, core, inputA_cb_config);
        tt::tt_metal::CreateCircularBuffer(program, core, inputB_cb_config);
        tt::tt_metal::CreateCircularBuffer(program, core, output_cb_config);

        // Create the data movement kernels and the compute kernel
        std::vector<uint32_t> reader_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(inputA.buffer()).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(inputB.buffer()).append_to(reader_compile_time_args);

        auto reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/distributed/6_distributed_fabric_matmul/kernels/"
            "fabric_matmul_reader_kernel.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
        auto writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/distributed/6_distributed_fabric_matmul/kernels/"
            "fabric_matmul_writer_kernel.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_compile_time_args = {M_tiles, K_tiles, outN_tiles};
        tt::tt_metal::CreateKernel(
            program,
            "tt_metal/programming_examples/distributed/6_distributed_fabric_matmul/kernels/mm.cpp",
            core,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

        std::vector<uint32_t> reader_rt_args = {
            device_id,
            semaphore.address(),
            inputA.buffer()->address(),
            inputB.buffer()->address(),
            M_tiles,
            K_tiles,
            N_tiles,
        };

        std::vector<uint32_t> writer_rt_args = {
            output.buffer()->address(),
            M_tiles,
            outN_tiles,
        };

        // Used by  FabricConnectionManager::build_from_args to make the connection.
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
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, tt::tt_metal::CoreRange{core, core}, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, tt::tt_metal::CoreRange{core, core}, writer_rt_args);

        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(coord), std::move(program));
        device_id++;
    }
    return mesh_workload;
}
