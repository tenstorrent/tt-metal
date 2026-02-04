// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal tt-metal consumer example: check available devices and optionally open a mesh device.
// Build and run from repo root (see README in this directory).

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <iostream>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    size_t num_available = GetNumAvailableDevices();
    size_t num_pcie = GetNumPCIeDevices();
    std::cout << "Available devices: " << num_available << ", PCIe devices: " << num_pcie << std::endl;

    if (num_available == 0) {
        std::cout << "No devices found. Check drivers and TT_VISIBLE_DEVICES.\n";
        return 0;
    }

    std::cout << "Opening 1x1 mesh device (consumer)...\n";
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
    std::cout << "Mesh device opened. Shape: " << mesh_device->num_rows() << "x" << mesh_device->num_cols()
              << std::endl;

    const MeshCoreCoord recv_core(MeshCoordinate(0, 0), CoreCoord(0, 0));
    std::cout << "Recv core: " << recv_core.core_coord.str() << std::endl;

    const SocketMemoryConfig socket_mem_config(BufferType::L1, 1024);
    constexpr uint32_t page_size = 64;
    constexpr uint32_t data_size = 1024;
    std::cout << "Page size: " << page_size << ", Data size: " << data_size << std::endl;
    std::cerr << "[h2d_socket] Creating H2D receiver socket...\n" << std::flush;

    auto socket = std::make_unique<H2DSocket>(
        mesh_device, recv_core, socket_mem_config.socket_storage_type,
        socket_mem_config.fifo_size, H2DMode::HOST_PUSH);
    socket->set_page_size(page_size);
    std::cout << "Socket page size: " << socket->get_page_size() << std::endl;
    std::cout << "Socket config buffer address: " << socket->get_config_buffer_address() << std::endl;
    std::cout << "Socket initialized" << std::endl;

    std::cout << "Creating receiver kernel..." << std::endl;
    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto recv_data_shard_params = ShardSpecBuffer(
        CoreRangeSet(recv_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto recv_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device.get());

    constexpr uint32_t num_iterations = 1;
    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(socket->get_config_buffer_address()),
                static_cast<uint32_t>(recv_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
            }});

    MeshWorkload mesh_workload;
    mesh_workload.add_program(MeshCoordinateRange(recv_core.device_coord), std::move(recv_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    std::cout << "Receiver kernel enqueued" << std::endl;

    std::cout << "Writing tensor to device..." << std::endl;
    const uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 7u);

    socket->write(src_vec.data(), num_pages);
    socket->barrier();
    std::cout << "Barrier done" << std::endl;

    // Readback via get_data_buffer() (same as MeshSocket::get_data_buffer() on receiver).
    std::cout << "Reading back tensor..." << std::endl;
    std::vector<uint32_t> readback(data_size / sizeof(uint32_t));
    auto recv_core_virtual = mesh_device->worker_core_from_logical_core(recv_core.core_coord);
    MetalContext::instance().get_cluster().read_core(
        readback.data(),
        data_size,
        tt_cxy_pair(mesh_device->get_device(recv_core.device_coord)->id(), recv_core_virtual),
        recv_buffer->address());
    std::cout << "Readback done: " << readback.size() << " elements." << std::endl;
    // print the first 10 elements of readback and src_vec
    std::cout << "Readback: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << readback[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Src vec: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << src_vec[i] << " ";
    }
    std::cout << std::endl;
    if (readback != src_vec) {
        std::cout << "Mismatch: tensor read back from device does not match sent data." << std::endl;
        return 1;
    }
    std::cout << "OK: tensor pushed via high-level H2D socket API and verified by readback." << std::endl;

    Finish(mesh_device->mesh_command_queue());
    std::cout << "Finished" << std::endl;
    mesh_device->close();
    
    std::cout << "Done.\n";
    return 0;
}
