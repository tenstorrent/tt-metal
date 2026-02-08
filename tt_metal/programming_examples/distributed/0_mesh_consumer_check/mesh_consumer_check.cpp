// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal tt-metal consumer example: check available devices and optionally open a mesh device.
// Writes data to the device via H2D socket and reads it back via D2H socket (loopback on device).
// Build and run from repo root (see README in this directory).
// Related: https://github.com/tenstorrent/tt-metal/issues/34274

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <iostream>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

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

    const MeshCoreCoord socket_core(MeshCoordinate(0, 0), CoreCoord(0, 0));
    std::cout << "Socket core: " << socket_core.core_coord.str() << std::endl;

    const SocketMemoryConfig socket_mem_config(BufferType::L1, 1024);
    constexpr uint32_t page_size = 64;
    constexpr uint32_t data_size = 1024;
    constexpr uint32_t num_iterations = 1;
    std::cout << "Page size: " << page_size << ", Data size: " << data_size << std::endl;

    std::cerr << "[h2d_socket] Creating H2D socket...\n" << std::flush;
    auto input_socket = std::make_unique<H2DSocket>(
        mesh_device, socket_core, socket_mem_config.socket_storage_type,
        socket_mem_config.fifo_size, H2DMode::HOST_PUSH);
    input_socket->set_page_size(page_size);

    std::cerr << "[d2h_socket] Creating D2H socket...\n" << std::flush;
    auto output_socket = std::make_unique<D2HSocket>(mesh_device, socket_core, socket_mem_config.fifo_size);
    output_socket->set_page_size(page_size);

    std::cout << "Socket config addresses - H2D: " << input_socket->get_config_buffer_address()
              << ", D2H: " << output_socket->get_config_buffer_address() << std::endl;
    std::cout << "Sockets initialized" << std::endl;

    std::cout << "Creating loopback kernel (H2D -> D2H)..." << std::endl;
    auto loopback_program = CreateProgram();
    CreateKernel(
        loopback_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket->get_config_buffer_address()),
                static_cast<uint32_t>(output_socket->get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
                static_cast<uint32_t>(false),
            }});

    MeshWorkload mesh_workload;
    mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord), std::move(loopback_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    std::cout << "Loopback kernel enqueued" << std::endl;

    std::cout << "Writing tensor to device via H2D socket..." << std::endl;
    const uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 7u);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t j = 0; j < num_pages; j++) {
        input_socket->write(src_vec.data() + (j * page_size_words), 1);
    }

    std::cout << "Reading back via D2H socket..." << std::endl;
    std::vector<uint32_t> readback(data_size / sizeof(uint32_t));
    for (uint32_t j = 0; j < num_pages; j++) {
        output_socket->read(readback.data() + (j * page_size_words), 1);
    }

    input_socket->barrier();
    output_socket->barrier();
    std::cout << "Barriers done" << std::endl;

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
        std::cout << "Mismatch: data read via D2H socket does not match data sent via H2D socket." << std::endl;
        return 1;
    }
    std::cout << "OK: tensor pushed via H2D socket and read back via D2H socket; verified." << std::endl;

    Finish(mesh_device->mesh_command_queue());
    std::cout << "Finished" << std::endl;
    mesh_device->close();

    std::cout << "Done.\n";
    return 0;
}
