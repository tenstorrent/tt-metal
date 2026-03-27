// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Launcher process for the PipelineManager device loopback.
// Creates H2D and D2H sockets, exports their descriptors, compiles and launches
// the loopback kernel on a Tensix core. The kernel runs until it receives a
// sentinel page (user_id == -1).
//
// The connector process (running PipelineManager + SocketPipeline) connects
// to the exported socket IDs and drives traffic.
//
// Usage:
//   pipeline_launcher --h2d-socket-id <id> --d2h-socket-id <id> [--fifo-size <bytes>]

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>

namespace {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

static constexpr uint32_t PAGE_SIZE_BYTES = 64;
static constexpr uint32_t DEFAULT_FIFO_SIZE = 1024;

std::string get_arg(const std::vector<std::string>& args, const std::string& key, const std::string& default_val = "") {
    for (size_t i = 0; i + 1 < args.size(); i++) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    if (!default_val.empty()) {
        return default_val;
    }
    throw std::runtime_error("Missing required argument: " + key);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> args;
        for (int i = 1; i < argc; i++) {
            args.emplace_back(argv[i]);
        }

        const std::string h2d_socket_id = get_arg(args, "--h2d-socket-id");
        const std::string d2h_socket_id = get_arg(args, "--d2h-socket-id");
        const uint32_t fifo_size =
            static_cast<uint32_t>(std::stoul(get_arg(args, "--fifo-size", std::to_string(DEFAULT_FIFO_SIZE))));

        std::cout << "Creating MeshDevice..." << std::endl;
        auto mesh_device = MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}});

        const MeshCoreCoord socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)};

        std::cout << "Creating H2D socket (id=" << h2d_socket_id << ")..." << std::endl;
        auto h2d_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
        h2d_socket.export_descriptor(h2d_socket_id);

        std::cout << "Creating D2H socket (id=" << d2h_socket_id << ")..." << std::endl;
        auto d2h_socket = D2HSocket(mesh_device, socket_core, fifo_size);
        d2h_socket.export_descriptor(d2h_socket_id);

        std::cout << "Compiling and launching loopback kernel..." << std::endl;
        auto program = CreateProgram();
        CreateKernel(
            program,
            "models/demos/deepseek_v3_b1/pipeline_manager/kernels/pipeline_loopback.cpp",
            socket_core.core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(h2d_socket.get_config_buffer_address()),
                    static_cast<uint32_t>(d2h_socket.get_config_buffer_address()),
                    PAGE_SIZE_BYTES,
                }});

        auto mesh_workload = MeshWorkload();
        mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        std::cout << "Kernel launched. Waiting for completion (connector sends sentinel to stop)..." << std::endl;
        Finish(mesh_device->mesh_command_queue());

        std::cout << "Kernel completed. Launcher exiting." << std::endl;
        mesh_device->close();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Launcher error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
