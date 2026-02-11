// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal {

// Metal-level socket_forward operation
// Forwards data from recv_socket to send_socket
// Both sockets must be on the same device and core
// latency_measurement_address: L1 address used for credit synchronization
// num_iterations: number of forward iterations the kernel will execute
void socket_forward(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    const distributed::MeshSocket& send_socket,
    std::size_t num_bytes,
    uint32_t latency_measurement_address,
    uint32_t num_iterations);

}  // namespace tt::tt_metal
