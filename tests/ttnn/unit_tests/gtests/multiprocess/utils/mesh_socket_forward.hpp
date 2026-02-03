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
void socket_forward(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    const distributed::MeshSocket& send_socket,
    std::size_t num_bytes);

}  // namespace tt::tt_metal
