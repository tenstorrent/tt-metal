// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Rate-mode send_async: sends data downstream for num_iterations without waiting for loopback acks.
// Timing is performed on the host side, not by the kernel.
void send_async_rate(
    distributed::MeshDevice* mesh_device,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    const distributed::MeshSocket& mesh_socket,
    uint32_t num_iterations);

// Rate-mode socket_forward: forwards data from recv_socket to send_socket for num_iterations.
// No initial handshake — synchronization relies on socket flow control.
void socket_forward_rate(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    const distributed::MeshSocket& send_socket,
    std::size_t num_bytes,
    uint32_t num_iterations);

// Rate-mode recv_async: drains data from recv_socket for num_iterations.
// Timing is performed on the host side, not by the kernel.
// When enable_correctness_check is true, validates each received page against expected data pattern.
void recv_async_rate(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshSocket& recv_socket,
    std::size_t num_bytes,
    uint32_t num_iterations,
    bool enable_correctness_check = false);

}  // namespace tt::tt_metal
