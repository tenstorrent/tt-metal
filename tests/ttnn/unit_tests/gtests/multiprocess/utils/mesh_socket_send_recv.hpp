// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Metal-level send_async operation
// Sends data from input_buffer through mesh_socket
// Optionally uses recv_socket for backward communication
// Optionally uses barrier_buffer for latency measurements (if provided, latencies are written to this buffer)
void send_async(
    distributed::MeshDevice* mesh_device,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    const distributed::MeshSocket& mesh_socket,
    const std::optional<distributed::MeshSocket>& recv_socket = std::nullopt,
    const std::optional<const Buffer*>& barrier_buffer = std::nullopt);

}  // namespace tt::tt_metal
