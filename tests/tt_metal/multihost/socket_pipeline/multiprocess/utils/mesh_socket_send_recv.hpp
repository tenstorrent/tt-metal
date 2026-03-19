// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Metal-level send_async operation
// Sends data from input_buffer through mesh_socket; uses recv_socket for backward acks (e.g. loopback).
// latency_measurement_address: L1 address used for credit synchronization and latency measurements
// num_iterations: number of send/recv iterations the kernel will execute
// enable_correctness_check: when true, the sender kernel validates received data matches expected values
void send_async(
    distributed::MeshDevice* mesh_device,
    const Buffer* input_buffer,
    DataFormat input_data_format,
    const distributed::MeshSocket& mesh_socket,
    const distributed::MeshSocket& recv_socket,
    uint32_t latency_measurement_address,
    uint32_t num_iterations,
    bool enable_correctness_check = false);

}  // namespace tt::tt_metal
