// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::distributed {

// Socket Handle exposed to the user.
class mesh_socket_t {
public:
    std::shared_ptr<MeshBuffer> data_buffer;
    std::shared_ptr<MeshBuffer> config_buffer;
};

// Specifies how sender cores on a Virtual Mesh connect to receiver cores on another Virtual Mesh.
// Used to determine which cores the socket config must be written to and the sender to receiver mapping.
// Cannot reuse senders and receivers in a single socket context. Each socket connection is 1:1.
class socket_connection_t {
public:
    std::pair<MeshCoordinate, CoreCoord> sender_core;
    std::pair<MeshCoordinate, CoreCoord> receiver_core;
};

// Specifies how memory is allocated for this socket.
// Socket memory is allocated in lockstep across each MeshDevice.
class socket_memory_config_t {
public:
    BufferType socket_type = BufferType::L1;
    SubDeviceId sender_sub_device{0};
    SubDeviceId receiver_sub_device{0};
    uint32_t fifo_size = 0;
};

// A socket context fully specifies the following:
// 1. The connections making up a socket (can only support single sender to single receiver -> cannot reuse sender and
// receiver cores in a socket)
// 2. Memory allocations required to setup the socket.
class socket_config_t {
public:
    std::vector<socket_connection_t> socket_connection_config;
    socket_memory_config_t socket_mem_config;
};

std::pair<mesh_socket_t, mesh_socket_t> create_sockets(
    std::shared_ptr<MeshDevice> sender, std::shared_ptr<MeshDevice> receiver, const socket_config_t& config);

}  // namespace tt::tt_metal::distributed
