// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::distributed {

// Multi-Dimensional coordinate struct used to access individual cores in a MeshDevice.
struct MeshCoreCoord {
    MeshCoordinate device_coord;
    CoreCoord core_coord;
};

// Specifies how sender cores on a Virtual Mesh connect to receiver cores on the same or another Virtual Mesh.
// Used to determine which cores the socket config must be written to and the sender to receiver mapping.
// Cannot reuse senders and receivers in a single socket context. Each socket connection is 1:1.
struct SocketConnection {
    MeshCoreCoord sender_core;
    MeshCoreCoord receiver_core;
};

// Specifies how memory is allocated for this socket.
// Socket memory is allocated in lockstep across each MeshDevice.
struct SocketMemoryConfig {
    BufferType socket_storage_type = BufferType::L1;
    uint32_t fifo_size = 0;
    // Up to the user: Can tie socket lifetime to sub device lifetime and regen socket
    // each time SD is loaded Or keep socket persistent in global mem pool and use across SD.
    // TODO: Should data cores be on a different sub device?
    std::optional<SubDeviceId> sender_sub_device = std::nullopt;
    std::optional<SubDeviceId> receiver_sub_device = std::nullopt;
};

// A socket config fully specifies the following:
// 1. The physical connections making up a socket (can only support single sender to single receiver -> cannot reuse
// sender and receiver cores in a socket)
// 2. Memory allocations required to setup the socket.
struct SocketConfig {
    std::vector<SocketConnection> socket_connection_config;
    SocketMemoryConfig socket_mem_config;
};

// Socket Handle exposed to the user.
// A user can use this object to allocate and open multiple connections between two different MeshDevices
// or within the same MeshDevice.
// The connectivity of sender/receiver endpoints over sockets is encapsulated in the socket_connection_config, passed
// through the socket_config object.
class MeshSocket {
public:
    // Sockets can only be created in sender/receiver pairs.
    static std::pair<MeshSocket, MeshSocket> create_sockets(
        const std::shared_ptr<MeshDevice>& sender,
        const std::shared_ptr<MeshDevice>& receiver,
        const SocketConfig& config);
    // Access the data-buffer associated with the socket on the reciver mesh. Can only be queried for receiver sockets.
    std::shared_ptr<MeshBuffer> get_data_buffer() const;
    // Access the config buffer associated with this socket.
    std::shared_ptr<MeshBuffer> get_config_buffer() const;
    // Access the underlying configuration of the instantiated socket (connectivity of senders/receivers and the socket
    // memory config).
    const SocketConfig& get_config() const;

private:
    MeshSocket(
        std::shared_ptr<MeshBuffer> data_buffer,
        std::shared_ptr<MeshBuffer> config_buffer,
        const SocketConfig& config) :
        data_buffer_(data_buffer), config_buffer_(config_buffer), config_(config) {}
    std::shared_ptr<MeshBuffer> data_buffer_;
    std::shared_ptr<MeshBuffer> config_buffer_;
    SocketConfig config_;
};

}  // namespace tt::tt_metal::distributed
