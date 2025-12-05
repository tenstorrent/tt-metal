// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <enchantum/enchantum.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <utility>

namespace tt::tt_metal::distributed {

// Multi-Dimensional coordinate struct used to access individual cores in a MeshDevice.
struct MeshCoreCoord {
    MeshCoordinate device_coord = MeshCoordinate(0);
    CoreCoord core_coord = CoreCoord(0, 0);

    // User-provided constructor to make this non-aggregate (prevents ambiguity with Reflectable concept)
    MeshCoreCoord() = default;
    MeshCoreCoord(const MeshCoordinate& device_coord, const CoreCoord& core_coord) :
        device_coord(device_coord), core_coord(core_coord) {}

    bool operator==(const MeshCoreCoord& other) const {
        return device_coord == other.device_coord && core_coord == other.core_coord;
    }

    static constexpr auto attribute_names = std::forward_as_tuple("device_coord", "core_coord");
    auto attribute_values() const { return std::forward_as_tuple(device_coord, core_coord); }
};

}  // namespace tt::tt_metal::distributed

namespace std {

template <>
struct hash<tt::tt_metal::distributed::MeshCoreCoord> {
    size_t operator()(const tt::tt_metal::distributed::MeshCoreCoord& coord) const noexcept;
};

}  // namespace std

namespace tt::tt_metal::distributed {

class H2DSocket {
public:
    H2DSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const std::vector<MeshCoreCoord>& recv_cores,
        BufferType buffer_type,
        uint32_t fifo_size);

    void reserve_pages(uint32_t num_pages);
    void push_pages(uint32_t num_pages);
    void notify_receiver();
    uint32_t get_page_size() const { return page_size_; }
    uint32_t* get_write_ptr() const { return host_data_buffer_->data() + (write_ptr_ / sizeof(uint32_t)); }
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }
    void set_page_size(uint32_t page_size);
    void barrier();
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> get_bytes_acked_buffer() const {
        return bytes_acked_buffer_;
    }

private:
    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> data_buffer_ = nullptr;
    std::vector<MeshCoreCoord> recv_cores_ = {};
    BufferType buffer_type_ = BufferType::L1;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_sent_ = 0;
    std::unordered_map<MeshCoreCoord, uint32_t> bytes_acked_ = {};
    uint32_t write_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    std::unique_ptr<tt::tt_metal::experimental::PinnedMemory> bytes_acked_pinned_memory_ = nullptr;
    std::unique_ptr<tt::tt_metal::experimental::PinnedMemory> data_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> bytes_acked_buffer_ = nullptr;
    std::shared_ptr<std::vector<uint32_t, tt::stl::aligned_allocator<uint32_t, 64>>> host_data_buffer_ = nullptr;
};

class D2HSocket {
public:
    D2HSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& sender_core,
        BufferType buffer_type,
        uint32_t fifo_size);

    void wait_for_pages(uint32_t num_pages);
    void pop_pages(uint32_t num_pages);
    void notify_sender();
    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }
    uint32_t* get_read_ptr() const { return data_buffer_->data() + (read_ptr_ / sizeof(uint32_t)); }
    void set_page_size(uint32_t page_size);
    void barrier();

private:
    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> data_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> bytes_sent_pinned_memory_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> data_buffer_ = nullptr;
    std::shared_ptr<tt::tt_metal::vector_aligned<uint32_t>> bytes_sent_buffer_ = nullptr;

    MeshCoreCoord sender_core_ = {};
    BufferType buffer_type_ = BufferType::DRAM;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t read_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
};

// Specifies how sender cores on a Virtual Mesh connect to receiver cores on the same or another Virtual Mesh.
// Used to determine which cores the socket config must be written to and the sender to receiver mapping.
// Cannot reuse senders and receivers in a single socket context. Each socket connection is 1:1.
struct SocketConnection {
    MeshCoreCoord sender_core;
    MeshCoreCoord receiver_core;

    // User-provided constructor to make this non-aggregate (prevents ambiguity with Reflectable concept)
    SocketConnection() = default;
    SocketConnection(const MeshCoreCoord& sender_core, const MeshCoreCoord& receiver_core) :
        sender_core(sender_core), receiver_core(receiver_core) {}

    bool operator==(const SocketConnection& other) const {
        return sender_core == other.sender_core && receiver_core == other.receiver_core;
    }

    static constexpr auto attribute_names = std::forward_as_tuple("sender_core", "receiver_core");
    auto attribute_values() const { return std::forward_as_tuple(sender_core, receiver_core); }
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

    // User-provided constructor to make this non-aggregate (prevents ambiguity with Reflectable concept)
    SocketMemoryConfig() = default;
    SocketMemoryConfig(
        BufferType socket_storage_type,
        uint32_t fifo_size,
        std::optional<SubDeviceId> sender_sub_device = std::nullopt,
        std::optional<SubDeviceId> receiver_sub_device = std::nullopt) :
        socket_storage_type(socket_storage_type),
        fifo_size(fifo_size),
        sender_sub_device(sender_sub_device),
        receiver_sub_device(receiver_sub_device) {}

    static constexpr auto attribute_names =
        std::forward_as_tuple("socket_storage_type", "fifo_size", "sender_sub_device", "receiver_sub_device");
    auto attribute_values() const {
        return std::forward_as_tuple(socket_storage_type, fifo_size, sender_sub_device, receiver_sub_device);
    }
};

// A socket config fully specifies the following:
// 1. The physical connections making up a socket (can only support single sender to single receiver -> cannot reuse
// sender and receiver cores in a socket)
// 2. Memory allocations required to setup the socket.
struct SocketConfig {
    std::vector<SocketConnection> socket_connection_config;
    SocketMemoryConfig socket_mem_config;
    // Specifies the ranks of the sender and receiver hosts in a multi-host context.
    // Used for inital handshaking and validation of the socket configs.
    std::optional<tt::tt_fabric::MeshId> sender_mesh_id = std::nullopt;
    std::optional<tt::tt_fabric::MeshId> receiver_mesh_id = std::nullopt;
    multihost::Rank sender_rank{0};
    multihost::Rank receiver_rank{0};
    std::shared_ptr<multihost::DistributedContext> distributed_context = nullptr;

    SocketConfig() = default;

    SocketConfig(
        const std::vector<SocketConnection>& socket_connection_config,
        const SocketMemoryConfig& socket_mem_config,
        std::optional<tt::tt_fabric::MeshId> sender_mesh_id = std::nullopt,
        std::optional<tt::tt_fabric::MeshId> receiver_mesh_id = std::nullopt,
        const std::shared_ptr<multihost::DistributedContext>& distributed_context = nullptr) :
        socket_connection_config(socket_connection_config),
        socket_mem_config(socket_mem_config),
        sender_mesh_id(sender_mesh_id),
        receiver_mesh_id(receiver_mesh_id),
        distributed_context(distributed_context) {}

    SocketConfig(
        const std::vector<SocketConnection>& socket_connection_config,
        const SocketMemoryConfig& socket_mem_config,
        multihost::Rank sender_rank,
        multihost::Rank receiver_rank,
        const std::shared_ptr<multihost::DistributedContext>& distributed_context = nullptr) :
        socket_connection_config(socket_connection_config),
        socket_mem_config(socket_mem_config),
        sender_rank(sender_rank),
        receiver_rank(receiver_rank),
        distributed_context(distributed_context) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "socket_connection_config",
        "socket_mem_config",
        "sender_mesh_id",
        "receiver_mesh_id",
        "sender_rank",
        "receiver_rank");
    auto attribute_values() const {
        return std::forward_as_tuple(
            socket_connection_config, socket_mem_config, sender_mesh_id, receiver_mesh_id, sender_rank, receiver_rank);
    }
};

enum class SocketEndpoint : uint8_t { SENDER, RECEIVER };

// Socket Handle exposed to the user.
// A user can use this object to allocate and open multiple connections between two different MeshDevices
// or within the same MeshDevice.
// The connectivity of sender/receiver endpoints over sockets is encapsulated in the socket_connection_config, passed
// through the socket_config object.
class MeshSocket {
public:
    MeshSocket(const std::shared_ptr<MeshDevice>& device, const SocketConfig& config);
    // Sockets can only be created in sender/receiver pairs.
    static std::pair<MeshSocket, MeshSocket> create_socket_pair(
        const std::shared_ptr<MeshDevice>& sender,
        const std::shared_ptr<MeshDevice>& receiver,
        const SocketConfig& base_config);
    // Access the data-buffer associated with the socket on the reciver mesh. Can only be queried for receiver sockets.
    std::shared_ptr<MeshBuffer> get_data_buffer() const;
    // Access the config buffer associated with this socket.
    std::shared_ptr<MeshBuffer> get_config_buffer() const;
    // Access the underlying configuration of the instantiated socket (connectivity of senders/receivers and the socket
    // memory config).
    const SocketConfig& get_config() const;
    // Access the socket endpoint type (SENDER or RECEIVER).
    SocketEndpoint get_socket_endpoint_type() const { return socket_endpoint_type_; }

    tt::tt_fabric::FabricNodeId get_fabric_node_id(SocketEndpoint endpoint, const MeshCoordinate& coord) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("config", "socket_endpoint_type", "fabric_node_id_map");
    auto attribute_values() const { return std::forward_as_tuple(config_, socket_endpoint_type_, fabric_node_id_map_); }

private:
    MeshSocket(
        std::shared_ptr<MeshBuffer> data_buffer,
        std::shared_ptr<MeshBuffer> config_buffer,
        const SocketConfig& config,
        SocketEndpoint socket_endpoint_type) :
        data_buffer_(std::move(data_buffer)),
        config_buffer_(std::move(config_buffer)),
        config_(config),
        socket_endpoint_type_(socket_endpoint_type) {}
    void process_host_ranks();
    void process_mesh_ids();
    static SocketConfig populate_mesh_ids(
        const std::shared_ptr<MeshDevice>& sender,
        const std::shared_ptr<MeshDevice>& receiver,
        const SocketConfig& base_config);
    void connect_with_peer(const std::shared_ptr<multihost::DistributedContext>& context);

    std::shared_ptr<MeshBuffer> data_buffer_;
    std::shared_ptr<MeshBuffer> config_buffer_;
    SocketConfig config_;
    SocketEndpoint socket_endpoint_type_;
    std::unordered_map<multihost::Rank, multihost::Rank> rank_translation_table_;
    // TODO: replace with enchantum::array
    std::array<std::unordered_map<MeshCoordinate, tt::tt_fabric::FabricNodeId>, enchantum::count<SocketEndpoint>>
        fabric_node_id_map_;
};

}  // namespace tt::tt_metal::distributed

namespace std {
template <>
struct hash<tt::tt_metal::distributed::SocketConnection> {
    size_t operator()(const tt::tt_metal::distributed::SocketConnection& conn) const noexcept;
};
template <>
struct hash<tt::tt_metal::distributed::SocketConfig> {
    size_t operator()(const tt::tt_metal::distributed::SocketConfig& config) const noexcept;
};
template <>
struct hash<tt::tt_metal::distributed::MeshSocket> {
    size_t operator()(const tt::tt_metal::distributed::MeshSocket& socket) const noexcept;
};

}  // namespace std
