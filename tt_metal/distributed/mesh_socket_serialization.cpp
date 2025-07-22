// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include "socket_peer_descriptor_generated.h"
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_metal::distributed {
namespace {

flatbuffers::Offset<distributed::flatbuffer::CoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreCoord& coord) {
    return distributed::flatbuffer::CreateCoreCoord(builder, coord.x, coord.y);
}

flatbuffers::Offset<distributed::flatbuffer::MeshCoordinate> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const MeshCoordinate& mesh_coord) {
    auto values_vector = builder.CreateVector(mesh_coord.coords().data(), mesh_coord.coords().size());
    return distributed::flatbuffer::CreateMeshCoordinate(builder, values_vector);
}

flatbuffers::Offset<distributed::flatbuffer::MeshCoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const MeshCoreCoord& mesh_core_coord) {
    auto device_coord = to_flatbuffer(builder, mesh_core_coord.device_coord);
    auto core_coord = to_flatbuffer(builder, mesh_core_coord.core_coord);
    return distributed::flatbuffer::CreateMeshCoreCoord(builder, device_coord, core_coord);
}

flatbuffers::Offset<distributed::flatbuffer::SocketConnection> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const SocketConnection& socket_connection) {
    auto sender_core = to_flatbuffer(builder, socket_connection.sender_core);
    auto receiver_core = to_flatbuffer(builder, socket_connection.receiver_core);
    return distributed::flatbuffer::CreateSocketConnection(builder, sender_core, receiver_core);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<distributed::flatbuffer::SocketConnection>>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<SocketConnection>& socket_connections) {
    std::vector<flatbuffers::Offset<distributed::flatbuffer::SocketConnection>> fb_connections;
    fb_connections.reserve(socket_connections.size());
    for (const auto& conn : socket_connections) {
        fb_connections.push_back(to_flatbuffer(builder, conn));
    }
    return builder.CreateVector(fb_connections);
}

flatbuffers::Offset<distributed::flatbuffer::SocketMemoryConfig> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const SocketMemoryConfig& socket_mem_config) {
    std::optional<uint32_t> sender_sub_device = std::nullopt;
    if (socket_mem_config.sender_sub_device.has_value()) {
        sender_sub_device = *(socket_mem_config.sender_sub_device.value());
    }

    std::optional<uint32_t> receiver_sub_device = std::nullopt;
    if (socket_mem_config.receiver_sub_device.has_value()) {
        receiver_sub_device = *(socket_mem_config.receiver_sub_device.value());
    }

    return distributed::flatbuffer::CreateSocketMemoryConfig(
        builder,
        static_cast<uint32_t>(socket_mem_config.socket_storage_type),
        socket_mem_config.fifo_size,
        sender_sub_device,
        receiver_sub_device);
}

CoreCoord from_flatbuffer(const distributed::flatbuffer::CoreCoord* fb_core_coord) {
    return CoreCoord(fb_core_coord->x(), fb_core_coord->y());
}

MeshCoordinate from_flatbuffer(const distributed::flatbuffer::MeshCoordinate* fb_mesh_coord) {
    return MeshCoordinate(*(fb_mesh_coord->values()));
}

MeshCoreCoord from_flatbuffer(const distributed::flatbuffer::MeshCoreCoord* fb_mesh_core_coord) {
    TT_FATAL(
        fb_mesh_core_coord->device_coord(),
        "Internal Error: Device coord is not present in MeshCoreCoord during deserialization");
    TT_FATAL(
        fb_mesh_core_coord->core_coord(),
        "Internal Error: Core coord is not present in MeshCoreCoord during deserialization");

    return MeshCoreCoord{
        from_flatbuffer(fb_mesh_core_coord->device_coord()), from_flatbuffer(fb_mesh_core_coord->core_coord())};
}

SocketConnection from_flatbuffer(const distributed::flatbuffer::SocketConnection* fb_socket_connection) {
    TT_FATAL(
        fb_socket_connection->sender_core(),
        "Internal Error: Sender core is not present in the socket connection during deserialization");
    TT_FATAL(
        fb_socket_connection->receiver_core(),
        "Internal Error: Receiver core is not present in the socket connection during deserialization");

    return SocketConnection{
        .sender_core = from_flatbuffer(fb_socket_connection->sender_core()),
        .receiver_core = from_flatbuffer(fb_socket_connection->receiver_core())};
}

std::vector<SocketConnection> from_flatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<distributed::flatbuffer::SocketConnection>>* fb_socket_connections) {
    std::vector<SocketConnection> socket_connections;
    if (fb_socket_connections) {
        socket_connections.reserve(fb_socket_connections->size());
        for (const auto* fb_conn : *fb_socket_connections) {
            socket_connections.push_back(from_flatbuffer(fb_conn));
        }
    }
    return socket_connections;
}

SocketMemoryConfig from_flatbuffer(const distributed::flatbuffer::SocketMemoryConfig* fb_socket_mem_config) {
    SocketMemoryConfig socket_mem_config;
    if (fb_socket_mem_config) {
        socket_mem_config.socket_storage_type = static_cast<BufferType>(fb_socket_mem_config->socket_storage_type());
        socket_mem_config.fifo_size = fb_socket_mem_config->fifo_size();

        // Handle optional fields
        if (fb_socket_mem_config->sender_sub_device().has_value()) {
            socket_mem_config.sender_sub_device = SubDeviceId(fb_socket_mem_config->sender_sub_device().value());
        }
        if (fb_socket_mem_config->receiver_sub_device().has_value()) {
            socket_mem_config.receiver_sub_device = SubDeviceId(fb_socket_mem_config->receiver_sub_device().value());
        }
    }
    return socket_mem_config;
}

}  // namespace

std::vector<uint8_t> serialize_to_bytes(const SocketPeerDescriptor& socket_peer_desc) {
    flatbuffers::FlatBufferBuilder builder;
    auto& socket_config = socket_peer_desc.config;
    // Build the connections FlatBuffer
    auto connections_vector_fb = to_flatbuffer(builder, socket_config.socket_connection_config);
    // Build the memory config FlatBuffer`
    auto mem_config_fb = to_flatbuffer(builder, socket_config.socket_mem_config);
    // Create the SocketConfig FlatBuffer
    auto socket_config_fb = distributed::flatbuffer::CreateSocketConfig(
        builder, connections_vector_fb, mem_config_fb, *socket_config.sender_rank, *socket_config.receiver_rank);
    // Build the SocketPeerDescriptor FlatBuffer (root object)
    auto socket_peer_desc_fb = distributed::flatbuffer::CreateSocketPeerDescriptor(
        builder,
        socket_config_fb,
        socket_peer_desc.config_buffer_address,
        socket_peer_desc.data_buffer_address,
        builder.CreateVector(socket_peer_desc.mesh_ids),
        builder.CreateVector(socket_peer_desc.chip_ids),
        *(socket_peer_desc.exchange_tag));

    builder.Finish(socket_peer_desc_fb);
    // Extract the FlatBuffer data as a vector of bytes
    // Note: FlatBufferBuilder's GetBufferPointer returns a pointer to the internal buffer, which is not std::byte.
    std::vector<uint8_t> byte_vector(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    return byte_vector;
}

SocketPeerDescriptor deserialize_from_bytes(const std::vector<uint8_t>& data) {
    // Verify the buffer: FlatBuffer Verifier requires a uint8_t pointer
    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    TT_FATAL(
        distributed::flatbuffer::VerifySocketPeerDescriptorBuffer(verifier),
        "Invalid FlatBuffer data of distributed socket metadata");

    auto socket_peer_desc_fb = distributed::flatbuffer::GetSocketPeerDescriptor(data.data());
    auto socket_config_fb = socket_peer_desc_fb->config();

    SocketPeerDescriptor socket_peer_desc;
    // Populate the SocketPeerDescriptor from the FlatBuffer (connections, memory config, ranks, peer address, Mesh IDs,
    // Chip IDs)
    socket_peer_desc.config.socket_connection_config = from_flatbuffer(socket_config_fb->socket_connections());
    socket_peer_desc.config.socket_mem_config = from_flatbuffer(socket_config_fb->socket_mem_config());
    socket_peer_desc.config.sender_rank = multihost::Rank{socket_config_fb->sender_rank()};
    socket_peer_desc.config.receiver_rank = multihost::Rank{socket_config_fb->receiver_rank()};
    socket_peer_desc.config_buffer_address = socket_peer_desc_fb->config_buffer_address();
    socket_peer_desc.data_buffer_address = socket_peer_desc_fb->data_buffer_address();
    if (socket_peer_desc_fb->mesh_ids()) {
        socket_peer_desc.mesh_ids.assign(
            socket_peer_desc_fb->mesh_ids()->begin(), socket_peer_desc_fb->mesh_ids()->end());
    }
    if (socket_peer_desc_fb->chip_ids()) {
        socket_peer_desc.chip_ids.assign(
            socket_peer_desc_fb->chip_ids()->begin(), socket_peer_desc_fb->chip_ids()->end());
    }
    socket_peer_desc.exchange_tag = multihost::Tag{socket_peer_desc_fb->exchange_tag()};
    return socket_peer_desc;
}

}  // namespace tt::tt_metal::distributed
