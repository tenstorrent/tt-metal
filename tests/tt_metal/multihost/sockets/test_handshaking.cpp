// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include "tests/tt_metal/multihost/common/multihost_test_tools.hpp"
#include <random>

using namespace tt::tt_metal::distributed::multihost;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal;

// This test does not use any devices or allocate any sockets.
// Create socket configs and populate dummy addresses in the peer descriptors.
// Send and receive them over the wire + validate to ensure that the handshaking
// works correctly.
TEST(MultiHostSocketTest, MultiProcessHandshaking) {
    // Get the current distributed context
    const auto& context = DistributedContext::get_current_world();
    TT_FATAL(context != nullptr, "DistributedContext is not initialized.");

    auto rank = *context->rank();  // Host rank
    auto size = *context->size();  // World size

    EXPECT_EQ(size, 2);  // Ensure a world size of 2 for this test

    std::unordered_map<Rank, Rank> rank_translation_table = {
        {Rank{0}, Rank{0}},
        {Rank{1}, Rank{1}},
    };

    // Create Socket Configs
    std::size_t l1_socket_fifo_size = 1024;
    std::size_t dram_socket_fifo_size = 2048;
    // 8x8 Worker Grid (simulate Wormhole)
    const auto worker_grid = CoreCoord(8, 8);

    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;
    std::vector<MeshCoordinate> sender_device_coords;
    std::vector<MeshCoordinate> recv_device_coords;

    uint32_t core_idx = 0;
    for (std::size_t x = 0; x < worker_grid.x; x++) {
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x, y));
            sender_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            recv_device_coords.push_back(MeshCoordinate(1, core_idx % 4));
            core_idx++;
        }
    }

    std::vector<SocketConnection> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        SocketConnection socket_connection(
            MeshCoreCoord(sender_device_coords[coord_idx], sender_logical_coords[coord_idx]),
            MeshCoreCoord(recv_device_coords[coord_idx], recv_logical_coords[coord_idx]));
        socket_connections.push_back(socket_connection);
    }
    // L1 Socket Config
    SocketMemoryConfig socket_mem_config_l1(BufferType::L1, l1_socket_fifo_size, SubDeviceId(0), SubDeviceId(1));
    SocketConfig socket_config_l1 =
        SocketConfig(socket_connections, socket_mem_config_l1, tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshId{1});
    // Dram Socket Config
    SocketMemoryConfig socket_mem_config_dram(BufferType::DRAM, dram_socket_fifo_size, SubDeviceId(2), SubDeviceId(3));
    SocketConfig socket_config_dram =
        SocketConfig(socket_connections, socket_mem_config_dram, tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshId{1});
    // This config will be used to ensure that the verification step works correctly
    // Descriptors will be forced to mismatch.
    SocketMemoryConfig incorrect_socket_mem_config(
        BufferType::DRAM, dram_socket_fifo_size, SubDeviceId(2), SubDeviceId(3));
    SocketConfig incorrect_socket_config = SocketConfig(
        socket_connections, incorrect_socket_mem_config, tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshId{1});
    // Generate dummy addresses for the sender and receiver buffers
    // In a real scenario, these would be allocated buffers on the MeshDevice.
    DeviceAddr l1_sender_config_buffer_address = 1 << 10;    // DUMMY ADDRESS: L1 Sender config buffer allocate at 1KB
    DeviceAddr l1_receiver_config_buffer_address = 2 << 10;  // DUMMY ADDRESS: L1 Receiver config buffer allocate at 2KB
    DeviceAddr l1_receiver_data_buffer_address = 3 << 10;    // DUMMY ADDRESS: L1 Receiver data buffer allocate at 3KB
    DeviceAddr dram_sender_config_buffer_address = 4 << 10;  // DUMMY ADDRESS: DRAM Sender config buffer allocate at 4KB
    DeviceAddr dram_receiver_config_buffer_address =
        5 << 10;  // DUMMY ADDRESS: DRAM Receiver config buffer allocate at 5KB
    DeviceAddr dram_receiver_data_buffer_address = 6 << 10;  // DUMMY ADDRESS: DRAM Receiver data buffer allocate at 6KB

    if (rank == 0) {
        SocketPeerDescriptor send_peer_descriptor_l1 = {
            .config = socket_config_l1,
            .config_buffer_address = l1_sender_config_buffer_address,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{1},
        };
        SocketPeerDescriptor send_peer_descriptor_dram = {
            .config = socket_config_dram,
            .config_buffer_address = dram_sender_config_buffer_address,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{2},
        };
        incorrect_socket_config.socket_mem_config.sender_sub_device = SubDeviceId(4);
        incorrect_socket_config.socket_mem_config.receiver_sub_device = SubDeviceId(5);
        SocketPeerDescriptor incorrect_socket_descriptor = {
            .config = incorrect_socket_config,
            .config_buffer_address = 0,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{3},
        };
        // Handshake on L1 Socket
        // SocketConfig validated here
        forward_descriptor_to_peer(send_peer_descriptor_l1, SocketEndpoint::SENDER, context, rank_translation_table);
        auto peer_desc = receive_and_verify_descriptor_from_peer(
            send_peer_descriptor_l1, SocketEndpoint::SENDER, context, rank_translation_table);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, l1_receiver_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, l1_receiver_data_buffer_address);
        // Handshake on DRAM Socket
        forward_descriptor_to_peer(send_peer_descriptor_dram, SocketEndpoint::SENDER, context, rank_translation_table);
        peer_desc = receive_and_verify_descriptor_from_peer(
            send_peer_descriptor_dram, SocketEndpoint::SENDER, context, rank_translation_table);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, dram_receiver_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, dram_receiver_data_buffer_address);
        forward_descriptor_to_peer(
            incorrect_socket_descriptor, SocketEndpoint::SENDER, context, rank_translation_table);
        // Validate that the incorrect socket descriptor is rejected
        EXPECT_THROW(
            receive_and_verify_descriptor_from_peer(
                incorrect_socket_descriptor, SocketEndpoint::SENDER, context, rank_translation_table),
            std::runtime_error);

    } else {
        SocketPeerDescriptor recv_peer_descriptor_l1 = {
            .config = socket_config_l1,
            .config_buffer_address = l1_receiver_config_buffer_address,
            .data_buffer_address = l1_receiver_data_buffer_address,
            .exchange_tag = Tag{1},
        };
        SocketPeerDescriptor recv_peer_descriptor_dram = {
            .config = socket_config_dram,
            .config_buffer_address = dram_receiver_config_buffer_address,
            .data_buffer_address = dram_receiver_data_buffer_address,
            .exchange_tag = Tag{2},
        };

        SocketPeerDescriptor incorrect_socket_descriptor = {
            .config = incorrect_socket_config,
            .config_buffer_address = 0,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{3},
        };
        // Handshake on L1 Socket
        // SocketConfig validated here
        auto peer_desc = receive_and_verify_descriptor_from_peer(
            recv_peer_descriptor_l1, SocketEndpoint::RECEIVER, context, rank_translation_table);
        forward_descriptor_to_peer(recv_peer_descriptor_l1, SocketEndpoint::RECEIVER, context, rank_translation_table);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, l1_sender_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, 0);  // Sender does not have a data buffer
        // Handshake on DRAM Socket
        peer_desc = receive_and_verify_descriptor_from_peer(
            recv_peer_descriptor_dram, SocketEndpoint::RECEIVER, context, rank_translation_table);
        forward_descriptor_to_peer(
            recv_peer_descriptor_dram, SocketEndpoint::RECEIVER, context, rank_translation_table);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, dram_sender_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, 0);  // Sender does not have a data buffer
        // Validate that the incorrect socket descriptor is rejected
        EXPECT_THROW(
            receive_and_verify_descriptor_from_peer(
                incorrect_socket_descriptor, SocketEndpoint::RECEIVER, context, rank_translation_table),
            std::runtime_error);
        forward_descriptor_to_peer(
            incorrect_socket_descriptor, SocketEndpoint::RECEIVER, context, rank_translation_table);
    }
}
