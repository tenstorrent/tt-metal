// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    auto context = DistributedContext::get_current_world();
    TT_FATAL(context != nullptr, "DistributedContext is not initialized.");

    auto rank = *context->rank();  // Host rank
    auto size = *context->size();  // World size

    EXPECT_EQ(size, 2);  // Ensure a world size of 2 for this test

    // Create Socket Configs
    std::size_t l1_socket_fifo_size = 1024;
    std::size_t dram_socket_fifo_size = 2048;
    // 8x8 Worker Grid (simulate Wormhole)
    const auto worker_grid = CoreCoord(8, 8);

    std::vector<CoreCoord> sender_logical_coords;
    std::vector<CoreCoord> recv_logical_coords;
    std::vector<uint32_t> sender_chip_ids;
    std::vector<uint32_t> recv_chip_ids;
    std::vector<MeshCoordinate> sender_device_coords;
    std::vector<MeshCoordinate> recv_device_coords;

    uint32_t core_idx = 0;
    for (std::size_t x = 0; x < worker_grid.x; x++) {
        for (std::size_t y = 0; y < worker_grid.y; y++) {
            sender_logical_coords.push_back(CoreCoord(x, y));
            recv_logical_coords.push_back(CoreCoord(x, y));
            sender_chip_ids.push_back(core_idx % 4);
            recv_chip_ids.push_back(4 + core_idx % 4);
            sender_device_coords.push_back(MeshCoordinate(0, core_idx % 4));
            recv_device_coords.push_back(MeshCoordinate(1, core_idx % 4));
            core_idx++;
        }
    }

    std::vector<SocketConnection> socket_connections;

    for (std::size_t coord_idx = 0; coord_idx < sender_logical_coords.size(); coord_idx++) {
        SocketConnection socket_connection = {
            .sender_core = {sender_device_coords[coord_idx], sender_logical_coords[coord_idx]},
            .receiver_core = {recv_device_coords[coord_idx], recv_logical_coords[coord_idx]}};
        socket_connections.push_back(socket_connection);
    }
    // L1 Socket Config
    SocketConfig socket_config_l1 = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {.socket_storage_type = BufferType::L1,
             .fifo_size = l1_socket_fifo_size,
             .sender_sub_device = SubDeviceId(0),
             .receiver_sub_device = SubDeviceId(1)},
        .sender_rank = Rank{0},
        .receiver_rank = Rank{1},
    };
    // Dram Socket Config
    SocketConfig socket_config_dram = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {.socket_storage_type = BufferType::DRAM,
             .fifo_size = dram_socket_fifo_size,
             .sender_sub_device = SubDeviceId(2),
             .receiver_sub_device = SubDeviceId(3)},
        .sender_rank = Rank{0},
        .receiver_rank = Rank{1},
    };
    // This config will be used to ensure that the verification step works correctly
    // Descriptors will be forced to mismatch.
    SocketConfig incorrect_socket_config = {
        .socket_connection_config = socket_connections,
        .socket_mem_config =
            {.socket_storage_type = BufferType::DRAM,
             .fifo_size = dram_socket_fifo_size,
             .sender_sub_device = SubDeviceId(2),
             .receiver_sub_device = SubDeviceId(3)},
        .sender_rank = Rank{0},
        .receiver_rank = Rank{1},
    };
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

        for (const auto& chip_id : sender_chip_ids) {
            send_peer_descriptor_l1.chip_ids.push_back(chip_id);
            send_peer_descriptor_l1.mesh_ids.push_back(0);  // Dummy mesh ID for sender
        }
        for (const auto& chip_id : sender_chip_ids) {
            send_peer_descriptor_dram.chip_ids.push_back(chip_id);
            send_peer_descriptor_dram.mesh_ids.push_back(2);  // Dummy mesh ID for sender
        }
        SocketPeerDescriptor incorrect_socket_descriptor = {
            .config = incorrect_socket_config,
            .config_buffer_address = 0,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{3},
        };
        // Incorrectly populate the chip_ids and mesh_ids to ensure verification works
        incorrect_socket_descriptor.chip_ids = {0, 1, 2, 3};
        incorrect_socket_descriptor.mesh_ids = {0, 0, 0, 0};
        // Handshake on L1 Socket
        // SocketConfig validated here
        forward_descriptor_to_peer(send_peer_descriptor_l1, SocketEndpoint::SENDER, context);
        auto peer_desc =
            receive_and_verify_descriptor_from_peer(send_peer_descriptor_l1, SocketEndpoint::SENDER, context);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, l1_receiver_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, l1_receiver_data_buffer_address);
        for (size_t i = 0; i < sender_chip_ids.size(); ++i) {
            EXPECT_EQ(peer_desc.chip_ids[i], recv_chip_ids[i]);
            EXPECT_EQ(peer_desc.mesh_ids[i], 1);
        }
        // Handshake on DRAM Socket
        forward_descriptor_to_peer(send_peer_descriptor_dram, SocketEndpoint::SENDER, context);
        peer_desc = receive_and_verify_descriptor_from_peer(send_peer_descriptor_dram, SocketEndpoint::SENDER, context);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, dram_receiver_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, dram_receiver_data_buffer_address);
        for (size_t i = 0; i < sender_chip_ids.size(); ++i) {
            EXPECT_EQ(peer_desc.chip_ids[i], recv_chip_ids[i]);
            EXPECT_EQ(peer_desc.mesh_ids[i], 3);
        }
        forward_descriptor_to_peer(incorrect_socket_descriptor, SocketEndpoint::SENDER, context);
        // Validate that the incorrect socket descriptor is rejected
        EXPECT_THROW(
            receive_and_verify_descriptor_from_peer(incorrect_socket_descriptor, SocketEndpoint::SENDER, context),
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

        for (const auto& chip_id : recv_chip_ids) {
            recv_peer_descriptor_l1.chip_ids.push_back(chip_id);
            recv_peer_descriptor_l1.mesh_ids.push_back(1);  // Dummy mesh ID for receiver
        }
        for (const auto& chip_id : recv_chip_ids) {
            recv_peer_descriptor_dram.chip_ids.push_back(chip_id);
            recv_peer_descriptor_dram.mesh_ids.push_back(3);  // Dummy mesh ID for receiver
        }
        SocketPeerDescriptor incorrect_socket_descriptor = {
            .config = incorrect_socket_config,
            .config_buffer_address = 0,
            .data_buffer_address = 0,  // Sender does not have a data buffer
            .exchange_tag = Tag{3},
        };
        // Incorrectly populate the chip_ids and mesh_ids to ensure verification works
        incorrect_socket_descriptor.chip_ids = {0, 1, 2, 3};
        incorrect_socket_descriptor.mesh_ids = {1, 1, 1, 1};
        // Handshake on L1 Socket
        // SocketConfig validated here
        auto peer_desc =
            receive_and_verify_descriptor_from_peer(recv_peer_descriptor_l1, SocketEndpoint::RECEIVER, context);
        forward_descriptor_to_peer(recv_peer_descriptor_l1, SocketEndpoint::RECEIVER, context);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, l1_sender_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, 0);  // Sender does not have a data buffer
        for (size_t i = 0; i < recv_chip_ids.size(); ++i) {
            EXPECT_EQ(peer_desc.chip_ids[i], sender_chip_ids[i]);
            EXPECT_EQ(peer_desc.mesh_ids[i], 0);
        }
        // Handshake on DRAM Socket
        peer_desc =
            receive_and_verify_descriptor_from_peer(recv_peer_descriptor_dram, SocketEndpoint::RECEIVER, context);
        forward_descriptor_to_peer(recv_peer_descriptor_dram, SocketEndpoint::RECEIVER, context);
        // Validate all other fields in the peer descriptor
        EXPECT_EQ(peer_desc.config_buffer_address, dram_sender_config_buffer_address);
        EXPECT_EQ(peer_desc.data_buffer_address, 0);  // Sender does not have a data buffer
        for (size_t i = 0; i < recv_chip_ids.size(); ++i) {
            EXPECT_EQ(peer_desc.chip_ids[i], sender_chip_ids[i]);
            EXPECT_EQ(peer_desc.mesh_ids[i], 2);
        }
        // Validate that the incorrect socket descriptor is rejected
        EXPECT_THROW(
            receive_and_verify_descriptor_from_peer(incorrect_socket_descriptor, SocketEndpoint::RECEIVER, context),
            std::runtime_error);
        forward_descriptor_to_peer(incorrect_socket_descriptor, SocketEndpoint::RECEIVER, context);
    }
}
