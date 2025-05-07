// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_metal::distributed {

// Create send/receive socket config buffers
std::shared_ptr<MeshBuffer> create_sender_socket_config_buffer(
    std::shared_ptr<MeshDevice> sender, const socket_config_t& config);
std::shared_ptr<MeshBuffer> create_receiver_socket_config_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config);

// Create socket data buffer on receiver
std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config);

// Write socket config data to allocated buffers
void populate_sender_socket_config_buffer(
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config);
void populate_receiver_socket_config_buffer(
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config);

//  =============== Additional utility functions  ===============

// Get the Fabric Encoding used by sender/receiver to hanshake with upstream/downstream chips
uint32_t get_sender_receiver_chip_fabric_encoding(
    MeshDevice* sender_device,
    MeshDevice* recv_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& recv_coord,
    FabricConfig fabric_config,
    bool get_sender_encoding);

// Given a MeshDevice and a logical device coordinate, determine the device's physical mesh id
uint32_t get_physical_mesh_id(MeshDevice* mesh_device, const MeshCoordinate& coord);

}  // namespace tt::tt_metal::distributed
