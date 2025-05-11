// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_metal::distributed {

// Create send/receive socket config buffers
std::shared_ptr<MeshBuffer> create_socket_config_buffer(
    const std::shared_ptr<MeshDevice>& device, const socket_config_t& config, bool is_sender);

// Create socket data buffer on receiver
std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    const std::shared_ptr<MeshDevice>& receiver, const socket_config_t& config);

// Write socket config data to allocated buffers
void write_socket_configs(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const std::shared_ptr<MeshBuffer>& peer_config_buffer,
    const std::shared_ptr<MeshBuffer>& socket_data_buffer,
    const socket_config_t& config,
    bool is_sender);

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
