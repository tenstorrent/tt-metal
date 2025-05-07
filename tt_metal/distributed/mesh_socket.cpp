// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::distributed {

std::pair<mesh_socket_t, mesh_socket_t> create_sockets(
    std::shared_ptr<MeshDevice> sender, std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    auto sender_config_buffer = create_sender_socket_config_buffer(sender, config);
    auto recv_config_buffer = create_receiver_socket_config_buffer(receiver, config);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);
    populate_sender_socket_config_buffer(sender_config_buffer, recv_config_buffer, socket_data_buffer, config);
    populate_receiver_socket_config_buffer(recv_config_buffer, sender_config_buffer, socket_data_buffer, config);
    auto sender_socket = mesh_socket_t{
        .data_buffer = socket_data_buffer,
        .config_buffer = sender_config_buffer,
    };
    auto receiver_socket = mesh_socket_t{
        .data_buffer = socket_data_buffer,
        .config_buffer = recv_config_buffer,
    };
    return {sender_socket, receiver_socket};
}

}  // namespace tt::tt_metal::distributed
