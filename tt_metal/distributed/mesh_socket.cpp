// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::distributed {

std::pair<mesh_socket_t, mesh_socket_t> mesh_socket_t::create_sockets(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const socket_config_t& config) {
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    auto sender_config_buffer = create_socket_config_buffer(sender, config, true /* is_sender */);
    auto recv_config_buffer = create_socket_config_buffer(receiver, config, false /* is_sender */);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);
    write_socket_configs(sender_config_buffer, recv_config_buffer, socket_data_buffer, config, true /* is_sender */);
    write_socket_configs(recv_config_buffer, sender_config_buffer, socket_data_buffer, config, false /* is_sender */);
    auto sender_socket = mesh_socket_t(
        nullptr,  // The sender socket does not have a data-buffer allocated
        sender_config_buffer,
        config);
    auto receiver_socket = mesh_socket_t(socket_data_buffer, recv_config_buffer, config);
    return {sender_socket, receiver_socket};
}

std::shared_ptr<MeshBuffer> mesh_socket_t::get_data_buffer() const {
    TT_FATAL(data_buffer_, "Cannot access the data buffer for a sender socket.");
    return data_buffer_;
};

std::shared_ptr<MeshBuffer> mesh_socket_t::get_config_buffer() const { return config_buffer_; }

const socket_config_t& mesh_socket_t::get_physical_config() const { return physical_config_; }

}  // namespace tt::tt_metal::distributed
