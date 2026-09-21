// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::ring_shift_fused {

struct operation_attributes_t {
    // The sockets of the ring: sender-side and receiver-side views, pairwise.
    // Every chip sends through one socket and receives through another (a
    // chip that is both a sender and a receiver of one socket hangs the
    // fabric handshake, so the ring is two sockets: even chips to odd and
    // odd to even, as the two-phase shift has them). A constructor, not an
    // aggregate: the op framework's reflection would otherwise try to
    // default-construct the sockets, which have no such constructor.
    const std::vector<tt::tt_metal::distributed::MeshSocket> send_sockets;
    const std::vector<tt::tt_metal::distributed::MeshSocket> recv_sockets;

    operation_attributes_t(
        std::vector<tt::tt_metal::distributed::MeshSocket> send, std::vector<tt::tt_metal::distributed::MeshSocket> recv) :
        send_sockets(std::move(send)), recv_sockets(std::move(recv)) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        for (size_t i = 0; i < send_sockets.size(); ++i) {
            attrs.emplace_back("send_socket_" + std::to_string(i), send_sockets[i]);
            attrs.emplace_back("recv_socket_" + std::to_string(i), recv_sockets[i]);
        }
        return attrs;
    }
};

struct tensor_args_t {
    std::vector<ttnn::Tensor> inputs;
    std::vector<ttnn::Tensor> preallocated_outputs;  // empty, or one per input
};

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::ring_shift_fused
