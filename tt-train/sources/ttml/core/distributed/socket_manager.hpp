// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <ttnn/distributed/bidirectional_fabric_socket.hpp>

#include "distributed.hpp"

namespace ttml::core::distributed {

using ttnn::distributed::EndpointSocketType;
using ttnn::distributed::ISocket;
using ttnn::distributed::SocketType;

/**
 * @brief Parameters for inter-host communication (between different hosts/ranks).
 */
struct InterHostParameters {
    std::shared_ptr<DistributedContext> distributed_ctx;
    Rank rank;
};

/**
 * @brief Parameters for intra-mesh communication (device-to-device within same host).
 * Supports multiple connections per socket.
 */
struct IntraMeshParameters {
    std::vector<tt::tt_metal::distributed::SocketConnection> connections;
};

class SocketManager {
public:
    explicit SocketManager(SocketType type);
    SocketManager(SocketManager&&) = delete;
    SocketManager(const SocketManager&) = delete;
    SocketManager& operator=(const SocketManager&) = delete;
    SocketManager& operator=(SocketManager&&) = delete;
    ~SocketManager() = default;

    // [[deprecated("Use unified API instead")]]
    void send(const ttnn::Tensor& tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

    // [[deprecated("Use unified API instead")]]
    [[nodiscard]] ttnn::Tensor recv(
        ttnn::Tensor tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

    // New unified API supporting both inter-host and intra-mesh communication
    void send(
        const ttnn::Tensor& tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

    [[nodiscard]] ttnn::Tensor recv(
        ttnn::Tensor tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

    // Intra-mesh only. The sender writes straight into the receiver's tensor
    // over fabric; the socket carries only the address handshake and the
    // completion token, so it is a different socket from the FIFO one (a
    // small L1 FIFO instead of an 80 MB DRAM one) and is cached separately.
    // One fabric packet stream per connection, so more connections means
    // more links in parallel -- up to the number of links between the two
    // chips, which the op checks.
    void send_direct(
        const ttnn::Tensor& tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

    [[nodiscard]] ttnn::Tensor recv_direct(
        ttnn::Tensor tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

private:
    struct DirectSocketPair {
        std::shared_ptr<DistributedContext> distributed_ctx;
        std::vector<tt::tt_metal::distributed::SocketConnection> connections;
        tt::tt_metal::distributed::MeshSocket send_socket;
        tt::tt_metal::distributed::MeshSocket recv_socket;
    };
    DirectSocketPair& get_direct_socket(
        const InterHostParameters& inter_host_params, const IntraMeshParameters& intra_mesh_params);

    std::unique_ptr<ISocket> create_socket(std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);
    std::unique_ptr<ttnn::distributed::BidirectionalFabricSocket> create_intra_mesh_socket(
        std::shared_ptr<DistributedContext> distributed_ctx, const IntraMeshParameters& params);
    ISocket* get_socket(Rank rank, std::shared_ptr<DistributedContext> distributed_ctx);
    ISocket* get_socket(const InterHostParameters& inter_host_params, const IntraMeshParameters& intra_mesh_params);

    SocketType m_type{SocketType::MPI};
    std::vector<std::unique_ptr<ttnn::distributed::ISocket>> m_inter_host_sockets;
    std::vector<std::unique_ptr<ttnn::distributed::BidirectionalFabricSocket>> m_intra_mesh_sockets;
    std::vector<std::unique_ptr<DirectSocketPair>> m_intra_mesh_direct_sockets;
};

}  // namespace ttml::core::distributed
