// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <ttnn/distributed/bidirectional_fabric_socket.hpp>

#include "distributed.hpp"

namespace ttml::autograd {
class Tensor;
}  // namespace ttml::autograd

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

enum class IntraMeshTransport {
    // Stream the tensor through the socket's DRAM FIFO (send_async / recv_async). A send that fits the FIFO completes
    // without its receiver.
    Fifo,
    // The sender writes straight into the receiver's tensor (send_direct_async / recv_direct_async). Faster, but the
    // caller has to pair the two ops itself, and the ops do not check the pairing:
    //  - A send completes only once its peer's recv has answered, and a recv only on its send's final token. An op
    //    without its peer never finishes, and stalls its device's command queue without an error.
    //  - So both tensors must be on the socket's mesh and cover every sending and receiving device of the
    //    connections. On one socket the k-th send pairs with the k-th recv; callers with equal parameters share it.
    //  - The sender lays the data out as in its own tensor: receive into a tensor allocated from the sent tensor's
    //    spec, or the data lands at the wrong addresses.
    Direct,
};

/**
 * @brief Parameters for intra-mesh communication (device-to-device within same host).
 * Supports multiple connections per socket.
 */
struct IntraMeshParameters {
    std::vector<tt::tt_metal::distributed::SocketConnection> connections;
    IntraMeshTransport transport = IntraMeshTransport::Fifo;

    bool operator==(const IntraMeshParameters&) const = default;
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

    // Receives a tensor's value, which the sender sends as its bf16 view (get_value()). A bf16 tensor receives it in
    // place. Any other tensor receives it into a new buffer, and assign() installs it in the tensor's stored dtype.
    void recv_value(ttml::autograd::Tensor& tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

    // New unified API supporting both inter-host and intra-mesh communication
    void send(
        const ttnn::Tensor& tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

    [[nodiscard]] ttnn::Tensor recv(
        ttnn::Tensor tensor,
        const InterHostParameters& inter_host_params,
        const IntraMeshParameters& intra_mesh_params);

    // Drops the intra-mesh sockets. They hold buffers of the mesh they were created on, so they must not outlive it.
    void clear_intra_mesh_sockets();

private:
    std::unique_ptr<ISocket> create_socket(std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);
    std::unique_ptr<ISocket> create_intra_mesh_socket(
        std::shared_ptr<DistributedContext> distributed_ctx, const IntraMeshParameters& params);
    ISocket* get_socket(Rank rank, std::shared_ptr<DistributedContext> distributed_ctx);
    ISocket* get_socket(const InterHostParameters& inter_host_params, const IntraMeshParameters& intra_mesh_params);

    SocketType m_type{SocketType::MPI};
    std::vector<std::unique_ptr<ttnn::distributed::ISocket>> m_inter_host_sockets;
    // An intra-mesh socket with the parameters it was created for, which is how it is found again.
    struct IntraMeshSocket {
        IntraMeshParameters params;
        std::unique_ptr<ISocket> socket;
    };
    std::vector<IntraMeshSocket> m_intra_mesh_sockets;
};

}  // namespace ttml::core::distributed
