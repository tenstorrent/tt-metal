// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_manager.hpp"

#include "autograd/auto_context.hpp"

namespace {

tt::tt_metal::distributed::SocketMemoryConfig _make_socket_mem_config() {
    tt::tt_metal::distributed::SocketMemoryConfig socket_mem_config{};
    socket_mem_config.socket_storage_type = ttnn::BufferType::DRAM;
    // bandwidth-delay product is roughly 10GB/s * 1us = 10MB
    socket_mem_config.fifo_size = 10U * 1024U * 1024U;  // 10MB
    return socket_mem_config;
}

std::vector<tt::tt_metal::distributed::SocketConnection> _make_socket_connection_config(
    const std::shared_ptr<ttnn::distributed::MeshDevice>& mesh_device) {
    auto mesh_rows = mesh_device->num_rows();
    auto mesh_cols = mesh_device->num_cols();
    std::vector<tt::tt_metal::distributed::SocketConnection> socket_connection_config;
    socket_connection_config.reserve(mesh_rows * mesh_cols);
    // TODO(rfurko): remove hardcoded values
    for (size_t row = 0; row < mesh_rows; ++row) {
        for (size_t col = 0; col < mesh_cols; ++col) {
            tt::tt_metal::distributed::MeshCoreCoord mesh_core_coord{
                tt::tt_metal::distributed::MeshCoordinate{static_cast<uint32_t>(row), static_cast<uint32_t>(col)},
                ttnn::CoreCoord{0, 0}};
            socket_connection_config.emplace_back(mesh_core_coord, mesh_core_coord);
        }
    }
    return socket_connection_config;
}

void _generate_fabric_socket_config(
    tt::tt_metal::distributed::SocketConfig& socket_config,
    const std::shared_ptr<ttnn::distributed::MeshDevice>& mesh_device) {
    socket_config.socket_mem_config = _make_socket_mem_config();
    socket_config.socket_connection_config = _make_socket_connection_config(mesh_device);
}

}  // namespace

namespace ttml::core::distributed {

SocketManager::SocketManager(SocketType type) : m_type(type) {
}

// TODO: Remove legacy send/recv/get_socket when revisiting pipeline parallelism - use unified API instead
void SocketManager::send(const ttnn::Tensor& tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank) {
    auto socket = get_socket(rank, distributed_ctx);
    socket->send(tensor);
}

ttnn::Tensor SocketManager::recv(ttnn::Tensor tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank) {
    auto socket = get_socket(rank, distributed_ctx);
    socket->recv(tensor);
    return tensor;
}

ISocket* SocketManager::get_socket(Rank rank, std::shared_ptr<DistributedContext> distributed_ctx) {
    for (const auto& socket : m_inter_host_sockets) {
        if (socket->get_rank() == rank && socket->get_distributed_context() == distributed_ctx) {
            return socket.get();
        }
    }

    auto socket = create_socket(distributed_ctx, rank);
    assert(socket != nullptr && "Failed to create socket");
    m_inter_host_sockets.push_back(std::move(socket));
    return m_inter_host_sockets.back().get();
}

// New unified API supporting both inter-host and intra-mesh communication
void SocketManager::send(
    const ttnn::Tensor& tensor,
    const InterHostParameters& inter_host_params,
    const IntraMeshParameters& intra_mesh_params) {
    auto socket = get_socket(inter_host_params, intra_mesh_params);
    socket->send(tensor);
}

ttnn::Tensor SocketManager::recv(
    ttnn::Tensor tensor, const InterHostParameters& inter_host_params, const IntraMeshParameters& intra_mesh_params) {
    auto socket = get_socket(inter_host_params, intra_mesh_params);
    socket->recv(tensor);
    return tensor;
}

ISocket* SocketManager::get_socket(
    const InterHostParameters& inter_host_params, const IntraMeshParameters& intra_mesh_params) {
    bool is_intra_mesh = (inter_host_params.distributed_ctx->rank() == inter_host_params.rank);

    if (is_intra_mesh) {
        // Lookup existing intra-mesh socket by connection config and distributed context
        for (const auto& socket : m_intra_mesh_sockets) {
            if (socket->get_rank() == inter_host_params.rank &&
                socket->get_distributed_context() == inter_host_params.distributed_ctx &&
                socket->get_socket_connections() == intra_mesh_params.connections) {
                return socket.get();
            }
        }

        // Create new intra-mesh socket
        auto socket = create_intra_mesh_socket(inter_host_params.distributed_ctx, intra_mesh_params);
        assert(socket != nullptr && "Failed to create intra-mesh socket");
        m_intra_mesh_sockets.push_back(std::move(socket));
        return m_intra_mesh_sockets.back().get();
    } else {
        // Lookup existing inter-host socket by rank and distributed context
        for (const auto& socket : m_inter_host_sockets) {
            if (socket->get_rank() == inter_host_params.rank &&
                socket->get_distributed_context() == inter_host_params.distributed_ctx) {
                return socket.get();
            }
        }

        // Create new inter-host socket
        // TODO: accept IntraMeshConfig as well and use connections from it if available
        auto socket = create_socket(inter_host_params.distributed_ctx, inter_host_params.rank);
        assert(socket != nullptr && "Failed to create inter-host socket");
        m_inter_host_sockets.push_back(std::move(socket));
        return m_inter_host_sockets.back().get();
    }
}

std::unique_ptr<ISocket> SocketManager::create_socket(std::shared_ptr<DistributedContext> distributed_ctx, Rank rank) {
    auto mesh_device = ttml::autograd::ctx().get_device_ptr();

    auto socket_config = tt::tt_metal::distributed::SocketConfig{};
    socket_config.distributed_context = distributed_ctx;

    if (m_type != SocketType::MPI) {
        _generate_fabric_socket_config(socket_config, mesh_device);
    }

    return ttnn::distributed::create_socket(
        m_type, ttnn::distributed::EndpointSocketType::BIDIRECTIONAL, mesh_device, rank, socket_config);
}

std::unique_ptr<ttnn::distributed::BidirectionalFabricSocket> SocketManager::create_intra_mesh_socket(
    std::shared_ptr<DistributedContext> distributed_ctx, const IntraMeshParameters& params) {
    auto mesh_device = ttml::autograd::ctx().get_device_ptr();

    const auto socket_mem_config = _make_socket_mem_config();
    tt::tt_metal::distributed::SocketConfig socket_config(params.connections, socket_mem_config);
    socket_config.distributed_context = distributed_ctx;

    // Use create_socket_pair for intra-mesh communication
    const auto [send_socket, recv_socket] =
        tt::tt_metal::distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config);

    // Directly construct BidirectionalFabricSocket with the socket pair
    return std::make_unique<ttnn::distributed::BidirectionalFabricSocket>(send_socket, recv_socket);
}

}  // namespace ttml::core::distributed
