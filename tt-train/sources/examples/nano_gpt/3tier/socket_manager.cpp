// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_manager.hpp"

namespace {

void _generate_fabric_socket_config(
    tt::tt_metal::distributed::SocketConfig& socket_config,
    const std::shared_ptr<ttnn::distributed::MeshDevice>& mesh_device) {
    auto socket_mem_config = tt::tt_metal::distributed::SocketMemoryConfig{};
    socket_mem_config.socket_storage_type = ttnn::BufferType::DRAM;
    socket_mem_config.fifo_size = 32U * 32U * 2U * 8U;  // 16K, 8 bfloat16 tiles
    socket_config.socket_mem_config = socket_mem_config;

    std::vector<tt::tt_metal::distributed::SocketConnection> socket_connection_config;
    auto mesh_rows = mesh_device->num_rows();
    auto mesh_cols = mesh_device->num_cols();
    for (size_t row = 0; row < mesh_rows; ++row) {
        for (size_t col = 0; col < mesh_cols; ++col) {
            tt::tt_metal::distributed::MeshCoreCoord mesh_core_coord{
                tt::tt_metal::distributed::MeshCoordinate{static_cast<uint32_t>(row), static_cast<uint32_t>(col)},
                ttnn::CoreCoord{0, 0}};
            socket_connection_config.emplace_back(mesh_core_coord, mesh_core_coord);
        }
    }

    socket_config.socket_connection_config = std::move(socket_connection_config);
}

}  // namespace

SocketManager::SocketManager(SocketType type) : m_type(type) {
}

void SocketManager::send(
    const ttnn::Tensor& tensor,
    std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx,
    ttml::core::distributed::Rank rank) {
    auto socket = get_socket(rank, distributed_ctx);
    socket->send(tensor);
}

void SocketManager::recv(
    ttnn::Tensor& tensor,
    std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx,
    ttml::core::distributed::Rank rank) {
    auto socket = get_socket(rank, distributed_ctx);
    socket->recv(tensor);
}

ISocket* SocketManager::get_socket(
    ttml::core::distributed::Rank rank, std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx) {
    for (const auto& socket : m_sockets) {
        // TODO: modify rank check based on the type of socket
        bool is_same_rank = (socket->get_sender_rank() == rank || socket->get_receiver_rank() == rank);
        if (is_same_rank && socket->get_distributed_context() == distributed_ctx) {
            return socket.get();
        }
    }

    auto socket = create_socket(distributed_ctx, rank);
    assert(socket != nullptr && "Failed to create socket");
    m_sockets.push_back(std::move(socket));
    return m_sockets.back().get();
}

std::unique_ptr<ISocket> SocketManager::create_socket(
    std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx, ttml::core::distributed::Rank rank) {
    auto mesh_device = ttml::autograd::ctx().get_shared_ptr_device();
    if (m_type == SocketType::MPI) {
        auto socket_config = tt::tt_metal::distributed::SocketConfig{};
        socket_config.distributed_context = distributed_ctx;
        return ttnn::distributed::create_socket(
            SocketType::MPI, ttnn::distributed::EndpointSocketType::BIDIRECTIONAL, mesh_device, rank, socket_config);
    } else {
        auto socket_config = tt::tt_metal::distributed::SocketConfig{};
        socket_config.distributed_context = distributed_ctx;
        _generate_fabric_socket_config(socket_config, mesh_device);
        return ttnn::distributed::create_socket(
            SocketType::FABRIC, ttnn::distributed::EndpointSocketType::BIDIRECTIONAL, mesh_device, rank, socket_config);
    }
}
