// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "socket_manager.hpp"

#include <algorithm>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_direct_async/recv_direct_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_direct_async/send_direct_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/send_recv_utils.hpp"

namespace {

// A sender/receiver socket pair between devices of one mesh that uses the direct socket ops. The tensor does not
// pass through the socket FIFO: the sender writes it straight into the receiver's tensor. Only the sender's two
// handshake pages go through the FIFO: its address before the data, a completion token after it. The ops require
// that FIFO in L1, because the receiver reads the handshake page in place.
class DirectFabricSocket final : public ttnn::distributed::ISocket {
public:
    // Two handshake pages: at most the previous op's token and this op's address are outstanding at once. The page
    // size follows the ops, which align it to the tensor's buffer; L1 and DRAM cover the tensors tt-train sends.
    // The socket allocates this much L1 on every worker core of every device, for as long as it lives.
    static uint32_t fifo_size(const ttnn::distributed::MeshDevice& mesh_device) {
        const auto& allocator = mesh_device.allocator();
        const uint32_t alignment = std::max(
            allocator->get_alignment(ttnn::BufferType::L1), allocator->get_alignment(ttnn::BufferType::DRAM));
        return 2U * ttnn::send_recv_utils::handshake_page_size(alignment);
    }

    DirectFabricSocket(
        const tt::tt_metal::distributed::MeshSocket& send_socket,
        const tt::tt_metal::distributed::MeshSocket& recv_socket) :
        m_send_socket(send_socket), m_recv_socket(recv_socket) {
    }

    void send(const ttnn::Tensor& tensor) override {
        ttnn::experimental::send_direct_async(tensor, m_send_socket);
    }
    void recv(ttnn::Tensor& tensor) override {
        ttnn::experimental::recv_direct_async(tensor, m_recv_socket);
    }

    // ISocket::get_rank is the rank of the host at the other end. Both ends of this socket are on this host.
    tt::tt_metal::distributed::multihost::Rank get_rank() const override {
        return get_distributed_context()->rank();
    }
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context()
        const override {
        return m_send_socket.get_config().distributed_context;
    }

private:
    tt::tt_metal::distributed::MeshSocket m_send_socket;
    tt::tt_metal::distributed::MeshSocket m_recv_socket;
};

tt::tt_metal::distributed::SocketMemoryConfig _make_socket_mem_config() {
    tt::tt_metal::distributed::SocketMemoryConfig socket_mem_config{};
    socket_mem_config.socket_storage_type = ttnn::BufferType::DRAM;
    // Bandwidth-delay product is roughly 10GB/s * 1us = 10MB; use an 8x safety margin and allocate 80MB.
    socket_mem_config.fifo_size = 10U * 1024U * 1024U * 2U * 4U;  // 80MB
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

void SocketManager::clear_intra_mesh_sockets() {
    m_intra_mesh_sockets.clear();
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

void SocketManager::recv_value(
    ttml::autograd::Tensor& tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank) {
    if (tensor.get_value(ttml::autograd::PreferredPrecision::NATIVE).dtype() == ttnn::DataType::BFLOAT16) {
        auto value = tensor.get_value_for_update();
        (void)recv(value.tensor(), std::move(distributed_ctx), rank);
    } else {
        tensor.assign(recv(ttnn::empty_like(tensor.get_value()), std::move(distributed_ctx), rank));
    }
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
        // Lookup existing intra-mesh socket by its parameters and distributed context
        for (const auto& [params, socket] : m_intra_mesh_sockets) {
            if (socket->get_distributed_context() == inter_host_params.distributed_ctx &&
                params == intra_mesh_params) {
                return socket.get();
            }
        }

        // Create new intra-mesh socket
        auto socket = create_intra_mesh_socket(inter_host_params.distributed_ctx, intra_mesh_params);
        assert(socket != nullptr && "Failed to create intra-mesh socket");
        m_intra_mesh_sockets.push_back({intra_mesh_params, std::move(socket)});
        return m_intra_mesh_sockets.back().socket.get();
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

std::unique_ptr<ISocket> SocketManager::create_intra_mesh_socket(
    std::shared_ptr<DistributedContext> distributed_ctx, const IntraMeshParameters& params) {
    auto mesh_device = ttml::autograd::ctx().get_device_ptr();

    const bool direct = params.transport == IntraMeshTransport::Direct;
    const auto socket_mem_config =
        direct ? tt::tt_metal::distributed::SocketMemoryConfig(
                     ttnn::BufferType::L1, DirectFabricSocket::fifo_size(*mesh_device))
               : _make_socket_mem_config();
    tt::tt_metal::distributed::SocketConfig socket_config(params.connections, socket_mem_config);
    socket_config.distributed_context = distributed_ctx;

    // Use create_socket_pair for intra-mesh communication
    const auto [send_socket, recv_socket] =
        tt::tt_metal::distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config);

    if (direct) {
        return std::make_unique<DirectFabricSocket>(send_socket, recv_socket);
    }
    return std::make_unique<ttnn::distributed::BidirectionalFabricSocket>(send_socket, recv_socket);
}

}  // namespace ttml::core::distributed
