// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "distributed.hpp"

namespace ttml::core::distributed {

using ttnn::distributed::EndpointSocketType;
using ttnn::distributed::ISocket;
using ttnn::distributed::SocketType;

class SocketManager {
public:
    explicit SocketManager(SocketType type);
    SocketManager(SocketManager&&) = delete;
    SocketManager(const SocketManager&) = delete;
    SocketManager& operator=(const SocketManager&) = delete;
    SocketManager& operator=(SocketManager&&) = delete;
    ~SocketManager() = default;

    void send(const ttnn::Tensor& tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

    [[nodiscard]] ttnn::Tensor recv(
        ttnn::Tensor tensor, std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

private:
    ISocket* get_socket(Rank rank, std::shared_ptr<DistributedContext> distributed_ctx);

    std::unique_ptr<ISocket> create_socket(std::shared_ptr<DistributedContext> distributed_ctx, Rank rank);

    SocketType m_type{SocketType::MPI};
    std::vector<std::unique_ptr<ttnn::distributed::ISocket>> m_sockets;
};

}  // namespace ttml::core::distributed
