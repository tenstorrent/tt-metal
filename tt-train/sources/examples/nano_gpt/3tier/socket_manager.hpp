// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/distributed/distributed.hpp"

using ttnn::distributed::EndpointSocketType;
using ttnn::distributed::ISocket;
using ttnn::distributed::SocketType;

class SocketManager {
public:
    SocketManager(SocketType type);

    void send(
        const ttnn::Tensor& tensor,
        std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx,
        ttml::core::distributed::Rank rank);

    void recv(
        ttnn::Tensor& tensor,
        std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx,
        ttml::core::distributed::Rank rank);

private:
    ISocket* get_socket(
        ttml::core::distributed::Rank rank,
        std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx);

    std::unique_ptr<ISocket> create_socket(
        std::shared_ptr<ttml::core::distributed::DistributedContext> distributed_ctx,
        ttml::core::distributed::Rank rank);

    SocketType m_type{SocketType::MPI};
    std::vector<std::unique_ptr<ttnn::distributed::ISocket>> m_sockets;
};
