// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_parallel_comm_ops.hpp"

#include "autograd/graph_utils.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr intermesh_send(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank) {
    auto& socket_manager = ttml::autograd::ctx().get_socket_manager();
    auto distributed_ctx = ttml::autograd::ctx().get_distributed_context();
    auto out = autograd::create_tensor(tensor->get_value());
    socket_manager.send(tensor->get_value(), distributed_ctx, rank);
    auto grad = [tensor, &socket_manager, distributed_ctx, rank]() {
        auto grad = ttnn::empty_like(tensor->get_value());
        grad = socket_manager.recv(grad, distributed_ctx, rank);
        tensor->add_grad(grad);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr intermesh_recv(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank) {
    auto& socket_manager = ttml::autograd::ctx().get_socket_manager();
    auto distributed_ctx = ttml::autograd::ctx().get_distributed_context();
    auto empty_like = ttnn::empty_like(tensor->get_value());
    empty_like = socket_manager.recv(empty_like, distributed_ctx, rank);
    auto out = autograd::create_tensor(empty_like);
    auto grad = [out, &socket_manager, distributed_ctx, rank]() {
        socket_manager.send(out->get_grad(), distributed_ctx, rank);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops::distributed
