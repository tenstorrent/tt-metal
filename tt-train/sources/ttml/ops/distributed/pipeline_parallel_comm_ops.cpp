#pragma once

#include "pipeline_parallel_comm_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr intermesh_send(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank) {
    auto& socket_manager = ttml::autograd::ctx().get_socket_manager();
    auto out = autograd::create_tensor(tensor->get_value());
    socket_manager.send(tensor->get_value(), tensor->get_distributed_context(), rank);
    auto grad = [tensor, &socket_manager]() {
        auto grad = ttnn::empty_like(tensor->get_value());
        grad = socket_manager.recv(grad, tensor->get_distributed_context(), rank);
        tensor->add_grad(grad);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr intermesh_recv(const autograd::TensorPtr& tensor, ttml::core::distributed::Rank rank) {
    auto& socket_manager = ttml::autograd::ctx().get_socket_manager();
    auto empty_like = ttnn::empty_like(tensor->get_value());
    empty_like = socket_manager.recv(empty_like, tensor->get_distributed_context(), rank);
    auto out = autograd::create_tensor(empty_like);
    auto grad = [out, &socket_manager]() {
        socket_manager.send(out->get_grad(), tensor->get_distributed_context(), rank);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops::distributed
