// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "comm_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr scatter(const autograd::TensorPtr& tensor, int dim) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::scatter(tensor->get_value(), dim));
    autograd::GradFunction grad = [tensor, out, dim]() { tensor->add_grad(ttnn::all_gather(out->get_grad(), dim)); };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_gather(const autograd::TensorPtr& tensor, int dim) {
    auto out = autograd::create_tensor(ttnn::all_gather(tensor->get_value(), dim));
    autograd::GradFunction grad = [tensor, out, dim]() {
        tensor->add_grad(ttnn_fixed::distributed::scatter(out->get_grad(), dim));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_reduce(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() { tensor->add_grad(out->get_grad()); };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out]() {
        tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad()));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops::distributed
