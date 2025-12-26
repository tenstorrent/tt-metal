// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "comm_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr reduce_scatter(const autograd::TensorPtr& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::reduce_scatter(tensor->get_value(), dim, cluster_axis));
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_gather(const autograd::TensorPtr& tensor, int dim, std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_gather(tensor->get_value(), dim, cluster_axis));
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        tensor->add_grad(ttnn_fixed::distributed::reduce_scatter(out->get_grad(), dim, cluster_axis));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor, bool noop_backward, std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_reduce(tensor->get_value(), cluster_axis));
    autograd::GradFunction grad = [tensor, out, noop_backward, cluster_axis]() {
        if (noop_backward) {
            tensor->add_grad(out->get_grad());
        } else {
            tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad(), cluster_axis));
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad(), cluster_axis));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops::distributed
