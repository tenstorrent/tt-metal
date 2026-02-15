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

autograd::TensorPtr reduce_scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::reduce_scatter(tensor->get_value(), dim, cluster_axis));
    /* d(x_0 + x_1 + ... + x_n) / dx_i = 1 for i=0,1,...,n and 0 otherwise */
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto* device = &autograd::ctx().get_device();
    auto mesh_shape = device->shape();
    uint32_t tp_size =
        cluster_axis.has_value() ? mesh_shape[cluster_axis.value()] : static_cast<uint32_t>(device->num_devices());

    auto scattered = ttnn_fixed::distributed::reduce_scatter(tensor->get_value(), dim, cluster_axis);
    /* average across TP as input is assumed to be replicated across TP axis*/
    auto out = autograd::create_tensor(ttnn::multiply(scattered, 1.F / static_cast<float>(tp_size)));

    /* input is replicated across TP axis, so d(nx/n) / dx = dx / dx = 1 for i=0,1,...,n and 0 otherwise */
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_gather(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_gather(tensor->get_value(), dim, cluster_axis));

    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::reduce_scatter(out->get_grad(), dim, cluster_axis));
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr all_reduce(
    const autograd::TensorPtr& tensor, const bool noop_backward, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_reduce(tensor->get_value(), cluster_axis));
    autograd::GradFunction grad = [tensor, out, noop_backward, cluster_axis]() {
        if (out->is_grad_initialized()) {
            if (noop_backward) {
                tensor->add_grad(out->get_grad());
            } else {
                tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad(), cluster_axis));
            }
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad(), cluster_axis));
        }
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr ring_shift(
    const autograd::TensorPtr& tensor,
    const std::optional<uint32_t> cluster_axis,
    const ttnn_fixed::distributed::RingShiftDirection direction) {
    // Forward pass: shift in the specified direction
    auto out =
        autograd::create_tensor(ttnn_fixed::distributed::ring_shift(tensor->get_value(), cluster_axis, direction));

    // Backward pass: shift in the opposite direction to route gradients back
    const auto opposite_direction = (direction == ttnn_fixed::distributed::RingShiftDirection::Forward)
                                        ? ttnn_fixed::distributed::RingShiftDirection::Backward
                                        : ttnn_fixed::distributed::RingShiftDirection::Forward;
    autograd::GradFunction grad = [tensor, out, cluster_axis, opposite_direction]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::ring_shift(out->get_grad(), cluster_axis, opposite_direction));
        }
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops::distributed
