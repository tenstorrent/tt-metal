// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "comm_ops.hpp"

#include <algorithm>
#include <variant>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/distributed/distributed.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// The gradient of a tensor that is replicated across `cluster_axis` (the output of all_reduce, broadcast, all_gather):
// one logical value, whose gradient is whole where the ranks' gradients agree, and is their sum where they differ (each
// rank then holds a partial sum, e.g. from a sharded matmul).
bool is_partial(const ttnn::Tensor& grad, const std::optional<uint32_t> cluster_axis) {
    if (cluster_axis.has_value()) {
        return core::distributed::is_sharded_on_axis(grad, *cluster_axis);
    }
    const auto placements = core::distributed::placements_per_axis(grad);
    return std::any_of(placements.begin(), placements.end(), [](const auto& placement) {
        return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placement);
    });
}

ttnn::Tensor whole_gradient(const ttnn::Tensor& grad, const std::optional<uint32_t> cluster_axis) {
    return is_partial(grad, cluster_axis) ? ttnn_fixed::distributed::all_reduce(grad, cluster_axis) : grad;
}

}  // namespace

autograd::TensorPtr reduce_scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::reduce_scatter(tensor->get_value(), dim, cluster_axis));
    /* d(x_0 + x_1 + ... + x_n) / dx_i = 1 for i=0,1,...,n and 0 otherwise */
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::mesh_partition(tensor->get_value(), dim, cluster_axis));
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr all_gather(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_gather(tensor->get_value(), dim, cluster_axis));

    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            // Each rank's slice of the whole gradient: summed and sliced in one collective when the gradient is
            // partial, sliced locally when it is whole.
            const auto& out_grad = out->get_grad();
            tensor->add_grad(
                is_partial(out_grad, cluster_axis)
                    ? ttnn_fixed::distributed::reduce_scatter(out_grad, dim, cluster_axis)
                    : ttnn_fixed::distributed::mesh_partition(out_grad, dim, cluster_axis));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_reduce(tensor->get_value(), cluster_axis));
    // d(x_0 + x_1 + ... + x_n) / dx_i = 1: every summand's gradient is the sum's.
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(whole_gradient(out->get_grad(), cluster_axis));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(whole_gradient(out->get_grad(), cluster_axis));
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops::distributed
