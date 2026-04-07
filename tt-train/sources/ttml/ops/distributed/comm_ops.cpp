// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "comm_ops.hpp"

#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

// Check if a gradient tensor is replicated on the given cluster axis.
// If replicated, the backward CCL can skip the all_reduce (identity).
bool is_grad_replicated_on_axis(const tt::tt_metal::Tensor& grad, const std::optional<uint32_t> cluster_axis) {
    const auto& placements = grad.tensor_topology().placements();
    uint32_t axis = cluster_axis.value_or(0);
    if (axis >= placements.size()) {
        return true;  // default: treat as replicated
    }
    return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placements[axis]);
}

}  // namespace

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
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr all_gather(
    const autograd::TensorPtr& tensor,
    const int dim,
    const std::optional<uint32_t> cluster_axis,
    const GradOutputType grad_output_type) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_gather(tensor->get_value(), dim, cluster_axis));

    autograd::GradFunction grad = [tensor, out, dim, cluster_axis, grad_output_type]() {
        if (out->is_grad_initialized()) {
            auto reduced_grad = ttnn_fixed::distributed::reduce_scatter(out->get_grad(), dim, cluster_axis);
            if (grad_output_type == GradOutputType::SHARDED) {
                tensor->add_grad(reduced_grad);
            } else {
                auto* device = &autograd::ctx().get_device();
                auto mesh_shape = device->shape();
                uint32_t tp_size = cluster_axis.has_value() ? mesh_shape[cluster_axis.value()]
                                                            : static_cast<uint32_t>(device->num_devices());
                tensor->add_grad(ttnn::multiply(reduced_grad, 1.F / static_cast<float>(tp_size)));
            }
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr all_reduce(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::all_reduce(tensor->get_value(), cluster_axis));
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            const auto& g = out->get_grad();
            if (is_grad_replicated_on_axis(g, cluster_axis)) {
                tensor->add_grad(g);
            } else {
                tensor->add_grad(ttnn_fixed::distributed::all_reduce(g, cluster_axis));
            }
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            const auto& g = out->get_grad();
            if (is_grad_replicated_on_axis(g, cluster_axis)) {
                tensor->add_grad(g);
            } else {
                tensor->add_grad(ttnn_fixed::distributed::all_reduce(g, cluster_axis));
            }
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
