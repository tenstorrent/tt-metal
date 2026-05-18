// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "comm_ops.hpp"

#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

ttnn::Tensor local_scatter(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto* device = &autograd::ctx().get_device();
    const auto mesh_shape = device->shape();
    const uint32_t num_parts =
        cluster_axis.has_value() ? mesh_shape[cluster_axis.value()] : static_cast<uint32_t>(device->num_devices());

    const auto tensor_shape = tensor.logical_shape();
    const int rank = static_cast<int>(tensor_shape.rank());
    const int normalized_dim = dim < 0 ? dim + rank : dim;
    TT_FATAL(
        normalized_dim >= 0 && normalized_dim < rank, "local_scatter: dim {} is out of range for rank {}", dim, rank);

    const uint32_t scatter_dim_size = tensor_shape[normalized_dim];
    TT_FATAL(
        scatter_dim_size % num_parts == 0U,
        "local_scatter: tensor dimension {} size {} must be divisible by scatter parts {}",
        normalized_dim,
        scatter_dim_size,
        num_parts);

    const uint32_t local_dim_size = scatter_dim_size / num_parts;
    std::vector<uint32_t> start_indices;
    std::vector<uint32_t> end_indices;
    start_indices.reserve(num_parts * static_cast<uint32_t>(rank));
    end_indices.reserve(num_parts * static_cast<uint32_t>(rank));

    for (uint32_t part = 0; part < num_parts; ++part) {
        for (int axis = 0; axis < rank; ++axis) {
            if (axis == normalized_dim) {
                start_indices.push_back(part * local_dim_size);
                end_indices.push_back((part + 1U) * local_dim_size);
            } else {
                start_indices.push_back(0U);
                end_indices.push_back(tensor_shape[axis]);
            }
        }
    }

    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, cluster_axis);
    const auto indices_shape = ttnn::Shape({num_parts * static_cast<uint32_t>(rank)});
    auto start_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        start_indices, indices_shape, device, ttnn::Layout::ROW_MAJOR, mapper.get());
    auto end_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        end_indices, indices_shape, device, ttnn::Layout::ROW_MAJOR, mapper.get());
    const auto step = ttnn::SmallVector<uint32_t>(static_cast<size_t>(rank), 1U);

    return ttnn::slice<uint32_t>(
        tensor,
        start_tensor,
        end_tensor,
        step,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        static_cast<uint32_t>(normalized_dim),
        num_parts);
}

}  // namespace

autograd::TensorPtr reduce_scatter(
    const autograd::TensorPtr& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(ttnn_fixed::distributed::reduce_scatter(tensor->get_value(), dim, cluster_axis));
    /* d(x_0 + x_1 + ... + x_n) / dx_i = 1 for i=0,1,...,n and 0 otherwise */
    autograd::GradFunction grad = [tensor, out, dim, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_gather(out->get_grad(), dim, cluster_axis));
            out->deallocate_value();
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
            out->deallocate_value();
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
            if (grad_output_type == GradOutputType::SHARDED) {
                auto reduced_grad = ttnn_fixed::distributed::reduce_scatter(out->get_grad(), dim, cluster_axis);
                tensor->add_grad(reduced_grad);
            } else {
                tensor->add_grad(local_scatter(out->get_grad(), dim, cluster_axis));
            }
            out->deallocate_value();
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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
            out->deallocate_value();
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast(const autograd::TensorPtr& tensor, const std::optional<uint32_t> cluster_axis) {
    auto out = autograd::create_tensor(tensor->get_value());
    autograd::GradFunction grad = [tensor, out, cluster_axis]() {
        if (out->is_grad_initialized()) {
            tensor->add_grad(ttnn_fixed::distributed::all_reduce(out->get_grad(), cluster_axis));
            out->deallocate_value();
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
            out->deallocate_value();
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops::distributed
