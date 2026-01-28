// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <optional>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_value(), mem_config);
        tensor->add_grad(res[0]);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        static const std::string approx_mode = "none";
        auto dL_dt = ttnn::experimental::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode);
        tensor->add_grad(dL_dt);
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw) {
    auto out = autograd::create_tensor(ttnn::silu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out, use_composite_bw]() {
        auto res = use_composite_bw ? ttnn::silu_bw(out->get_grad(), tensor->get_value())
                                    : std::vector<std::optional<ttnn::Tensor>>(
                                          {ttml::metal::silu_bw(tensor->get_value(), out->get_grad())});
        assert(res.size() == 1U && "Silu backward should return only one gradient");
        tensor->add_grad(res.front().value());
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim) {
    auto log_softmax = ttnn_fixed::log_softmax(tensor->get_value(), dim);
    auto out = autograd::create_tensor(log_softmax);
    autograd::GradFunction grad = [tensor, out, dim]() {
        auto softmax = ttnn::exp(out->get_value());
        auto sum_grad_over_dim = ttnn_fixed::sum_over_dim(out->get_grad(), dim);
        auto grad = ttnn::subtract(out->get_grad(), ttnn::multiply(softmax, sum_grad_over_dim));
        tensor->add_grad(grad);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim) {
    auto log_softmax = ttnn::moreh_softmax(
        tensor->get_value(),
        /* axis */ dim,
        /* output */ std::nullopt,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::softmax());
    auto out = autograd::create_tensor(log_softmax);

    autograd::GradFunction grad = [tensor, out, dim]() {
        auto grad = ttnn::moreh_softmax_backward(
            out->get_value(),
            out->get_grad(),
            /* axis */ dim,
            /* output */ std::nullopt,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::LOGSOFTMAX,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        tensor->add_grad(grad);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr softmax(const autograd::TensorPtr& tensor, int dim) {
    auto softmax_out = ttml::metal::softmax(
        tensor->get_value(),
        /* axis */ dim);
    auto out = autograd::create_tensor(softmax_out);

    autograd::GradFunction grad = [tensor, out, dim]() {
        // Composite softmax backward:
        // grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output, dim, keepdim=true))
        auto softmax_output = out->get_value();
        auto grad_output = out->get_grad();
        auto grad_times_output = ttnn::multiply(grad_output, softmax_output);
        auto sum_grad_times_output = ttnn_fixed::sum_moreh(grad_times_output, dim, /*keep_dim=*/true);
        auto grad_minus_sum = ttnn::subtract(grad_output, sum_grad_times_output);
        auto grad = ttnn::multiply(softmax_output, grad_minus_sum);
        tensor->add_grad(grad);
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    auto shape = ttnn::Shape({1, 1, 1, 1});
    auto out =
        autograd::create_tensor(core::empty(shape, &autograd::ctx().get_device(), tensor->get_value().memory_config()));
    ttnn::moreh_mean(
        tensor->get_value(),
        std::nullopt,
        true,
        std::nullopt,
        out->get_value(),
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    autograd::GradFunction grad = [tensor, out]() {
        auto resulting_shape = tensor->get_value().logical_shape();
        auto res = ttnn::moreh_mean_backward(
            out->get_grad(),
            std::nullopt,
            false,
            resulting_shape,
            std::nullopt,
            std::nullopt,
            core::ComputeKernelConfig::precise());
        tensor->add_grad(res);
    };
    auto links = autograd::get_links(tensor);

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim) {
    if (new_batch_dim == 1 || tensor->get_value().logical_shape()[0] == new_batch_dim) {
        return tensor;
    }
    auto out = ttml::autograd::create_tensor();
    auto repeats = ttnn::Shape({new_batch_dim, 1, 1, 1});
    // currently assuming tensor came with shape: {1,X,Y,Z} and we want to get {B,X,Y,Z}
    out->set_value(ttnn::repeat(tensor->get_value(), repeats));

    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn_fixed::sum_over_batch(out->get_grad());
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links = autograd::get_links(tensor);

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr exp(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn::exp(tensor->get_value()));

    autograd::GradFunction grad = [tensor, out]() {
        // d/dx exp(x) = exp(x)
        // dL/dx = dL/d(exp(x)) * exp(x) = grad * out
        tensor->add_grad(ttnn::multiply(out->get_grad(), out->get_value()));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr reciprocal(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn::reciprocal(tensor->get_value()));

    autograd::GradFunction grad = [tensor, out]() {
        // d/dx (1/x) = -1/x^2 = -out^2
        // dL/dx = grad * (-out^2)
        auto neg_out_sq = ttnn::neg(ttnn::multiply(out->get_value(), out->get_value()));
        tensor->add_grad(ttnn::multiply(out->get_grad(), neg_out_sq));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr neg(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn::neg(tensor->get_value()));

    autograd::GradFunction grad = [tensor, out]() {
        // d/dx (-x) = -1
        tensor->add_grad(ttnn::neg(out->get_grad()));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr sum(const autograd::TensorPtr& tensor, int dim, bool keepdim) {
    auto out = autograd::create_tensor(ttnn::sum(tensor->get_value(), dim, keepdim));

    autograd::GradFunction grad = [tensor, out]() {
        // Gradient of sum is broadcast back along the summed dimension
        // If keepdim=true, grad has shape with 1 at dim, broadcast to original shape
        // If keepdim=false, need to unsqueeze first
        auto grad_val = out->get_grad();

        // Simply add the gradient - ttnn::add handles broadcasting
        // The gradient gets broadcast from reduced shape back to original shape
        tensor->add_grad(ttnn::multiply(grad_val, ttnn::ones_like(tensor->get_value())));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr max(const autograd::TensorPtr& tensor, int dim, bool keepdim) {
    // Compute max value
    auto max_val = ttnn::max(tensor->get_value(), dim, keepdim);
    auto out = autograd::create_tensor(max_val);

    autograd::GradFunction grad = [tensor, out, dim, keepdim]() {
        // Gradient is 1 at position of max, 0 elsewhere
        // For ties: gradient goes to one arbitrary position (distributed evenly among tied positions)
        // This matches PyTorch's subgradient behavior for tied maxes

        auto max_val_broadcast = out->get_value();
        // Create mask: 1 where input equals max, 0 elsewhere
        auto mask = ttnn::eq(tensor->get_value(), max_val_broadcast);

        // For ties: distribute gradient evenly among all max positions
        // This is the mathematically correct subgradient
        auto mask_sum = ttnn::sum(mask, dim, keepdim);
        auto mask_normalized = ttnn::multiply(mask, ttnn::reciprocal(ttnn::add(mask_sum, 1e-10F)));

        // Gradient flows to max position(s)
        tensor->add_grad(ttnn::multiply(out->get_grad(), mask_normalized));
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
