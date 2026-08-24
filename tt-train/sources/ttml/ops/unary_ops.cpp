// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

// ttnn::experimental::gelu_bw takes a string, not a GeluVariant, and only special-cases the exact
// string "tanh" -- anything else selects the exact-derivative polynomial kernel. FAST_LUT never
// reaches here: gelu() rejects it when a gradient would be built, because ttnn has no LUT backward
// kernel to pair with the LUT forward.
const std::string& gelu_bw_approximate(GeluVariant variant) {
    static const std::string k_none = "none";
    static const std::string k_tanh = "tanh";
    return variant == GeluVariant::TANH ? k_tanh : k_none;
}

}  // namespace

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_value(), mem_config);
        tensor->add_grad(res[0]);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

GeluVariant gelu_variant_from_string(std::string_view name) {
    if (name == "none" || name == "accurate") {
        return GeluVariant::ACCURATE;
    }
    if (name == "tanh") {
        return GeluVariant::TANH;
    }
    if (name == "fast_lut") {
        return GeluVariant::FAST_LUT;
    }
    throw std::invalid_argument(
        fmt::format("Unknown GELU variant: {}. Supported variants [none, accurate, tanh, fast_lut]", name));
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, GeluVariant variant) {
    // FAST_LUT is forward-only: its backward would run the exact GELU derivative, which is not the
    // derivative of the piecewise-linear forward, so training through it descends the wrong
    // objective. Reject it exactly when add_backward_node() would create a node, so inference paths
    // (GradMode::DISABLED) keep the fast kernel.
    if (variant == GeluVariant::FAST_LUT && autograd::ctx().get_gradient_mode() == autograd::GradMode::ENABLED &&
        autograd::any_requires_grad(tensor)) {
        throw std::invalid_argument(
            "gelu: GeluVariant::FAST_LUT is forward-only -- ttnn has no LUT backward kernel, so the "
            "gradient would be the exact GELU derivative rather than the derivative of the LUT "
            "forward. Use GeluVariant::TANH for a trainable approximation, or run with gradient mode "
            "disabled.");
    }

    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value(), variant));
    autograd::GradFunction grad = [tensor, out, variant]() {
        auto dL_dt = ttnn::experimental::gelu_bw(out->get_grad(), tensor->get_value(), gelu_bw_approximate(variant));
        tensor->add_grad(dL_dt);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

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
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr exp(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::exp(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn::exp_bw(out->get_grad(), tensor->get_value());
        tensor->add_grad(res[0].value());
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::clip(tensor->get_value(), lo, hi));
    autograd::GradFunction grad = [tensor, out, lo, hi]() {
        auto res = ttnn::clip_bw(out->get_grad(), tensor->get_value(), lo, hi);
        tensor->add_grad(res[0]);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
