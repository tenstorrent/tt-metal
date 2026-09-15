// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <array>
#include <optional>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn/tensor/tensor.hpp"
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

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, GeluVariant variant) {
    // Unlike ttnn, fast_lut variant is not supported (no fast-lut backward kernel)
    if (variant == GeluVariant::FAST_LUT) {
        throw std::invalid_argument("gelu: GeluVariant::FAST_LUT is not supported for training");
    }

    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value(), variant));
    autograd::GradFunction grad = [tensor, out, variant]() {
        auto dL_dt = ttnn::gelu_bw(out->get_grad(), tensor->get_value(), variant);
        tensor->add_grad(dL_dt[0].value());
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

autograd::TensorPtr sigmoid(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::sigmoid(tensor->get_value()));

    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn::sigmoid_bw(out->get_grad(), tensor->get_value());
        tensor->add_grad(res[0]);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr sum_over_dim(const autograd::TensorPtr& tensor, int dim) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn_fixed::sum_ttnn(tensor->get_value(), dim, /* keep_dim */ true));

    autograd::GradFunction grad = [tensor, out, dim]() {
        // Every input element along `dim` contributed once, so the gradient is
        // the upstream gradient broadcast back over the reduced axis.
        auto input_shape = tensor->get_value().logical_shape();
        const auto rank = static_cast<int>(input_shape.rank());
        const int axis = dim < 0 ? dim + rank : dim;

        ttnn::SmallVector<uint32_t> repeats(rank, 1U);
        repeats[axis] = input_shape[axis];
        tensor->add_grad(ttnn::repeat(out->get_grad(), ttnn::Shape(repeats)));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr cumsum(const autograd::TensorPtr& tensor, int dim) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::cumsum(tensor->get_value(), dim, /* dtype */ std::nullopt, /* reverse_order */ false));

    autograd::GradFunction grad = [tensor, out, dim]() {
        // out_i = sum_{j<=i} x_j, so dL/dx_i = sum_{j>=i} g_j: a reverse cumsum.
        tensor->add_grad(ttnn::cumsum(out->get_grad(), dim, /* dtype */ std::nullopt, /* reverse_order */ true));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr softplus(const autograd::TensorPtr& tensor, float beta, float threshold) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::softplus(tensor->get_value(), beta, threshold));

    autograd::GradFunction grad = [tensor, out, beta, threshold]() {
        auto res = ttnn::softplus_bw(out->get_grad(), tensor->get_value(), beta, threshold);
        tensor->add_grad(res[0]);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr l2_norm(const autograd::TensorPtr& tensor, float epsilon) {
    auto out = autograd::create_tensor();

    auto input = tensor->get_value();
    auto squared_sum = ttnn_fixed::sum_ttnn(ttnn::multiply(input, input), 3, /* keep_dim */ true);
    auto inv_norm = ttnn::rsqrt(ttnn::add(squared_sum, epsilon));
    auto normalized = ttnn::multiply(input, inv_norm);
    out->set_value(normalized);

    autograd::GradFunction grad = [tensor, out, inv_norm, normalized]() {
        // With r = rsqrt(sum(x^2) + eps) and y = x * r,
        //     dL/dx = r * (g - y * sum(g * y))
        // which keeps the reduction in terms of y instead of recomputing it from x.
        auto upstream = out->get_grad();
        auto projection = ttnn_fixed::sum_ttnn(ttnn::multiply(upstream, normalized), 3, /* keep_dim */ true);
        tensor->add_grad(ttnn::multiply(inv_norm, ttnn::subtract(upstream, ttnn::multiply(normalized, projection))));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

namespace {

// Slice `length` elements starting at `start` along `dim`, then pad the opposite
// end back to the original extent with zeros. Used by shift_along_dim for both
// the forward (pad at the front) and the backward (pad at the back).
ttnn::Tensor shift_impl(const ttnn::Tensor& input, int axis, uint32_t shift, bool pad_front) {
    const auto shape = input.logical_shape();
    const auto rank = static_cast<int>(shape.rank());
    const uint32_t extent = shape[axis];

    ttsl::SmallVector<uint32_t> start(rank, 0U);
    ttsl::SmallVector<uint32_t> end(rank);
    ttsl::SmallVector<uint32_t> step(rank, 1U);
    for (int i = 0; i < rank; ++i) {
        end[i] = shape[i];
    }
    if (pad_front) {
        // Keep [0, extent - shift) and prepend zeros.
        end[axis] = extent - shift;
    } else {
        // Keep [shift, extent) and append zeros.
        start[axis] = shift;
    }
    auto kept = ttnn::slice(input, start, end, step);

    auto pad_shape = shape;
    pad_shape[axis] = shift;
    auto zeros = core::zeros(pad_shape, &autograd::ctx().get_device(), input.dtype());

    std::vector<ttnn::Tensor> parts =
        pad_front ? std::vector<ttnn::Tensor>{zeros, kept} : std::vector<ttnn::Tensor>{kept, zeros};
    return ttnn::concat(parts, axis);
}

}  // namespace

autograd::TensorPtr shift_along_dim(const autograd::TensorPtr& tensor, int dim, int shift) {
    if (shift == 0) {
        return tensor;
    }
    if (shift < 0) {
        throw std::runtime_error("shift_along_dim expects a non-negative shift");
    }

    const auto rank = static_cast<int>(tensor->get_value().logical_shape().rank());
    const int axis = dim < 0 ? dim + rank : dim;
    const auto amount = static_cast<uint32_t>(shift);

    auto out = autograd::create_tensor();
    out->set_value(shift_impl(tensor->get_value(), axis, amount, /* pad_front */ true));

    autograd::GradFunction grad = [tensor, out, axis, amount]() {
        // out[t] = x[t - shift], so dL/dx[t] = g[t + shift] -- the same shift
        // the other way, which drops the leading rows and pads the tail.
        tensor->add_grad(shift_impl(out->get_grad(), axis, amount, /* pad_front */ false));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr transpose(const autograd::TensorPtr& tensor, int dim0, int dim1) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::transpose(tensor->get_value(), dim0, dim1));

    autograd::GradFunction grad = [tensor, out, dim0, dim1]() {
        tensor->add_grad(ttnn::transpose(out->get_grad(), dim0, dim1));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
