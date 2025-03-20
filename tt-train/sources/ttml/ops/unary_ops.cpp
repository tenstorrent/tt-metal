// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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
        tt::tt_metal::MemoryConfig mem_config;
        static const std::string approx_mode = "none";
        auto res = ttnn::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode, mem_config);
        assert(res.size() == 1U && "Gelu backward should return only one gradient");
        tensor->add_grad(res.front().value());
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr silu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn::silu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn::silu_bw(out->get_grad(), tensor->get_value());
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

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    auto shape = core::create_shape({1, 1, 1, 1});
    autograd::TensorPtr out = autograd::create_tensor(core::from_vector({0.F}, shape, &autograd::ctx().get_device()));
    ttnn::moreh_mean(
        tensor->get_value(),
        std::nullopt,
        true,
        std::nullopt,
        out->get_value(),
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    autograd::GradFunction grad = [tensor, out]() {
        auto resulting_shape = tensor->get_value().get_logical_shape();
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
    if (new_batch_dim == 1 || tensor->get_value().get_logical_shape()[0] == new_batch_dim) {
        return tensor;
    }
    auto out = ttml::autograd::create_tensor();
    auto repeats = core::create_shape({new_batch_dim, 1, 1, 1});
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

autograd::TensorPtr dytanh(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& scale,
    const autograd::TensorPtr& gain,
    const autograd::TensorPtr& bias) {
    if (scale->get_value().get_logical_shape().to_array_4D() != std::array<uint32_t, 4>{1, 1, 1, 1}) {
        throw std::invalid_argument(
            fmt::format("Scale must be a 4D tensor with one element, got {}", scale->get_value().get_logical_shape()));
    }
    auto x = tensor->get_value();
    auto scale_tensor = scale->get_value();
    auto gain_tensor = gain->get_value();
    auto bias_tensor = bias->get_value();

    auto scaled = ttnn::experimental::mul(x, scale_tensor);
    auto tanh_scaled = ttnn::tanh(scaled);
    auto gained = ttnn::experimental::mul(tanh_scaled, gain_tensor);
    auto biased = ttnn::experimental::add(gained, bias_tensor);
    auto out = autograd::create_tensor(biased);

    autograd::GradFunction grad = [tensor, out, scale, gain, bias, tanh_scaled]() {
        auto dL_dout = out->get_grad();

        // dL/dscale = dL/dout * gain * 1 - tanh^2(scaled) * x
        auto dL_dscale = ttnn::experimental::mul(dL_dout, gain->get_value());
        dL_dscale = ttnn::experimental::mul(tensor->get_value(), dL_dscale);
        auto one_minus_tanh = ttnn::experimental::sub(ttnn::ones_like(tanh_scaled), ttnn::square(tanh_scaled));
        dL_dscale = ttnn::experimental::mul(one_minus_tanh, dL_dscale);
        // α is a scalar broadcast over the entire batch, so we perform a sum
        // over all dims.
        dL_dscale = ttnn_fixed::sum_moreh(dL_dscale, {0, 1, 2, 3}, /*keep_dim=*/true);
        scale->add_grad(dL_dscale);

        // dL/dx = dL/dout * gain * 1 - tanh^2(scaled) * scale
        auto dL_dx = ttnn::experimental::mul(dL_dout, gain->get_value());
        dL_dx = ttnn::experimental::mul(dL_dx, one_minus_tanh);
        dL_dx = ttnn::experimental::mul(dL_dx, scale->get_value());
        tensor->add_grad(dL_dx);

        // dL/dgain = dL/dout * tanh(scaled)
        auto dL_dgain = ttnn::experimental::mul(dL_dout, tanh_scaled);
        dL_dgain = ttnn_fixed::sum_moreh(dL_dgain, {0, 1, 2}, /*keep_dim=*/true);
        gain->add_grad(dL_dgain);

        // dL/dbias = dL/dout
        auto dL_dbias = ttnn_fixed::sum_moreh(dL_dout, {0, 1, 2}, /*keep_dim=*/true);
        bias->add_grad(dL_dbias);
    };
    auto links = autograd::get_links(tensor, scale, gain, bias);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
