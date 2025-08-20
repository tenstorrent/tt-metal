// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_op.hpp"

#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

// simplified version of layernorm
// it works only for 4D tensors and for the last dimension
autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
    auto tensor_shape = tensor->get_value().logical_shape();
    auto mean = core::empty(
        ttnn::Shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}),
        &autograd::ctx().get_device(),
        tensor->get_value().memory_config());
    auto rstd = ttnn::empty_like(mean);
    auto output = ttnn::empty_like(tensor->get_value());

    auto out_tensors = ttnn::moreh_layer_norm(
        tensor->get_value(),
        1,
        1e-6F,
        /* gamma */ gamma->get_value(),
        /* beta */ beta->get_value(),
        output,
        mean,
        rstd,
        /* memory_config */ std::nullopt,
        /* compute_kernel_config */ std::nullopt);

    auto out = autograd::create_tensor();
    out->set_value(out_tensors[0].value());
    mean = out_tensors[1].value();
    rstd = out_tensors[2].value();

    autograd::GradFunction grad = [tensor, out, mean, rstd, gamma, beta]() {
        auto input_grad = ttnn::empty_like(tensor->get_value());
        auto gamma_grad = ttnn::empty_like(gamma->get_value());
        auto beta_grad = ttnn::empty_like(beta->get_value());

        auto res = ttnn::moreh_layer_norm_backward(
            out->get_grad(),
            tensor->get_value(),
            mean,
            rstd,
            1,
            gamma->get_value(),
            input_grad,
            gamma_grad,
            beta_grad,
            /* memory_config */ std::nullopt,
            /* compute_kernel_config */ std::nullopt);

        tensor->add_grad(res[0].value());
        gamma->add_grad(res[1].value());
        beta->add_grad(res[2].value());
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr composite_layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
    auto tensor_shape = tensor->get_value().logical_shape();

    auto shape = ttnn::Shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1});
    auto mean = core::zeros(shape, &autograd::ctx().get_device(), tensor->get_value().dtype());
    ttnn::moreh_mean(
        tensor->get_value(),
        3,  // last dimension
        true,
        std::nullopt,
        mean,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());

    auto tensor_squared = ttnn::square(tensor->get_value());
    auto mean_squared = core::zeros_like(mean);
    ttnn::moreh_mean(
        tensor_squared,
        3,  // last dimension
        true,
        std::nullopt,
        mean_squared,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto variance = ttnn::subtract(mean_squared, ttnn::square(mean));

    const float eps = 1e-6F;
    auto rstd = ttnn::rsqrt(ttnn::add(variance, eps));

    auto normalized_tensor = ttnn::multiply(ttnn::subtract(tensor->get_value(), mean), rstd);
    auto output = ttnn::add(ttnn::multiply(normalized_tensor, gamma->get_value()), beta->get_value());
    auto out = autograd::create_tensor(output);

    autograd::GradFunction grad = [tensor, out, gamma, beta, mean, rstd]() {
        auto dout = out->get_grad();

        // recalculate normalized tensor to save memory and avoid storing it
        auto normalized_tensor = ttnn::multiply(ttnn::subtract(tensor->get_value(), mean), rstd);

        auto dbeta = ttnn::moreh_sum(
            dout,
            /* dim */ ttnn::SmallVector<int64_t>{0, 1, 2},
            /* keep_dim */ true,
            /* output */ std::nullopt,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        auto dgamma = ttnn::moreh_sum(
            ttnn::multiply(dout, normalized_tensor),
            /* dim */ ttnn::SmallVector<int64_t>{0, 1, 2},
            /* keep_dim */ true,
            /* output */ std::nullopt,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        auto dtensor_normalized = ttnn::multiply(dout, gamma->get_value());

        // dtensor = (dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)) * rstd

        // dnorm.mean(-1, keepdim=True)
        auto dnorm_shape = dtensor_normalized.logical_shape();
        auto shape = ttnn::Shape({dnorm_shape[0], dnorm_shape[1], dnorm_shape[2], 1});
        auto dnorm_mean = core::zeros(shape, &autograd::ctx().get_device(), tensor->get_value().dtype());
        ttnn::moreh_mean(
            dtensor_normalized,
            /* dim */ 3,
            /* keep_dim */ true,
            /* divisor */ std::nullopt,
            /* output */ dnorm_mean,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        // dnorm.mean(-1, keepdim=True)
        auto dnorm_norm = ttnn::multiply(dtensor_normalized, normalized_tensor);
        auto dnorm_norm_mean = core::zeros_like(dnorm_mean);
        ttnn::moreh_mean(
            dnorm_norm,
            /* dim */ 3,
            /* keep_dim */ true,
            /* divisor */ std::nullopt,
            /* output */ dnorm_norm_mean,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        // norm * (dnorm * norm).mean(-1, keepdim=True)
        auto norm_dnorm_norm_mean = ttnn::multiply(normalized_tensor, dnorm_norm_mean);
        auto dtensor = ttnn::subtract(ttnn::subtract(dtensor_normalized, dnorm_mean), norm_dnorm_norm_mean);
        dtensor = ttnn::multiply(dtensor, rstd);

        tensor->add_grad(dtensor);
        gamma->add_grad(dgamma);
        beta->add_grad(dbeta);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
