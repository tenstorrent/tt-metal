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

// // simplified version of layernorm
// // it works only for 4D tensors and for the last dimension
// autograd::TensorPtr custom_layernorm(
//     const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
//     auto tensor_shape = tensor->get_value().get_shape();
//     auto mean = core::empty(
//         core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}),
//         &autograd::ctx().get_device(),
//         tensor->get_value().memory_config());
//     auto rstd = ttnn::empty_like(mean);
//     auto output = ttnn::empty_like(tensor->get_value());

//     auto out_tensors = ttnn::moreh_layer_norm(
//         tensor->get_value(),
//         1,
//         1e-6F,
//         /* gamma */ gamma->get_value(),
//         /* beta */ beta->get_value(),
//         output,
//         mean,
//         rstd,
//         /* memory_config */ std::nullopt,
//         /* compute_kernel_config */ std::nullopt);

//     auto out = autograd::create_tensor();
//     out->set_value(out_tensors[0].value());
//     mean = out_tensors[1].value();
//     rstd = out_tensors[2].value();

//     autograd::GradFunction grad = [tensor, out, mean, rstd, gamma, beta]() {
//         auto input_grad = ttnn::empty_like(tensor->get_value());
//         auto gamma_grad = ttnn::empty_like(gamma->get_value());
//         auto beta_grad = ttnn::empty_like(beta->get_value());

//         auto res = ttnn::moreh_layer_norm_backward(
//             out->get_grad(),
//             tensor->get_value(),
//             mean,
//             rstd,
//             1,
//             gamma->get_value(),
//             input_grad,
//             gamma_grad,
//             beta_grad,
//             /* memory_config */ std::nullopt,
//             /* compute_kernel_config */ std::nullopt);

//         tensor->add_grad(res[0].value());
//         gamma->add_grad(res[1].value());
//         beta->add_grad(res[2].value());
//     };

//     auto links = autograd::get_links(tensor);
//     out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

//     return out;
// }

autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta) {
    auto tensor_shape = tensor->get_value().get_shape();

    auto shape = core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1});
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

    autograd::GradFunction grad = [tensor, out, mean, rstd, normalized_tensor, variance, gamma, beta, eps]() {
        auto dout = out->get_grad();
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

        // dvariance = np.sum(dx_norm * (x - mean) * -0.5 * (variance + epsilon) ** (-1.5), axis=-1, keepdims=True)
        // dx_norm * (x - mean) * -0.5 * (variance + epsilon) ** (-1.5)
        auto x_sub_mean = ttnn::subtract(tensor->get_value(), mean);
        auto x_sub_mean_half = ttnn::multiply(x_sub_mean, -0.5F);
        auto x_variance_1_5 = ttnn::power(ttnn::add(variance, eps), -1.5F);
        auto dvariance_before_reduce =
            ttnn::multiply(dtensor_normalized, ttnn::multiply(x_sub_mean_half, x_variance_1_5));

        auto dvariance = ttnn::moreh_sum(
            dvariance_before_reduce,
            /* dim */ ttnn::SmallVector<int64_t>{0, 1, 2},
            /* keep_dim */ true,
            /* output */ std::nullopt,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        // dmean =
        //      np.sum(dx_norm * -1 / np.sqrt(variance + epsilon), axis=1, keepdims=True) +
        //      dvariance * np.mean(-2 * (x - mean), axis=1, keepdims=True)

        // np.sum(dx_norm * -1 / np.sqrt(variance + epsilon), axis=1, keepdims=True)
        auto dmean_p0 = ttnn::multiply(ttnn::neg(dtensor_normalized), ttnn::rsqrt(ttnn::add(variance, eps)));
        dmean_p0 = ttnn::moreh_sum(
            dmean_p0,
            /* dim */ 3,
            /* keep_dim */ true,
            /* output */ std::nullopt,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise());

        // dvariance * np.mean(-2 * (x - mean), axis=1, keepdims=True)
        auto x_sub_mean_mult_neg2 = ttnn::multiply(x_sub_mean, -2);
        auto before_reduction_shape = x_sub_mean_mult_neg2.get_shape();
        auto shape =
            core::create_shape({before_reduction_shape[0], before_reduction_shape[1], before_reduction_shape[2], 1});
        auto x_sub_mean_mult_neg2_reduced =
            core::zeros(shape, &autograd::ctx().get_device(), tensor->get_value().dtype());
        ttnn::moreh_mean(
            tensor->get_value(),
            3,  // last dimension
            true,
            std::nullopt,
            x_sub_mean_mult_neg2_reduced,
            std::nullopt,
            /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());

        auto dmean_p1 = ttnn::multiply(dvariance, x_sub_mean_mult_neg2_reduced);
        auto dmean = ttnn::add(dmean_p0, dmean_p1);

        // dx = dx_norm / np.sqrt(variance + epsilon) + dvariance * 2 * (x - mean) / H + dmean / H
        auto dtensor_p0 = ttnn::multiply(dtensor_normalized, ttnn::rsqrt(ttnn::add(variance, eps)));
        auto dtensor_p1 = ttnn::multiply(dvariance, ttnn::multiply(x_sub_mean, 2));
        dtensor_p1 = ttnn::add(dtensor_p1, dmean);

        auto tensor_shape = tensor->get_value().get_shape();
        dtensor_p1 = ttnn::multiply(dtensor_p1, 1.F / static_cast<float>(tensor_shape[3]));
        auto dtensor = ttnn::add(dtensor_p0, dtensor_p1);

        tensor->add_grad(dtensor);
        gamma->add_grad(dgamma);
        beta->add_grad(dbeta);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
