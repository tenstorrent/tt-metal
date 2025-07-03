// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto difference = ops::sub(target, prediction);  // TODO: @rfurko-tt use "ttnn::squared_difference"
    auto squared_difference =
        ops::mul(difference, difference);  // TODO: need to add backward "ttnn::squared_difference_bw" might be faster
    if (reduce == ReduceType::MEAN) {
        return ops::mean(squared_difference);
    } else {
        throw std::logic_error("Unsupported MSE reduction type");
    }
}

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto loss = ttml::metal::cross_entropy_fw(prediction->get_value(), target->get_value());
    auto shape = core::create_shape({1, 1, 1, 1});
    autograd::TensorPtr out = autograd::create_tensor(core::from_vector({0.F}, shape, &autograd::ctx().get_device()));
    ttnn::moreh_mean(
        loss,
        std::nullopt,
        true,
        std::nullopt,
        out->get_value(),
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());

    autograd::GradFunction grad = [target, prediction, out]() {
        auto volume = target->get_value().logical_volume();
        float scaler = 1.0F / static_cast<float>(volume);
        auto grad =
            ttml::metal::cross_entropy_bw(prediction->get_value(), target->get_value(), out->get_grad(), scaler);
        prediction->add_grad(grad);
    };

    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr nll_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    if (reduce != ReduceType::MEAN) {
        throw std::logic_error("Unsupported NLL reduction type, only MEAN is supported");
    }

    auto* device = &autograd::ctx().get_device();
    auto divisor = core::empty(ttnn::Shape({1, 1}), device, prediction->get_value().memory_config());

    auto tensor_shape = prediction->get_value().logical_shape();
    uint32_t Ndim = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];
    uint32_t Cdim = tensor_shape[3];
    auto reshaped_tensor = ttnn::reshape(prediction->get_value(), ttnn::Shape({Ndim, Cdim}));
    auto loss_tensor = ttnn::moreh_nll_loss(
        reshaped_tensor,
        target->get_value(),
        /* reduction */ "mean",
        /* weight_tensor */ std::nullopt,
        /* divisor_tensor */ divisor,
        /* output_tensor */ std::nullopt,
        /* ignore_index */ -100,
        /* memory_config */ prediction->get_value().memory_config(),
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto out = autograd::create_tensor(loss_tensor);

    autograd::GradFunction grad = [prediction, target, out, Ndim, Cdim, device, divisor]() {
        auto out_grad = core::empty(ttnn::Shape({Ndim, Cdim}), device, prediction->get_value().memory_config());
        auto grad = ttnn::moreh_nll_loss_backward(
            target->get_value(),
            out->get_grad(),
            /* reduction_mean */ true,
            /* weight_tensor */ std::nullopt,
            /* input_grad_tensor */ out_grad,
            /* divisor_tensor */ divisor,
            /* ignore_index */ -100,
            /* memory_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        grad = ttnn::reshape(grad, prediction->get_value().logical_shape());
        prediction->add_grad(grad);
    };
    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
