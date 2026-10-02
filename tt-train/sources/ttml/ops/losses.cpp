// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <limits>
#include <stdexcept>
#include <ttnn/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

ttnn::Tensor canonicalize_nll_target(const ttnn::Tensor& target, uint32_t sample_count) {
    const auto& shape = target.logical_shape();
    const bool is_flat_row = (shape.rank() == 1U && shape[0] == sample_count) ||
                             (shape.rank() == 2U && shape[0] == 1U && shape[1] == sample_count);
    if (is_flat_row && target.dtype() == ttnn::DataType::INT32 && target.layout() == ttnn::Layout::TILE) {
        return target;
    }

    auto row_major_target = target;
    if (row_major_target.layout() != ttnn::Layout::ROW_MAJOR) {
        row_major_target = ttnn::to_layout(row_major_target, ttnn::Layout::ROW_MAJOR);
    }
    row_major_target = ttnn::reshape(row_major_target, ttnn::Shape({sample_count}));
    return ttnn::to_layout(row_major_target, ttnn::Layout::TILE, ttnn::DataType::INT32);
}

}  // namespace

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
    if (reduce != ReduceType::NONE && reduce != ReduceType::MEAN) {
        throw std::logic_error(fmt::format(
            "Unsupported cross entropy reduction type, only NONE and MEAN are supported. Got: {}",
            enchantum::to_string(reduce)));
    }

    auto prediction_shape = prediction->get_shape();
    auto target_shape = target->get_shape();

    if (prediction_shape.rank() != 4U || target_shape.rank() != 2U) {
        throw std::logic_error(
            fmt::format(
                "Cross entropy loss expects: prediction rank = 4, target rank = 2.\n"
                "Got: prediction shape {}, target shape {}",
                prediction_shape,
                target_shape));
    }

    if (prediction_shape[0] != target_shape[0]) {
        throw std::logic_error(
            fmt::format(
                "Cross entropy loss: batch dimension (dim 0) must match.\n"
                "Got: prediction shape {}, target shape {}",
                prediction_shape,
                target_shape));
    }

    if (prediction_shape[-2] != target_shape[-1]) {
        throw std::logic_error(
            fmt::format(
                "Cross entropy loss: prediction dim -2 must equal target dim -1.\n"
                "Got: prediction shape {}, target shape {}",
                prediction_shape,
                target_shape));
    }

    auto loss = ttml::metal::cross_entropy_fw(prediction->get_value(), target->get_value());
    autograd::TensorPtr out;

    if (reduce == ReduceType::NONE) {
        out = autograd::create_tensor(loss);
    } else {
        auto shape = ttnn::Shape({1, 1, 1, 1});
        out = autograd::create_tensor(core::empty(shape, &autograd::ctx().get_device(), loss.memory_config()));
        ttnn::moreh_mean(
            loss,
            std::nullopt,
            true,
            std::nullopt,
            out->get_value(),
            std::nullopt,
            /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    }
    autograd::GradFunction grad = [target, prediction, out, reduce]() {
        float scaler = 1.0F;
        if (reduce == ReduceType::MEAN) {
            auto volume = target->get_value().logical_volume();
            scaler = 1.0F / static_cast<float>(volume);
        }
        auto grad =
            ttml::metal::cross_entropy_bw(prediction->get_value(), target->get_value(), out->get_grad(), scaler);
        prediction->add_grad(grad);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, prediction, target));
    return out;
}

autograd::TensorPtr nll_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    if (reduce != ReduceType::MEAN) {
        throw std::logic_error("Unsupported NLL reduction type, only MEAN is supported");
    }

    const auto& prediction_value = prediction->get_value();
    const auto& target_value = target->get_value();
    const auto& tensor_shape = prediction_value.logical_shape();

    if (tensor_shape.rank() != 4U) {
        throw std::logic_error(fmt::format("NLL loss expects prediction rank 4, got shape {}", tensor_shape));
    }
    if (!ttnn::is_device_tensor(prediction_value) || !ttnn::is_device_tensor(target_value)) {
        throw std::logic_error("NLL loss expects prediction and target tensors on device");
    }
    if (prediction_value.device() != target_value.device()) {
        throw std::logic_error("NLL loss expects prediction and target tensors on the same device or mesh");
    }
    if (prediction_value.dtype() != ttnn::DataType::BFLOAT16 || prediction_value.layout() != ttnn::Layout::TILE) {
        throw std::logic_error(fmt::format(
            "NLL loss expects a BFLOAT16 TILE prediction tensor, got dtype {} and layout {}",
            prediction_value.dtype(),
            prediction_value.layout()));
    }
    if (target_value.dtype() != ttnn::DataType::INT32 && target_value.dtype() != ttnn::DataType::UINT32) {
        throw std::logic_error(
            fmt::format("NLL loss expects an INT32 or UINT32 target tensor, got {}", target_value.dtype()));
    }
    if (prediction_value.memory_config().is_sharded() || target_value.memory_config().is_sharded()) {
        throw std::logic_error("NLL loss does not support sharded prediction or target tensors");
    }

    const uint64_t sample_count = static_cast<uint64_t>(tensor_shape[0]) * tensor_shape[1] * tensor_shape[2];
    if (sample_count > std::numeric_limits<uint32_t>::max()) {
        throw std::logic_error(fmt::format("NLL loss sample count {} exceeds UINT32_MAX", sample_count));
    }
    if (tensor_shape[3] == 0U) {
        throw std::logic_error("NLL loss expects a non-empty class dimension");
    }
    if (target_value.logical_volume() != sample_count) {
        throw std::logic_error(fmt::format(
            "NLL loss expects target volume to match prediction sample count {}, got target shape {} with volume {}",
            sample_count,
            target_value.logical_shape(),
            target_value.logical_volume()));
    }

    const auto Ndim = static_cast<uint32_t>(sample_count);
    const uint32_t Cdim = tensor_shape[3];
    auto canonical_target = canonicalize_nll_target(target_value, Ndim);
    auto* device = prediction_value.device();
    auto divisor = core::empty(ttnn::Shape({1, 1}), device, prediction_value.memory_config());
    auto reshaped_tensor = ttnn::reshape(prediction_value, ttnn::Shape({Ndim, Cdim}));
    auto loss_tensor = ttnn::moreh_nll_loss(
        reshaped_tensor,
        canonical_target,
        /* reduction */ "mean",
        /* weight_tensor */ std::nullopt,
        /* divisor_tensor */ divisor,
        /* output_tensor */ std::nullopt,
        /* ignore_index */ -100,
        /* memory_config */ prediction_value.memory_config(),
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto out = autograd::create_tensor(loss_tensor);

    autograd::GradFunction grad = [prediction, canonical_target, out, Ndim, Cdim, device, divisor]() {
        auto out_grad = core::empty(ttnn::Shape({Ndim, Cdim}), device, prediction->get_value().memory_config());
        auto grad = ttnn::moreh_nll_loss_backward(
            canonical_target,
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
    out->set_node(autograd::add_backward_node(std::move(grad), out, prediction, target));

    return out;
}

}  // namespace ttml::ops
