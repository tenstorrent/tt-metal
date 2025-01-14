// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"

namespace ttml::ops {

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias);

void ttnn_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out,
    const ttnn::WormholeComputeKernelConfig& config = ttml::core::ComputeKernelConfig::matmul());

void moreh_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out,
    const ttnn::WormholeComputeKernelConfig& config = ttml::core::ComputeKernelConfig::matmul());

}  // namespace ttml::ops
