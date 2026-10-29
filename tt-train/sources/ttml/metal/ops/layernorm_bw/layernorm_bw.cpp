// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/layernorm_bw_device_operation.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> layernorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& mean_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const LayerNormBackwardGrads& grads) {
    // dL_dgamma and dL_dbeta require sum over batches so we cannot perform this sum in the kernel.
    // Instead we reduce their components here.
    auto reduction_with_reshape_to2D = [](const ttnn::Tensor& tensor) {
        return ttnn::reshape(
            ttnn::sum(
                ttnn::reshape(
                    tensor,
                    ttnn::Shape(
                        {tensor.logical_shape()[0] * tensor.logical_shape()[1] * tensor.logical_shape()[2],
                         tensor.logical_shape()[3]})),  // dgamma_components
                /* dim_arg */ ttsl::SmallVector<int>{0},
                /* keep_dim */ true,
                /* output_mem_config */ std::nullopt,
                /*compute_kernel_config */ core::ComputeKernelConfig::precise()),
            ttnn::Shape({1, 1, 1, tensor.logical_shape()[3]}));
    };

    TT_FATAL(
        grads.input || grads.gamma || grads.beta,
        "layernorm_bw: at least one of grads.input, grads.gamma and grads.beta must be set");

    // Call the device operation for dx and dgamma only
    // Returns: [dx, dgamma_components]; a skipped grad is nullopt
    std::vector<std::optional<ttnn::Tensor>> result(2U);
    if (grads.input || grads.gamma) {
        result = ttnn::prim::ttml_layernorm_bw(
            input_tensor,
            gamma_tensor,
            mean_tensor,
            rstd_tensor,
            dL_dout_tensor,
            {.grad_input = grads.input, .grad_gamma = grads.gamma});
    }

    return {
        result[0],  // dx - already complete
        grads.gamma ? std::make_optional(reduction_with_reshape_to2D(result[1].value())) : std::nullopt,
        // dL_dbeta is the row sum of dL_dout itself, so it needs no kernel output.
        grads.beta ? std::make_optional(reduction_with_reshape_to2D(dL_dout_tensor)) : std::nullopt};
}

}  // namespace ttml::metal
