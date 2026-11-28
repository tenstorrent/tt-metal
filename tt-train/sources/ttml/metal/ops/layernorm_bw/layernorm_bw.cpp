// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw.hpp"

#include "device/layernorm_bw_device_operation.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

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
        grads.gamma ? std::make_optional(ttnn_fixed::sum_over_leading_dims(result[1].value())) : std::nullopt,
        // dL_dbeta is the row sum of dL_dout itself, so it needs no kernel output.
        grads.beta ? std::make_optional(ttnn_fixed::sum_over_leading_dims(dL_dout_tensor)) : std::nullopt};
}

}  // namespace ttml::metal
