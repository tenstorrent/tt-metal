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
    const bool compute_grad_input,
    const bool compute_grad_gamma,
    const bool compute_grad_beta) {
    // Call the device operation
    // Returns: [dx, dgamma_components, dbeta_components]; a skipped grad is nullopt
    auto result = ttnn::prim::ttml_layernorm_bw(
        input_tensor,
        gamma_tensor,
        mean_tensor,
        rstd_tensor,
        dL_dout_tensor,
        /* preallocated_dx */ std::nullopt,
        /* preallocated_dgamma_components */ std::nullopt,
        /* preallocated_dbeta_components */ std::nullopt,
        compute_grad_input,
        compute_grad_gamma,
        compute_grad_beta);

    // dL_dgamma and dL_dbeta require sum over batches so we cannot perform this sum in the kernel.
    // Instead we return the components and reduce them here.

    return {
        result[0],  // dx - already complete
        compute_grad_gamma ? std::make_optional(ttnn_fixed::sum_over_leading_dims(result[1].value())) : std::nullopt,
        compute_grad_beta ? std::make_optional(ttnn_fixed::sum_over_leading_dims(result[2].value())) : std::nullopt};
}

}  // namespace ttml::metal
