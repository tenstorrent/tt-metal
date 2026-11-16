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
    const ttnn::Tensor& dL_dout_tensor) {
    // Call the device operation
    // Returns: [dx, dgamma_components, dbeta_components]
    auto result = ttnn::prim::ttml_layernorm_bw(input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dL_dout_tensor);

    // dL_dgamma and dL_dbeta require sum over batches so we cannot perform this sum in the kernel.
    // Instead we return the components and reduce them here.

    return {
        result[0],  // dx - already complete
        ttnn_fixed::sum_over_leading_dims(result[1]),
        ttnn_fixed::sum_over_leading_dims(result[2])};
}

}  // namespace ttml::metal
