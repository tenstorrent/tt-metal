// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "device/rmsnorm_bw_device_operation.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,   // [B,1,S,C]
        gamma_tensor,   // [1,1,1,C]
        rms_tensor,     //[B,1,S,1]
        dL_dout_tensor  //[B,1,S,C]
    );

    // dL_dgamma requires sum over batches so we cannot perform this sum in the kernel. Instead we return the
    // dL_dgamma_components and reduce it here.
    return {result[0], ttnn_fixed::sum_over_leading_dims(result[1])};  // [B,1,S,C] -> [1,1,1,C]
}

}  // namespace ttml::metal
