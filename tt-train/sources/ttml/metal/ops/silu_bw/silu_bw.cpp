// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/silu_bw_device_operation.hpp"

namespace ttml::metal::ops::silu_bw {

std::vector<std::optional<ttnn::Tensor>> SiLUBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor) {
    return {ttnn::prim::ttml_silu_bw(
        input_tensor,   // [B,1,S,C]
        dL_dout_tensor  //[B,1,S,C]
        )[0]};
}

}  // namespace ttml::metal::ops::silu_bw
