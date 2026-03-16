// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_elemwise_bw.hpp"

#include "device/swiglu_elemwise_bw_device_operation.hpp"

namespace ttml::metal {

SwigluElemwiseBwResult swiglu_elemwise_bw(
    const ttnn::Tensor& linear1,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& dL_dprod,
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate) {
    auto result =
        ttnn::prim::ttml_swiglu_elemwise_bw(linear1, gate, dL_dprod, preallocated_dL_dlinear1, preallocated_dL_dgate);
    return SwigluElemwiseBwResult{.dL_dlinear1 = result.dL_dlinear1, .dL_dgate = result.dL_dgate};
}

}  // namespace ttml::metal
