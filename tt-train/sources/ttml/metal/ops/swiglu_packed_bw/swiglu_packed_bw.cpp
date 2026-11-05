// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw.hpp"

#include "device/swiglu_packed_bw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_packed_bw(
    const ttnn::Tensor& packed, const ttnn::Tensor& dL_dh, const std::optional<ttnn::Tensor>& preallocated_dL_dpacked) {
    return ttnn::prim::ttml_swiglu_packed_bw(packed, dL_dh, preallocated_dL_dpacked);
}

}  // namespace ttml::metal
