// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_cross_entropy_bw.hpp"

#include "device/vp_cross_entropy_bw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor vp_cross_entropy_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& log_normalizer,
    const ttnn::Tensor& target,
    const ttnn::Tensor& grad,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    float scale) {
    return ttnn::prim::ttml_vp_cross_entropy_bw(
        logits, log_normalizer, target, grad, local_V, cluster_axis, first_v, scale);
}

}  // namespace ttml::metal
