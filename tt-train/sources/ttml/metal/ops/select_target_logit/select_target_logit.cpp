// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "select_target_logit.hpp"

#include "device/select_target_logit_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor select_target_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v) {
    return ttnn::prim::ttml_select_target_logit(logit, target, local_V, cluster_axis, first_v);
}

}  // namespace ttml::metal
