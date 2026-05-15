// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target.hpp"

#include "device/subtract_at_target_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor subtract_at_target(
    const ttnn::Tensor& input,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    float subtract_value) {
    return ttnn::prim::ttml_subtract_at_target(
        input, target, local_V, cluster_axis, first_v, std::nullopt, subtract_value);
}

}  // namespace ttml::metal
