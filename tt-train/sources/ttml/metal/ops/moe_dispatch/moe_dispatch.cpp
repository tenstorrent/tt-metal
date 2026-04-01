// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_dispatch.hpp"

#include "device/moe_dispatch_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor moe_dispatch(
    const ttnn::Tensor& sorted_hidden,
    const ttnn::Tensor& w_up,
    uint32_t cluster_axis,
    const std::vector<std::vector<uint32_t>>& expert_offsets_per_device,
    const std::vector<std::vector<uint32_t>>& expert_counts_per_device,
    uint32_t E_local) {
    auto result = ttnn::prim::ttml_moe_dispatch(
        sorted_hidden, w_up, cluster_axis, expert_offsets_per_device, expert_counts_per_device, E_local);
    return result[0];
}

}  // namespace ttml::metal
