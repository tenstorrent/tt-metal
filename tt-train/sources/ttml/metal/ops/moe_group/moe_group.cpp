// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group.hpp"

#include "device/moe_group_device_operation.hpp"

namespace ttml::metal {

MoeGroupResult moe_group(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k) {
    return ttnn::prim::ttml_moe_group(dispatched, metadata, local_expert_ids, e_local, k);
}

}  // namespace ttml::metal
