// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup.hpp"

#include "device/moe_ungroup_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor moe_ungroup(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& scores,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k,
    uint32_t d,
    uint32_t b,
    uint32_t s) {
    return ttnn::prim::ttml_moe_ungroup(
        expert_out, plan, offsets, metadata, scores, local_expert_ids, e_local, k, d, b, s);
}

}  // namespace ttml::metal
