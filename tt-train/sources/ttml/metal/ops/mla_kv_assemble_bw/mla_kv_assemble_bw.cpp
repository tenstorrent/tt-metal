// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_bw.hpp"

#include "device/mla_kv_assemble_bw_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> mla_kv_assemble_bw(
    const ttnn::Tensor& dK,
    const ttnn::Tensor& dV,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    auto result = ttnn::prim::ttml_mla_kv_assemble_bw(dK, dV, n_heads, qk_nope_dim, qk_rope_dim, v_dim);
    return {std::move(result[0]), std::move(result[1])};
}

}  // namespace ttml::metal
