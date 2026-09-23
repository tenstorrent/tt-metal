// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_fw.hpp"

#include "device/mla_kv_assemble_fw_device_operation.hpp"

namespace ttml::metal {

std::tuple<ttnn::Tensor, ttnn::Tensor> mla_kv_assemble_fw(
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    auto result = ttnn::prim::ttml_mla_kv_assemble_fw(kv_up, k_pe, n_heads, qk_nope_dim, qk_rope_dim, v_dim);
    return {std::move(result[0]), std::move(result[1])};
}

}  // namespace ttml::metal
