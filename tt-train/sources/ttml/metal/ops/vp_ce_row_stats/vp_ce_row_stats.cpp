// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_ce_row_stats.hpp"

#include "device/vp_ce_row_stats_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor vp_ce_row_stats(const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col) {
    return ttnn::prim::ttml_vp_ce_row_stats(logits, mode, max_col);
}

}  // namespace ttml::metal
