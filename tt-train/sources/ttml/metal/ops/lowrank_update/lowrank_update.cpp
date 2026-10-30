// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lowrank_update.hpp"

#include "device/lowrank_update_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor lowrank_update(
    const ttnn::Tensor& acc, const ttnn::Tensor& a, const ttnn::Tensor& w, float scale, bool transpose_b) {
    return ttnn::prim::ttml_lowrank_update(acc, a, w, scale, transpose_b);
}

}  // namespace ttml::metal
