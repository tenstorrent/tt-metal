// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_contract.hpp"

#include "device/skinny_contract_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor skinny_contract(
    const ttnn::Tensor& small,
    const ttnn::Tensor& big,
    float scale,
    bool transpose_out,
    bool big_transposed,
    bool small_transposed) {
    // The kernel wants small^T ([r, T]: one tile row of T/32 tiles) as its resident A operand.
    auto small_t = small_transposed ? small : ttnn::transpose(small, -2, -1);
    return ttnn::prim::ttml_skinny_contract(small_t, big, scale, transpose_out, big_transposed);
}

}  // namespace ttml::metal
