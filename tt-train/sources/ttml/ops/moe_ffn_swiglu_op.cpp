// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ffn_swiglu_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "metal/ops/sparse_matmul/sparse_matmul.hpp"

namespace ttml::ops {

ttnn::Tensor moe_ffn_swiglu_fw(
    const ttnn::Tensor& grouped,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down) {
    // Gate and up-projection into the intermediate dim.
    auto gate_proj = ttml::metal::sparse_matmul(grouped, w_gate, offsets);
    auto up_proj = ttml::metal::sparse_matmul(grouped, w_up, offsets);

    // SiLU(gate) * up. Pad rows are zero in both operands, so they stay zero.
    auto activated = ttnn::multiply(ttnn::silu(gate_proj), up_proj);
    gate_proj.deallocate();
    up_proj.deallocate();

    auto out = ttml::metal::sparse_matmul(activated, w_down, offsets);
    activated.deallocate();
    return out;
}

}  // namespace ttml::ops
