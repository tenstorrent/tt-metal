// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw.hpp"

#include <ttnn/operations/matmul/matmul.hpp>

#include "device/swiglu_fw_device_operation.hpp"
#include "metal/ops/swiglu_gate_up/swiglu_gate_up.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    const ttnn::Tensor& w3,
    bool use_two_phases) {
    if (use_two_phases) {
        // Design A: 2-step pipeline
        // Step 1: M = SiLU(X @ W1) * (X @ W3) — custom fused gate-up kernel
        ttnn::Tensor M = swiglu_gate_up(input_tensor, w1, w3);
        // Step 2: Y = M @ W2 — standard matmul
        return ttnn::matmul(M, w2);
    }
    return ttnn::prim::ttml_swiglu_fw(
        input_tensor,  // [B, 1, S, C]
        w1,            // [1, 1, C, H]
        w2,            // [1, 1, H, C]
        w3,            // [1, 1, C, H]
        std::nullopt);
}

}  // namespace ttml::metal
