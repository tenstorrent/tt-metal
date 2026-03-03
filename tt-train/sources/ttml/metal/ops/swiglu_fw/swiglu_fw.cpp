// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>

#include "device/swiglu_fw_device_operation.hpp"
#include "metal/ops/swiglu_gate_up/swiglu_gate_up.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::metal {

ttnn::Tensor swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w1,
    const ttnn::Tensor& w2,
    const ttnn::Tensor& w3,
    SwigluFwPath path) {
    switch (path) {
        case SwigluFwPath::Composite: {
            // Composite path: performs the same math as LlamaMLP composite.
            // silu(X@W1), X@W3, multiply, then (result)@W2; used for performance comparison.
            auto xw1 = ttnn_fixed::matmul(input_tensor, w1);
            auto xw3 = ttnn_fixed::matmul(input_tensor, w3);
            auto swished = ttnn::silu(xw1);
            auto gated = ttnn::multiply(swished, xw3);
            return ttnn_fixed::matmul(gated, w2);
        }
        case SwigluFwPath::GateUp: {
            // Design A: 2-step pipeline
            // Step 1: M = SiLU(X @ W1) * (X @ W3) — custom fused gate-up kernel
            ttnn::Tensor M = swiglu_gate_up(input_tensor, w1, w3);
            // Step 2: Y = M @ W2 — standard matmul
            return ttnn::matmul(M, w2);
        }
        case SwigluFwPath::FullFusion:
            return ttnn::prim::ttml_swiglu_fw(
                input_tensor,  // [B, 1, S, C]
                w1,            // [1, 1, C, H]
                w2,            // [1, 1, H, C]
                w3,            // [1, 1, C, H]
                std::nullopt);
    }
    __builtin_unreachable();
}

}  // namespace ttml::metal
