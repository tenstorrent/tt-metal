// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void fuse_grad_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        // Verifier-owned backward composition: a zero derivative suppresses
        // every incoming gradient class to canonical +0; a nonzero derivative
        // copies the gradient and optionally applies the declared fixed scale.
        // This runs after the selected evaluator and overlay have retired, so
        // it preserves the common two-row fused transport and LREG budget.
        vFloat factor = dst_reg[d];
        vFloat y = 0.0f;
        v_if(factor != 0.0f) {
            y = dst_reg[32 + d];
            y = y * TT_INLINE_PROGRAM_GRADIENT_SCALE;
        }
        v_endif;
        y = convert<vFloat16b>(y, RoundMode::Nearest);
        dst_reg[d] = y;
    }
}
