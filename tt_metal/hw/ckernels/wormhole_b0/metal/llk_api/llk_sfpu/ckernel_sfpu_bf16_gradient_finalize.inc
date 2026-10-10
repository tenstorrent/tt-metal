// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void fuse_grad_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat y = dst_reg[d] * dst_reg[32 + d];
        y = convert<vFloat16b>(y, RoundMode::Nearest);
        dst_reg[d] = y;
    }
}
