// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill rotation, compute. Output tile c of the block = sum_a H32[c, a] * (X_a @ diag(d_a) H32 / 32): one matmul per
// input tile with the weight tile of sign H32[c, a] = (-1)^popcount(c & a). The weight entries are +-1/32 (one bit)
// and HiFi3 covers the whole bf16 activation, so every product is exact; fp32 accumulation.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t m_count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_h = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    if (m_count == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_h, cb_x, cb_out);  // SrcA = in1 (the weight tiles), SrcB = in0 (x)
    matmul_init(cb_x, cb_h);
    cb_wait_front(cb_h, 64);
    for (uint32_t m = 0; m < m_count; ++m) {
        cb_wait_front(cb_x, 32);
        for (uint32_t c = 0; c < 32; ++c) {
            tile_regs_acquire();
            for (uint32_t a = 0; a < 32; ++a) {
                matmul_tiles(cb_x, cb_h, a, 2 * a + (__builtin_popcount(c & a) & 1), 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_x, 32);
    }
}
