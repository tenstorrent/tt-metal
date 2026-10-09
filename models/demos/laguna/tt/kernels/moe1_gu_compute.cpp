// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE gate/up compute (Laguna): per active expert unit, gate = x @ Wg[:, nt] into DST 0 and
// up = x @ Wu[:, nt] into DST 1, then DST 0 = silu(gate) * up * routing weight, packed as one bf16 tile.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_meta = 2;
    constexpr uint32_t cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);

    cb_wait_front(cb_meta, 1);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    uint32_t rw[slots];
    for (uint32_t u = 0; u < n; ++u) {
        rw[u] = read_tile_value(cb_meta, 0, 1 + u);
    }
    cb_pop_front(cb_meta, 1);
    if (n == 0) {
        return;
    }

    cb_wait_front(cb_in0, Kt);
    for (uint32_t u = 0; u < n; ++u) {
        tile_regs_acquire();
        matmul_init(cb_in0, cb_w);
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t k0 = 0; k0 < Kt; k0 += chunk) {
                cb_wait_front(cb_w, chunk);
                for (uint32_t i = 0; i < chunk; ++i) {
                    matmul_tiles(cb_in0, cb_w, k0 + i, i, half);
                }
                cb_pop_front(cb_w, chunk);
            }
        }
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, rw[u]);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_in0, Kt);
}
