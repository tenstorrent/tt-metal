// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC: per pair, [4, 32] x tiles @ [K, gate | up] (ct = 2, LoFi, fp32 dest) -> silu(gate) * up * w on the
// valid rows -> one [4, 32] bf16 h tile.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_binary.h"
#endif

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route = get_compile_time_arg_val(3);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(4);
    constexpr uint32_t blocks_per_pair = get_compile_time_arg_val(5);
    constexpr uint32_t block_k = get_compile_time_arg_val(6);
    constexpr uint32_t ct = 2;  // gate, up
    constexpr uint32_t block_tiles = block_k * ct;
    constexpr uint32_t kt = block_k * blocks_per_pair;

    custom_mm_block_init<false, true, false>(cb_x, cb_w, cb_h, ct);
    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_route, 1);

    for (uint32_t p = 0; p < num_pairs; p++) {
        const uint32_t w_bits = read_tile_value(cb_route, 0, p);
        tile_regs_acquire();
        for (uint32_t b = 0; b < blocks_per_pair - 1; b++) {
            cb_wait_front(cb_w, block_tiles);
            custom_mm_block<false>(cb_x, cb_w, b * block_k, 0, 0, block_k, ct);
            cb_pop_front(cb_w, block_tiles);
        }
        cb_wait_front(cb_w, block_tiles);
        custom_mm_block<true>(cb_x, cb_w, (blocks_per_pair - 1) * block_k, 0, 0, block_k, ct);
        cb_pop_front(cb_w, block_tiles);

        // Only rows 0-3 are packed (tiny [4, 32] h tile); the SFPU ops cover the whole dest tile.
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, w_bits);
        tile_regs_commit();

        cb_reserve_back(cb_h, 1);
        tile_regs_wait();
        pack_tile(0, cb_h);
        tile_regs_release();
        cb_push_back(cb_h, 1);

        // The SFPU inits reprogram address modes the matmul relies on.
        custom_mm_block_init_short<false, true, false>(cb_x, cb_w, cb_h, ct);
    }

    cb_pop_front(cb_route, 1);
    cb_pop_front(cb_x, kt);
}
