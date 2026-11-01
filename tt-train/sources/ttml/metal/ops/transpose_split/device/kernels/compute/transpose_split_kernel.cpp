// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transpose_split compute: one tile transpose per item; tail items (>= dx_items) are scaled by tail_scale.
// FUSE_LOWRANK (iteration 25): a dx item (t, r < rows_t) also adds (tail_scale * tail_t^T) @ w_r, the product of
// the row block's s_dh tile (the scaled transposed tail tile, computed once per new t into cb_a) and w's tile r,
// accumulated in fp32 DEST with the transposed tile and packed once: no separate lowrank_update pass over dx.

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 1;
constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Tt = get_compile_time_arg_val(1);
constexpr uint32_t rows_t = get_compile_time_arg_val(2);
constexpr uint32_t dx_items = get_compile_time_arg_val(3);
constexpr uint32_t scale_bits = get_compile_time_arg_val(4);
constexpr uint32_t dxt = get_compile_time_arg_val(5);  // dx tile columns
constexpr uint32_t one_bits = 0x3F800000U;
#ifdef FUSE_LOWRANK
constexpr uint32_t cb_tail_in = 2;
constexpr uint32_t cb_a = 3;
constexpr uint32_t cb_w = 4;
#endif

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    [[maybe_unused]] uint32_t cur_t = 0xFFFFFFFFU;
    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
#ifdef FUSE_LOWRANK
        const uint32_t t = work / dxt;
        const uint32_t r = work - t * dxt;
        const bool fused = (work < dx_items) && (r < rows_t);
        if (fused && t != cur_t) {
            // The row block's s_dh tile: the tail tile transposed and scaled, exactly the tail output's values.
            if (cur_t != 0xFFFFFFFFU) {
                cb_pop_front(cb_a, 1);
            }
            cb_wait_front(cb_tail_in, 1);
            tile_regs_acquire();
            transpose_init(cb_tail_in);
            transpose_tile(cb_tail_in, 0, 0);
            if constexpr (scale_bits != one_bits) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scale_bits);
            }
            tile_regs_commit();
            pack_and_push(0, cb_a);
            cb_pop_front(cb_tail_in, 1);
            cb_wait_front(cb_a, 1);
            cur_t = t;
        }
#endif
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
#ifdef FUSE_LOWRANK
        if (fused) {
            // DEST0 = s_dh_t @ w_r (fp32), DEST1 = the transposed input tile, DEST0 += DEST1 on the SFPU.
            cb_wait_front(cb_w, 1);
            reconfig_data_format(cb_a, cb_w);
            matmul_init(cb_a, cb_w, /*transpose*/ 0);
            matmul_tiles(cb_a, cb_w, 0, 0, 0);
            reconfig_data_format_srca(cb_in);
            transpose_init(cb_in);
            transpose_tile(cb_in, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            pack_and_push(0, cb_out);
            cb_pop_front(cb_w, 1);
            cb_pop_front(cb_in, 1);
            continue;
        }
        reconfig_data_format_srca(cb_in);
#endif
        transpose_init(cb_in);
        transpose_tile(cb_in, 0, 0);
        if constexpr (scale_bits != one_bits) {
            if (work >= dx_items) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scale_bits);
            }
        }
        tile_regs_commit();
        pack_and_push(0, cb_out);
        cb_pop_front(cb_in, 1);
    }
#ifdef FUSE_LOWRANK
    if (cur_t != 0xFFFFFFFFU) {
        cb_pop_front(cb_a, 1);
    }
#endif
}
