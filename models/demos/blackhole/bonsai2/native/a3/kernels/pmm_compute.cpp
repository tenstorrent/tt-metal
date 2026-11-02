// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Short-prefill ternary matmul, compute. Per (G, column): dequantize the 32 code tiles into cb_w
// (code * row-broadcast scale, row (G%4)*8 + t/4 of the scale tile), reload the column's MT fp32
// partials from cb_p (G > 0), accumulate MT x 32 tile matmuls, and pack the partials back to cb_p or,
// at the last block, bf16 to cb_out. MT <= 4 = the fp32 DST half.
// Runtime args: nc.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t nc = get_arg_val<uint32_t>(0);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t dbg = get_compile_time_arg_val(2);  // 1: drop the partial reload (debug)
    constexpr uint32_t cb_x = 0, cb_c = 1, cb_s = 2, cb_w = 3, cb_p = 4, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    for (uint32_t G = 0; G < ng; ++G) {
        cb_wait_front(cb_x, MT * 32);
        for (uint32_t c = 0; c < nc; ++c) {
            // dequant
            cb_wait_front(cb_s, 1);
            cb_wait_front(cb_c, 32);
            reconfig_data_format(cb_c, cb_s);  // srcA codes, srcB scales
            mul_bcast_rows_init(cb_c, cb_s);
            pack_reconfig_data_format(cb_w);
            cb_reserve_back(cb_w, 32);
            for (uint32_t q = 0; q < 8; ++q) {
                tile_regs_acquire();
                for (uint32_t t = 0; t < 4; ++t) {
                    const uint32_t i = q * 4 + t;
                    mul_tiles_bcast_rows(cb_c, cb_s, i, 0, t, (G % 4) * 8 + i / 4);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 4; ++t) {
                    pack_tile(t, cb_w);
                }
                tile_regs_release();
            }
            cb_push_back(cb_w, 32);
            cb_pop_front(cb_c, 32);
            cb_pop_front(cb_s, 1);
            // matmul on top of the reloaded partials
            cb_wait_front(cb_w, 32);
            tile_regs_acquire();
            if (G > 0) {
                cb_wait_front(cb_p, MT);
                if constexpr (dbg == 0) {
                    reconfig_data_format_srca(cb_p);
                    copy_init(cb_p);
                    for (uint32_t m = 0; m < MT; ++m) {
                        copy_tile(cb_p, m, m);
                    }
                }
                cb_pop_front(cb_p, MT);
            }
            reconfig_data_format(cb_w, cb_x);  // matmul: srcA = in1 (cb_w), srcB = in0 (cb_x)
            matmul_block_init(cb_x, cb_w, 0, 1, MT, 32);  // one W tile serves all MT row tiles
            for (uint32_t t = 0; t < 32; ++t) {
                matmul_block(cb_x, cb_w, t, t, 0, false, 1, MT, 32);
            }
            tile_regs_commit();
            const uint32_t ocb = G == ng - 1 ? cb_out : cb_p;
            pack_reconfig_data_format(ocb);
            cb_reserve_back(ocb, MT);
            tile_regs_wait();
            for (uint32_t m = 0; m < MT; ++m) {
                pack_tile(m, ocb);
            }
            tile_regs_release();
            cb_push_back(ocb, MT);
            cb_pop_front(cb_w, 32);
        }
        cb_pop_front(cb_x, MT * 32);
    }
}
