// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The RoPE mirror alone: CBs as rope_mirror.h with in 0, rotated 1, cos 2, sin 3, scalar 4, rotated*(-1) 5, x*cos 6,
// rotated*sin 7, out 16.  Under an fp32 dest ROUND_RNE narrows every FPU result to bf16 on the SFPU before the pack.
// Runtime arg 0: rows.

#include "rope_mirror.h"
#ifdef ROUND_RNE
#include "api/compute/eltwise_unary/typecast.h"
static_assert(static_cast<uint32_t>(DataFormat::Float32) == 0 && static_cast<uint32_t>(DataFormat::Float16_b) == 5);
#endif

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    CircularBuffer scalar_cb(4);
    scalar_cb.wait_front(1);
    compute_kernel_hw_startup(1, 4, 5);
#ifdef ROUND_RNE
    // fp32 dest: the same sequence with an RNE narrowing in the dest before each pack
    CircularBuffer in_cb(0), rot_cb(1), cos_cb(2), sin_cb(3), rot_neg_cb(5), xcos_cb(6), rsin_cb(7), out_cb(16);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < 2; ++j) {
            uint32_t rot_src = 1;
            if (j == 0) {
                reconfig_data_format(1, 4);
                pack_reconfig_data_format(5);
                rot_cb.wait_front(1);
                rot_neg_cb.reserve_back(1);
                tile_regs_acquire();
                mul_bcast_scalar_init(1, 4);
                mul_tiles_bcast_scalar(1, 4, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, 5);
                tile_regs_release();
                rot_neg_cb.push_back(1);
                rot_cb.pop_front(1);
                rot_src = 5;
            }
            for (uint32_t k = 0; k < 2; ++k) {
                const uint32_t in0_cb_id = k == 0 ? rot_src : 0, in1_cb_id = k == 0 ? 3 : 2,
                               product_cb_id = k == 0 ? 7 : 6;
                CircularBuffer in0_cb(in0_cb_id), in1_cb(in1_cb_id), product_cb(product_cb_id);
                reconfig_data_format(in0_cb_id, in1_cb_id);
                pack_reconfig_data_format(product_cb_id);
                in0_cb.wait_front(1);
                in1_cb.wait_front(1);
                product_cb.reserve_back(1);
                tile_regs_acquire();
                mul_init(in0_cb_id, in1_cb_id);
                mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
                typecast_tile_init<0, 5>();
                typecast_tile<0, 5, true>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, product_cb_id);
                tile_regs_release();
                product_cb.push_back(1);
                in0_cb.pop_front(1);
                in1_cb.pop_front(1);
            }
            xcos_cb.wait_front(1);
            rsin_cb.wait_front(1);
            out_cb.reserve_back(1);
            reconfig_data_format(6, 7);
            pack_reconfig_data_format(16);
            tile_regs_acquire();
            add_init(6, 7);
            add_tiles(6, 7, 0, 0, 0);
            typecast_tile_init<0, 5>();
            typecast_tile<0, 5, true>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 16);
            tile_regs_release();
            out_cb.push_back(1);
            xcos_cb.pop_front(1);
            rsin_cb.pop_front(1);
        }
    }
#else
    rope64_rows<0, 1, 2, 3, 4, 5, 6, 7, 16>(rows);
#endif
}
