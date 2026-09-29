// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// post_attention compute (fp32 dest): the bf16-dest SFPU sigmoid of each gate tile packed bf16 (the chain's unary
// sigmoid), then the SFPU product attention x sigmoid with the bf16-dest rounding and zero rule (the chain's
// binary_ng multiply), packed bf16.  CBs: 0 gate, 1 attention, 2 sigmoid, 16 out (bf16, 8 each).

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/tile_move_copy.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t CB_GATE = 0, CB_ATT = 1, CB_SIG = 2, CB_OUT = 16, HEAD_TILES = 8, GROUP = 4;
    compute_kernel_hw_startup(CB_GATE, CB_ATT, CB_SIG);
    cb_wait_front(CB_GATE, HEAD_TILES);
    cb_wait_front(CB_ATT, HEAD_TILES);

    {
        FUSED_ZONE("fz_qs_pa_c_sigmoid");
        reconfig_data_format(CB_GATE, CB_GATE);
        pack_reconfig_data_format(CB_SIG);
        sigmoid_tile_init<false>();
        for (uint32_t b = 0; b < HEAD_TILES; b += GROUP) {
            cb_reserve_back(CB_SIG, GROUP);
            tile_regs_acquire();
            copy_init(CB_GATE);
            for (uint32_t i = 0; i < GROUP; ++i) {
                copy_tile(CB_GATE, b + i, i);
                sigmoid_tile<VectorMode::RC, false, false>(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < GROUP; ++i) {
                pack_tile(i, CB_SIG);
            }
            tile_regs_release();
            cb_push_back(CB_SIG, GROUP);
        }
        cb_pop_front(CB_GATE, HEAD_TILES);
    }

    {
        FUSED_ZONE("fz_qs_pa_c_mul");
        cb_wait_front(CB_SIG, HEAD_TILES);
        pack_reconfig_data_format(CB_OUT);
        mul_binary_tile_init();
        for (uint32_t b = 0; b < HEAD_TILES; b += 2) {
            cb_reserve_back(CB_OUT, 2);
            tile_regs_acquire();
            copy_init(CB_ATT);
            copy_tile(CB_ATT, b, 0);
            copy_tile(CB_ATT, b + 1, 2);
            copy_init(CB_SIG);
            copy_tile(CB_SIG, b, 1);
            mul_binary_tile<false>(0, 1, 0);
            copy_tile(CB_SIG, b + 1, 3);
            mul_binary_tile<false>(2, 3, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            pack_tile(2, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, 2);
        }
        cb_pop_front(CB_SIG, HEAD_TILES);
        cb_pop_front(CB_ATT, HEAD_TILES);
    }
}
