// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 5 compute (fp32 dest), one tile per core: the decode post_attention kernel's two ops on one tile
// -- the bf16-dest SFPU sigmoid of the gate tile packed bf16 (the chain's unary sigmoid), then the SFPU product
// attention x sigmoid with the bf16-dest rounding and zero rule (the chain's binary_ng multiply), packed bf16.
// CBs: 0 gate, 1 attention, 2 sigmoid, 16 out (bf16, 1 each).
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/tile_move_copy.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t CB_GATE = 0, CB_ATT = 1, CB_SIG = 2, CB_OUT = 16;
    compute_kernel_hw_startup(CB_GATE, CB_ATT, CB_SIG);
    cb_wait_front(CB_GATE, 1);
    cb_wait_front(CB_ATT, 1);
    {
        FUSED_ZONE("fz_qr_pa_c_sigmoid");
        reconfig_data_format(CB_GATE, CB_GATE);
        pack_reconfig_data_format(CB_SIG);
        sigmoid_tile_init<false>();
        cb_reserve_back(CB_SIG, 1);
        tile_regs_acquire();
        copy_init(CB_GATE);
        copy_tile(CB_GATE, 0, 0);
        sigmoid_tile<VectorMode::RC, false, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SIG);
        tile_regs_release();
        cb_push_back(CB_SIG, 1);
        cb_pop_front(CB_GATE, 1);
    }
    {
        FUSED_ZONE("fz_qr_pa_c_mul");
        cb_wait_front(CB_SIG, 1);
        pack_reconfig_data_format(CB_OUT);
        mul_binary_tile_init();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_acquire();
        copy_init(CB_ATT);
        copy_tile(CB_ATT, 0, 0);
        copy_init(CB_SIG);
        copy_tile(CB_SIG, 0, 1);
        mul_binary_tile<false>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_SIG, 1);
        cb_pop_front(CB_ATT, 1);
    }
}
