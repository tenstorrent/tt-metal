// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// widen_partial compute (fp32 dest): each bf16 tile through the dest and packed fp32 (exact widening = the chain's
// typecast).  CBs: 0 in (bf16), 16 out (fp32).  Runtime arg 0: tiles.

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "../../kernels/zones.h"

void kernel_main() {
    FUSED_ZONE("fz_qs_wid_c_main");
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t CB_IN = 0, CB_OUT = 16;
    compute_kernel_hw_startup(CB_IN, CB_IN, CB_OUT);
    reconfig_data_format(CB_IN, CB_IN);
    pack_reconfig_data_format(CB_OUT);
    copy_init(CB_IN);
    cb_wait_front(CB_IN, tiles);
    cb_reserve_back(CB_OUT, tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_tile(CB_IN, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, tiles);
    cb_pop_front(CB_IN, tiles);
}
