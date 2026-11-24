// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 4 down compute (TRISC): y(v)[MT rows][DC columns] = h(v) [MT][HK] @ W_d[HK][DC] for each sub-block v, h from
// HIN_CB ([K][row] tiles), the weights from the two-expert ring (WD_CB, [DKB][DC] blocks), one row tile a pass (1 x DC
// in a bf16 half-sync DEST), y packed to OUT_CB (bfp8). The expert's weight blocks are freed after its last sub-block.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "kd_common.hpp"

using namespace kd;

uint32_t w_freed = 0, w_ready = 0;

FORCE_INLINE uint32_t w_block(uint32_t b) {
    if (b + 1 > w_ready) {
        cb_wait_front(WD_CB, (b + 1) * DBLK);
        w_ready = b + 1;
    }
    const uint32_t front = w_freed % DWS;
    return static_cast<uint32_t>(static_cast<int32_t>((front + b) % DWS) - static_cast<int32_t>(front)) * DBLK;
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(HIN_CB, WD_CB, OUT_CB);
    matmul_block_init(HIN_CB, WD_CB, false, DC, 1, HK);
    for (uint32_t v = 0; v < NSB; ++v) {
        cb_wait_front(HIN_CB, H_TILES);
        cb_reserve_back(OUT_CB, Y_TILES);
        for (uint32_t r = 0; r < MT; ++r) {
            tile_regs_acquire();
            for (uint32_t b = 0; b < DNB; ++b) {
                const uint32_t wo = w_block(b);
                for (uint32_t k = 0; k < DKB; ++k) {
                    matmul_block(HIN_CB, WD_CB, (b * DKB + k) * MT + r, wo + k * DC, 0, false, DC, 1, HK);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < DC; ++c) {
                pack_tile<true>(c, OUT_CB, r * DC + c);
            }
            tile_regs_release();
        }
        cb_push_back(OUT_CB, Y_TILES);
        cb_pop_front(HIN_CB, H_TILES);
        if (v % SBE == SBE - 1) {
            cb_pop_front(WD_CB, DNB * DBLK);
            w_freed += DNB;
            w_ready = 0;
        }
    }
}
