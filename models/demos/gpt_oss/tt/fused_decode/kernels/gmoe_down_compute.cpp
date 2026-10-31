// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE down, compute (dflash/grouped_moe.py GroupedMoE.down). Per unit (expert e, output column tile c;
// gmoe_down_reader.cpp): the expert's slots in groups of up to 4 [rt slots x 1 column] accumulated over the KD K tiles
// in DST (FP32), packed one tile per slot.
//
// compile-time args: [cb_meta_c, cb_a, cb_w, cb_out, KD, NU, MAXS]

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_meta = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t KD = get_compile_time_arg_val(4);
    constexpr uint32_t NU = get_compile_time_arg_val(5);
    constexpr uint32_t MAXS = get_compile_time_arg_val(6);
    constexpr uint32_t E = 32;
    constexpr uint32_t RT = 4;  // slots per DST block (FP32: 4 tiles)

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t u0 = meta[0], u1 = meta[1];

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);
    uint32_t loaded = E;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t e = u / NU, m = meta[2 + e];
        if (m == 0) {
            continue;
        }
        if (m > MAXS) {
            // a big expert (gmoe_down_reader.cpp): per group of MAXS slots, this core's units of it
            if (loaded != E) {
                cb_pop_front(cb_a, MAXS * KD);
                loaded = E;
            }
            const uint32_t ue1 = u1 < (e + 1) * NU ? u1 : (e + 1) * NU;
            for (uint32_t g0 = 0; g0 < m; g0 += MAXS) {
                const uint32_t n = m - g0 < MAXS ? m - g0 : MAXS;
                cb_wait_front(cb_a, MAXS * KD);
                for (uint32_t uu = u; uu < ue1; ++uu) {
                    cb_wait_front(cb_w, KD);
                    for (uint32_t g = 0; g < n; g += RT) {
                        const uint32_t rt = n - g < RT ? n - g : RT;
                        matmul_block_init(cb_a, cb_w, 0, 1, rt, KD);
                        tile_regs_acquire();
                        for (uint32_t k = 0; k < KD; ++k) {
                            matmul_block(cb_a, cb_w, g * KD + k, k, 0, false, 1, rt, KD);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t r = 0; r < rt; ++r) {
                            cb_reserve_back(cb_out, 1);
                            pack_tile(r, cb_out);
                            cb_push_back(cb_out, 1);
                        }
                        tile_regs_release();
                    }
                    cb_pop_front(cb_w, KD);
                }
                cb_pop_front(cb_a, MAXS * KD);
            }
            u = ue1 - 1;
            continue;
        }
        if (e != loaded) {
            if (loaded != E) {
                cb_pop_front(cb_a, MAXS * KD);
            }
            cb_wait_front(cb_a, MAXS * KD);
            loaded = e;
        }
        cb_wait_front(cb_w, KD);
        for (uint32_t g = 0; g < m; g += RT) {
            const uint32_t rt = m - g < RT ? m - g : RT;
            matmul_block_init(cb_a, cb_w, 0, 1, rt, KD);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KD; ++k) {
                matmul_block(cb_a, cb_w, g * KD + k, k, 0, false, 1, rt, KD);
            }
            tile_regs_commit();
            tile_regs_wait();
            // one tile per push (several per push corrupted one landing on the CB's last page)
            for (uint32_t r = 0; r < rt; ++r) {
                cb_reserve_back(cb_out, 1);
                pack_tile(r, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_w, KD);
    }
    if (loaded != E) {
        cb_pop_front(cb_a, MAXS * KD);
    }
    cb_pop_front(cb_meta, 1);
}
