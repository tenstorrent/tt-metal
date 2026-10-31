// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE gate|up + SwiGLU, compute (dflash/grouped_moe.py GroupedMoE.gate_up). Per unit (expert e, column tile
// j; gmoe_gu_reader.cpp): for each pair of the expert's slots, gate and up [2 slots x 1 column] = the 32-row bias
// tiles + the sums over the K tiles, in DST (FP32), then GPT-OSS SwiGLU (gmoe_swiglu.hpp) into the gate tile, packed
// one tile per slot.
//
// compile-time args: [cb_meta_c, cb_x, cb_w, cb_b, cb_out, KT, NU]

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "gmoe_swiglu.hpp"

void kernel_main() {
    constexpr uint32_t cb_meta = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_b = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t KT = get_compile_time_arg_val(5);
    constexpr uint32_t NU = get_compile_time_arg_val(6);
    constexpr uint32_t E = 32;
    constexpr uint32_t GROUP = 4;  // slots per in0 block (gmoe_gu_reader.cpp)

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t u0 = meta[0], u1 = meta[1];

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    // slots [p0, p0 + gm) of the in0 block at the front (cb_x) x the unit's weights / bias at the front -> cb_out
    auto block = [&](uint32_t gm) {
        for (uint32_t p = 0; p < gm; p += 2) {
            const uint32_t rt = gm - p < 2 ? gm - p : 2;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_b);
            copy_init(cb_b);
            for (uint32_t r = 0; r < rt; ++r) {
                copy_tile(cb_b, 0, 2 * r);
                copy_tile(cb_b, 1, 2 * r + 1);
            }
            reconfig_data_format(cb_w, cb_x);
            matmul_block_init(cb_x, cb_w, 0, 2, rt, KT);
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w, p * KT + k, 2 * k, 0, false, 2, rt, KT);
            }
            MATH((ckernel::gmoe_swiglu_init()));
            for (uint32_t r = 0; r < rt; ++r) {
                MATH((ckernel::gmoe_swiglu<DST_ACCUM_MODE>(2 * r, 2 * r + 1, 2 * r)));
            }
            tile_regs_commit();
            pack_reconfig_data_format(cb_out);
            tile_regs_wait();
            for (uint32_t r = 0; r < rt; ++r) {
                cb_reserve_back(cb_out, 1);
                pack_tile(2 * r, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
        }
    };
    uint32_t loaded = E;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t e = u / NU, m = meta[2 + e];
        if (m == 0) {
            continue;
        }
#ifdef GMOE_GU_GMAJOR
        if (m > GROUP) {
            // a big expert (512 / 1024-row passes; gmoe_gu_reader.cpp): per group of GROUP slots, this core's units
            if (loaded != E) {
                cb_pop_front(cb_x, GROUP * KT);
                loaded = E;
            }
            const uint32_t ue1 = u1 < (e + 1) * NU ? u1 : (e + 1) * NU;
            for (uint32_t g = 0; g < m; g += GROUP) {
                cb_wait_front(cb_x, GROUP * KT);
                for (uint32_t uu = u; uu < ue1; ++uu) {
                    cb_wait_front(cb_w, 2 * KT);
                    cb_wait_front(cb_b, 2);
                    block(m - g < GROUP ? m - g : GROUP);
                    cb_pop_front(cb_w, 2 * KT);
                    cb_pop_front(cb_b, 2);
                }
                cb_pop_front(cb_x, GROUP * KT);
            }
            u = ue1 - 1;
            continue;
        }
#endif
        if (m <= GROUP && e != loaded) {
            if (loaded != E) {
                cb_pop_front(cb_x, GROUP * KT);
            }
            cb_wait_front(cb_x, GROUP * KT);
            loaded = e;
        }
        if (m > GROUP && loaded != E) {
            cb_pop_front(cb_x, GROUP * KT);
            loaded = E;
        }
        cb_wait_front(cb_w, 2 * KT);
        cb_wait_front(cb_b, 2);
        for (uint32_t g = 0; g < m; g += GROUP) {
            const uint32_t gm = m - g < GROUP ? m - g : GROUP;
            if (m > GROUP) {
                cb_wait_front(cb_x, GROUP * KT);
            }
            for (uint32_t p = 0; p < gm; p += 2) {
                const uint32_t rt = gm - p < 2 ? gm - p : 2;
                // DST (r, c) = slot p + r's gate (c = 0) / up (c = 1): the 32-row bias tiles copied in first, the
                // matmul accumulates onto them (FP32), SwiGLU in place (no BF16 round trip of the sums and add pass)
                tile_regs_acquire();
                reconfig_data_format_srca(cb_b);
                copy_init(cb_b);
                for (uint32_t r = 0; r < rt; ++r) {
                    copy_tile(cb_b, 0, 2 * r);
                    copy_tile(cb_b, 1, 2 * r + 1);
                }
                // (the inits do not switch unpacker formats: state_configure is a no-op outside the sentinel build)
                reconfig_data_format(cb_w, cb_x);  // matmul: SrcA <- in1 (BFP4 weights), SrcB <- in0 (BFP8 rows)
                matmul_block_init(cb_x, cb_w, 0, 2, rt, KT);
                for (uint32_t k = 0; k < KT; ++k) {
                    matmul_block(cb_x, cb_w, p * KT + k, 2 * k, 0, false, 2, rt, KT);
                }
                MATH((ckernel::gmoe_swiglu_init()));
                for (uint32_t r = 0; r < rt; ++r) {
                    MATH((ckernel::gmoe_swiglu<DST_ACCUM_MODE>(2 * r, 2 * r + 1, 2 * r)));
                }
                tile_regs_commit();
                pack_reconfig_data_format(cb_out);
                tile_regs_wait();
                // one tile per push (two tiles per push corrupted the second when it was the CB's last page)
                for (uint32_t r = 0; r < rt; ++r) {
                    cb_reserve_back(cb_out, 1);
                    pack_tile(2 * r, cb_out);
                    cb_push_back(cb_out, 1);
                }
                tile_regs_release();
            }
            if (m > GROUP) {
                cb_pop_front(cb_x, GROUP * KT);
            }
        }
        cb_pop_front(cb_w, 2 * KT);
        cb_pop_front(cb_b, 2);
    }
    if (loaded != E) {
        cb_pop_front(cb_x, GROUP * KT);
    }
    cb_pop_front(cb_meta, 1);
}
