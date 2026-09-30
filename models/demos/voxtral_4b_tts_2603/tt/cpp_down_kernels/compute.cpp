// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ down projection (see tt/cpp_down.py): y = x @ w2.
//
// The MT x PN output block is computed as PN / CT subblocks of MT x CT, each fitting DEST
// (full-sync fp32, 8 tiles). With ONE subblock DEST is acquired once and every K block accumulates
// into it; with more, K-partials spill to an fp32 CB between K blocks, as in cpp_swiglu.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PN = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t NSUB = PN / CT;
    constexpr uint32_t SUB = MT * CT;

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;
    constexpr auto cb_p = tt::CBIndex::c_24;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_y);
    matmul_block_init(cb_x, cb_w, 0, CT, MT, KB);

    if constexpr (NSUB == 1) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < NB; ++b) {
            cb_wait_front(cb_x, MT * KB);
            cb_wait_front(cb_w, KB * PN);
            uint32_t i1 = 0;
            for (uint32_t k = 0; k < KB; ++k) {
                matmul_block(cb_x, cb_w, k, i1, 0, 0, CT, MT, KB);
                i1 += PN;
            }
            cb_pop_front(cb_x, MT * KB);
            cb_pop_front(cb_w, KB * PN);
        }
        tile_regs_commit();
        cb_reserve_back(cb_y, SUB);
        tile_regs_wait();
        for (uint32_t i = 0; i < SUB; ++i) {
            pack_tile(i, cb_y);
        }
        tile_regs_release();
        cb_push_back(cb_y, SUB);
        return;
    }

    for (uint32_t b = 0; b < NB; ++b) {
        const bool last = b == NB - 1;
        cb_wait_front(cb_x, MT * KB);
        cb_wait_front(cb_w, KB * PN);
        for (uint32_t s = 0; s < NSUB; ++s) {
            tile_regs_acquire();
            if (b > 0) {
                reconfig_data_format_srca(cb_w, cb_p);
                copy_init(cb_p);
                cb_wait_front(cb_p, SUB);
                for (uint32_t i = 0; i < SUB; ++i) {
                    copy_tile(cb_p, i, i);
                }
                cb_pop_front(cb_p, SUB);
                reconfig_data_format_srca(cb_p, cb_w);
                matmul_block_init(cb_x, cb_w, 0, CT, MT, KB);
            }
            uint32_t i1 = s * CT;
            for (uint32_t k = 0; k < KB; ++k) {
                matmul_block(cb_x, cb_w, k, i1, 0, 0, CT, MT, KB);
                i1 += PN;
            }
            tile_regs_commit();
            const auto cb_out = last ? cb_y : cb_p;
            cb_reserve_back(cb_out, SUB);
            tile_regs_wait();
            for (uint32_t i = 0; i < SUB; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, SUB);
        }
        cb_pop_front(cb_x, MT * KB);
        cb_pop_front(cb_w, KB * PN);
    }
}
