// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    const uint32_t B1 = get_arg_val<uint32_t>(0);
    const uint32_t B2 = get_arg_val<uint32_t>(1);
    const uint32_t Ht = get_arg_val<uint32_t>(2);
    const uint32_t Wt_per_core = get_arg_val<uint32_t>(3);
    const bool do_mask_h = get_arg_val<uint32_t>(4) == 1;
    const bool do_mask_w = get_arg_val<uint32_t>(5) == 1;

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_scaler = tt::CB::c_in1;
    constexpr auto cb_mask_h_w = tt::CB::c_in2;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr auto cb_intermed1 = tt::CB::c_intermed1;
    constexpr auto cb_out = tt::CB::c_out0;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);
    cb_wait_front(cb_scaler, onetile);

    if (do_mask_h || do_mask_w) {
        cb_wait_front(cb_mask_h_w, onetile * 2);
    }

    uint32_t B1B2Ht = B1 * B2 * Ht;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        bool enable_reload = false;
        uint32_t num_tile_done = 0;
        for (uint32_t b1 = 0; b1 < B1; ++b1) {
            for (uint32_t b2 = 0; b2 < B2; ++b2) {
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    bool last_row = (ht == Ht - 1);
                    bool last_col = (wt == Wt_per_core - 1);
                    bool last_out = (num_tile_done == B1B2Ht - 1);
                    bool do_mask = (do_mask_h && last_row) || (do_mask_w && last_col);

                    // get tile from reader
                    cb_wait_front(cb_in0, onetile);

                    if (do_mask) {
                        ACQ();
                        copy_tile_init();
                        copy_tile(cb_in0, 0, dst0);

                        if (do_mask_h && last_row) {
                            copy_tile_init();
                            copy_tile(cb_mask_h_w, 0, dst1);
                            mask_tile_init();
                            mask_tile(dst0, dst1);
                        }

                        if (do_mask_w && last_col) {
                            copy_tile_init();
                            copy_tile(cb_mask_h_w, 1, dst1);
                            mask_tile_init();
                            mask_tile(dst0, dst1);
                        }
                        cb_reserve_back(cb_intermed0, onetile);
                        pack_tile(dst0, cb_intermed0);
                        cb_push_back(cb_intermed0, onetile);
                        REL();
                    }

                    ACQ();
                    if (enable_reload) {
                        cb_wait_front(cb_intermed1, onetile);
                        copy_tile_init();
                        copy_tile(cb_intermed1, 0, 0);
                        cb_pop_front(cb_intermed1, onetile);
                    }

                    if (do_mask)
                        cb_wait_front(cb_intermed0, onetile);

                    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
                    reduce_tile(REDUCE_OP, REDUCE_DIM, (do_mask) ? (cb_intermed0) : (cb_in0), cb_scaler, 0, 0, 0);
                    reduce_revert_delta();

                    if (do_mask)
                        cb_pop_front(cb_intermed0, onetile);

                    cb_pop_front(cb_in0, onetile);

                    if (last_out) {
                        cb_reserve_back(cb_out, onetile);
                        pack_tile(0, cb_out);
                        cb_push_back(cb_out, onetile);

                    } else {
                        cb_reserve_back(cb_intermed1, onetile);
                        pack_tile(0, cb_intermed1);
                        cb_push_back(cb_intermed1, onetile);
                    }
                    REL();
                    enable_reload = true;
                    num_tile_done++;
                }
            }
        }
    }
}
}  // namespace NAMESPACE
