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
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);


    auto cb_input = tt::CB::c_in0;
    constexpr auto cb_scaler = tt::CB::c_in2;
    constexpr auto cb_mask_w = tt::CB::c_in3;
    constexpr auto cb_accum_dst = tt::CB::c_intermed0;
    constexpr auto cb_masked_input = tt::CB::c_intermed1;
    constexpr auto cb_out = tt::CB::c_out0;
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    binary_op_init_common(cb_input, cb_input);

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            cb_input = tt::CB::c_in0;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                ACQ();
                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    cb_wait_front(cb_input, onetile);

                    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
                    reduce_tile(REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                    reduce_revert_delta();

                    cb_pop_front(cb_input, onetile);
                }
                cb_reserve_back(cb_accum_dst, onetile);
                pack_tile(reduce_dst_idx, cb_accum_dst);
                cb_push_back(cb_accum_dst, onetile);
                REL();
            }

            if (do_mask_w) {
                ACQ();
                cb_wait_front(cb_input, onetile);
                copy_tile_init();
                copy_tile(cb_input, 0, reduce_dst_idx);
                copy_tile(cb_mask_w, 0, mask_dst_idx);
                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);

                cb_reserve_back(cb_masked_input, onetile);
                pack_tile(reduce_dst_idx, cb_masked_input);
                cb_push_back(cb_masked_input, onetile);

                cb_pop_front(cb_input, onetile);
                cb_input = cb_masked_input;
                REL();
            }

            ACQ();
            cb_wait_front(cb_input, onetile);
            if (!is_w_single_tile) {
                cb_wait_front(cb_accum_dst, onetile);
                copy_tile_init();
                copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

            reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
            reduce_tile(REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, 0, 0, reduce_dst_idx);
            reduce_revert_delta();

            cb_reserve_back(cb_out, onetile);
            pack_tile(reduce_dst_idx, cb_out);
            cb_push_back(cb_out, onetile);

            cb_pop_front(cb_input, onetile);
            if (!is_w_single_tile) {
                cb_pop_front(cb_accum_dst, onetile);
            }
            REL();
        }
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }
    cb_pop_front(cb_scaler, onetile);
}
}  // namespace NAMESPACE
