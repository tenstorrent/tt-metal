// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

namespace NAMESPACE {
void MAIN {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_scale_mask = tt::CBIndex::c_27;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    cb_wait_front(cb_bcast_scaler, 1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_wait_front(cb_fused_scale, 1);
#endif

    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            // apply fused scale [*= 1/sqrt(...)]
            ACQ();
            mul_tiles_bcast_scalar_init_short(cb_in0, cb_fused_scale);
            cb_wait_front(cb_in0, ndst);
            cb_reserve_back(cb_scale_mask, ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                mul_tiles_bcast_scalar(cb_in0, cb_fused_scale, wt8, 0, wt8);  // mul bcast-HW -> DST[wt8]
                pack_tile(wt8, cb_scale_mask);                                // reuse exps buffer
            }
            cb_push_back(cb_scale_mask, ndst);
            cb_pop_front(cb_in0, ndst);
            REL();
        }

        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            ACQ();
            if (wait_mask) {
                cb_wait_front(cb_fused_attn, wt + ndst);  // cumulative wait for up to Wt tiles, only at first ht
            }
            cb_wait_front(cb_scale_mask, ndst);
            add_bcast_rows_init_short(cb_scale_mask, cb_fused_attn);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_tiles_bcast_rows(cb_scale_mask, cb_fused_attn, wt8, wt + wt8, wt8);  // tile *= 1/(sum(exp(x)))
            }
            cb_pop_front(cb_scale_mask, ndst);
            cb_reserve_back(cb_exps, ndst);
            exp_tile_init<true>();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                exp_tile<true>(wt8);      // exp on DST[0]
                pack_tile(wt8, cb_exps);  // reuse the exps buffer again, this time in a circular manner
            }
            cb_push_back(cb_exps, ndst);
            REL();
        }
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            cb_pop_front(cb_fused_attn, Wt);
            ht = 0;
            wait_mask = true;
        }
#else

        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            ACQ();
            cb_wait_front(cb_in0, ndst);
            copy_tile_init();  // need to copy from CB to DST to be able to run sfpu math
            for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                copy_tile(cb_in0, wt8, wt8);  // copy from c_in[0] to DST[0]
            }
            cb_pop_front(cb_in0, ndst);

            cb_reserve_back(cb_exps, ndst);
            exp_tile_init<true>();
            for (uint32_t wt8 = 0; wt8 < ndst; ++wt8) {
                exp_tile<true>(wt8);      // exp on DST[0]
                pack_tile(wt8, cb_exps);  // DST[0]->cb_id[wt]
            }
            cb_push_back(cb_exps, ndst);
            REL();
        }
#endif

        ACQ();
        cb_reserve_back(cb_recipsumexps, onetile);
        reduce_init_delta<false>();
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(cb_exps, wt + 1);        // must be a cumulative wait for correctness
            constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
            reduce_tile(cb_exps, cb_bcast_scaler, wt, bcast_scaler0, dst0);
        }
        reduce_revert_delta();
        recip_tile_init();
        recip_tile(dst0);  // DST[0] = 1/sum(exp(x))
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, 1);

        REL();

        cb_wait_front(cb_recipsumexps, 1);  // will reuse Wt times for bcast

        // now cb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a umulative wait for Wt tiles in cb_exps
        mul_bcast_cols_init_short(cb_exps, cb_recipsumexps);
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            ACQ();
            cb_reserve_back(tt::CBIndex::c_16, ndst);
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                // wt+wt8 since we pop Wt after the entire loop
                mul_tiles_bcast<BroadcastType::COL>(
                    cb_exps, cb_recipsumexps, wt + wt8, 0, wt8);  // tile *= 1/(sum(exp(x)))
                pack_tile(wt8, tt::CBIndex::c_16);
            }
            cb_push_back(tt::CBIndex::c_16, ndst);
            REL();
        }
        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_exps, Wt);
    }  // NCHt loop
    // cb_pop_front(cb_bcast_scaler, 1); // we don't actually have to do this
    // cb_pop_front(cb_fused_scale, 1); // we don't actually have to do this
}
}  // namespace NAMESPACE
