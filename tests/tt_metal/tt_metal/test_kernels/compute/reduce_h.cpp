// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"

template<bool not_at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void dummy_init(uint32_t icb = 0, uint32_t icb_scaler = 1, uint32_t ocb = 16)
{
#ifdef SHORT_INIT
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));

    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));
#endif
    PACK(( llk_pack_reduce_config_v2<reduce_dim, not_at_start, false, DST_ACCUM_MODE>(ocb) ));
}

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr bool at_start = get_compile_time_arg_val(3);
    dummy_init<!at_start>(tt::CB::c_in0, tt::CB::c_in2);
#ifndef SHORT_INIT
    reduce_init<at_start>(tt::CB::c_in0, tt::CB::c_in2);
#else
    reduce_init_delta<at_start>(tt::CB::c_out0, tt::CB::c_in0, tt::CB::c_in2);
#endif

    cb_wait_front(tt::CB::c_in2, 1); // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            acquire_dst();
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                cb_wait_front(tt::CB::c_in0, onetile);
#if (MATH_ONLY == 1)
                UNPACK(( llk_unpack_AB(tt::CB::c_in0, tt::CB::c_in2, 0, 0) ));
                // REDUCE_OP is expected to come from add_define
                reduce_tile_math(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP is expected to come from add_define
                reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
#endif
                cb_pop_front(tt::CB::c_in0, onetile);
            }

            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);
            release_dst();
        }
    }
#ifdef SHORT_INIT
    reduce_revert_delta(tt::CB::c_out0);
#endif
}
}
