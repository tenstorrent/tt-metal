// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"

/* This dummy initialization function is called prior to reduce_init to ensure proper
 * initialization of the HW and to test reduce_init_short/reduce_init_delta calls.
 *
 * - If SHORT_INIT is defined, this function provides API calls
 *   which initialize the HW properly when supplemented with reduce_init_short or
 *   reduce_init_delta (note that these two inits are the same except for the "at_start"
 *   argument; reference reduce.h for more details).
 * - If SHORT_INIT is not defined, only the PACK configuration function is called with
 *   a negative value of the defined "at_start" template argument because full reduce_init
 *   provides other API calls.
 *
 * If "at_start = 1", the value that is passed to llk_pack_reduce_config_v2 is 0.
 * If "at_start = 0", the value that is passed to llk_pack_reduce_config_v2 is 1.
 *
 * After dummy_init is called, the proper reduce init call will be invoked with the defined
 * value of the argument, not the negated value. This will ensure that the "at_start"
 * argument is tested. Reference llk_pack_reduce_config_v2 for more details.
 */
template <bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void dummy_init(uint32_t icb = 0, uint32_t icb_scaler = 1, uint32_t ocb = 16) {
#ifdef SHORT_INIT
    UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(icb, icb_scaler)));

    PACK((llk_pack_init()));
    PACK((llk_pack_dest_init<false, DST_ACCUM_MODE>()));
#endif
    PACK((llk_pack_reduce_config_v2<reduce_dim, !at_start, false, DST_ACCUM_MODE>(ocb)));
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr bool at_start = get_compile_time_arg_val(3);
    dummy_init<at_start>(tt::CBIndex::c_0, tt::CBIndex::c_2);
#ifndef SHORT_INIT
    reduce_init<at_start>(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
#else
    reduce_init_delta<at_start>(tt::CBIndex::c_16, tt::CBIndex::c_0, tt::CBIndex::c_2);
#endif

    cb_wait_front(tt::CBIndex::c_2, 1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            acquire_dst();
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(tt::CBIndex::c_0, onetile);
#if (MATH_ONLY == 1)
                UNPACK((llk_unpack_AB(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0)));
                // REDUCE_OP is expected to come from add_define
                reduce_tile_math(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP is expected to come from add_define
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
#endif
                cb_pop_front(tt::CBIndex::c_0, onetile);
            }

            cb_reserve_back(tt::CBIndex::c_16, onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_16);
            cb_push_back(tt::CBIndex::c_16, onetile);
            release_dst();
        }
    }
#ifdef SHORT_INIT
    reduce_revert_delta(tt::CBIndex::c_16);
#endif
}
}  // namespace NAMESPACE
