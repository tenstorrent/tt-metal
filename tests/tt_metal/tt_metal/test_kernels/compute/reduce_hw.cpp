// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "debug/dprint.h"

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
#ifndef SHORT_INIT
    reduce_init<true>(tt::CB::c_in0, tt::CB::c_in2);
#else
    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in2, tt::CB::c_out0);
    reduce_init_delta<false>(tt::CB::c_out0, tt::CB::c_in0, tt::CB::c_in2);
#endif

    cb_wait_front(tt::CB::c_in2, 1); // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        acquire_dst();
        for(uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for(uint32_t wt = 0; wt < Wt; ++wt) {
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
        }
        cb_reserve_back(tt::CB::c_out0, onetile);
        pack_tile(reduce_dst_idx, tt::CB::c_out0);
        cb_push_back(tt::CB::c_out0, onetile);
        release_dst();
    }
#ifdef SHORT_INIT
    reduce_revert_delta(tt::CB::c_out0);
#endif
}
}
