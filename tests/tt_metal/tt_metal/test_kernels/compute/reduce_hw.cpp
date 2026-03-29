// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    constexpr uint32_t onetile = 1;

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_in(0);
    experimental::DataflowBuffer dfb_in_scaler(1);
    experimental::DataflowBuffer dfb_out(2);
    compute_kernel_hw_startup(dfb_in.get_id(), dfb_in_scaler.get_id(), dfb_out.get_id());
    reduce_init(dfb_in.get_id(), dfb_in_scaler.get_id(), dfb_out.get_id());
#else
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
#endif

#ifdef ARCH_QUASAR
    dfb_in_scaler.wait_front(onetile);
#else
    cb2.wait_front(onetile);  // scaler tile from the reader
#endif
    for (uint32_t nc = 0; nc < NC; nc++) {
        int reduce_dst_idx = 0;
        acquire_dst();
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
#ifdef ARCH_QUASAR
                dfb_in.wait_front(onetile);
#if (MATH_ONLY == 1)
                UNPACK((llk_unpack_AB_reduce(dfb_in.get_id(), dfb_in_scaler.get_id(), 0, 0)));
                // REDUCE_OP is expected to come from add_define
                reduce_tile_math(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP is expected to come from add_define
                reduce_tile(dfb_in.get_id(), dfb_in_scaler.get_id(), 0, 0, reduce_dst_idx);
#endif
                dfb_in.pop_front(onetile);
#else
                cb0.wait_front(onetile);
#if (MATH_ONLY == 1)
                UNPACK((llk_unpack_AB_reduce(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0)));
                // REDUCE_OP is expected to come from add_define
                reduce_tile_math(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP is expected to come from add_define
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
#endif
                cb_pop_front(tt::CBIndex::c_0, onetile);
#endif
            }
        }
#ifdef ARCH_QUASAR
        dfb_out.reserve_back(onetile);
        pack_tile(reduce_dst_idx, dfb_out.get_id());
        dfb_out.push_back(onetile);
#else
        cb16.reserve_back(onetile);
        pack_tile(reduce_dst_idx, tt::CBIndex::c_16);
        cb16.push_back(onetile);
#endif
        release_dst();
    }
    reduce_uninit();
}
