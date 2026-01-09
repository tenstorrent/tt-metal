// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    cb2.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            acquire_dst();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                cb0.wait_front(onetile);
#if (MATH_ONLY == 1)
                UNPACK((llk_unpack_AB(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0)));
                // REDUCE_OP is expected to come from add_define
                reduce_tile_math(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP is expected to come from add_define
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
#endif
                cb0.pop_front(onetile);
            }

            cb16.reserve_back(onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_16);
            cb16.push_back(onetile);
            release_dst();
        }
    }
    reduce_uninit();
}
}  // namespace NAMESPACE
