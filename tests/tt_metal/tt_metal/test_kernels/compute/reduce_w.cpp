// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
    reduce_max_row_init();

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
                reduce_tile_max_row(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, reduce_dst_idx);
                cb_pop_front(tt::CBIndex::c_0, onetile);
            }
            // dprint_tensix_dest_reg(0);

            cb_reserve_back(tt::CBIndex::c_16, onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_16);
            cb_push_back(tt::CBIndex::c_16, onetile);
            release_dst();
        }
    }
    reduce_uninit();
}
}  // namespace NAMESPACE
