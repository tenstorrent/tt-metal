// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);

    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_in(dfb::in_data);
    DataflowBuffer dfb_in_scaler(dfb::in_scaler);
    DataflowBuffer dfb_out(dfb::out);
    compute_kernel_hw_startup(dfb_in.get_id(), dfb_in_scaler.get_id(), dfb_out.get_id());
    reduce_init<REDUCE_OP, REDUCE_DIM>(dfb_in.get_id(), dfb_in_scaler.get_id(), dfb_out.get_id());

    dfb_in_scaler.wait_front(onetile);
    for (uint32_t nc = 0; nc < NC; nc++) {
        int reduce_dst_idx = 0;
        acquire_dst();
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                dfb_in.wait_front(onetile);
#if (MATH_ONLY == 1)
                UNPACK((llk_unpack_AB_reduce(dfb_in.get_id(), dfb_in_scaler.get_id(), 0, 0)));
                // REDUCE_OP and REDUCE_DIM are expected to come from add_define
                reduce_tile_math<REDUCE_OP, REDUCE_DIM>(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP and REDUCE_DIM are expected to come from add_define
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb_in.get_id(), dfb_in_scaler.get_id(), 0, 0, reduce_dst_idx);
#endif
                dfb_in.pop_front(onetile);
            }
        }
        dfb_out.reserve_back(onetile);
        pack_tile(reduce_dst_idx, dfb_out.get_id());
        dfb_out.push_back(onetile);
        release_dst();
    }
    reduce_uninit();
}
