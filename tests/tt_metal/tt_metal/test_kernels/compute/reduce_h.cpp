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
    compute_kernel_hw_startup(dfb::in_data, dfb::in_scaler, dfb::out);
    constexpr bool swap_operands = (REDUCE_DIM == ReduceDim::REDUCE_ROW) && (REDUCE_OP != PoolType::MAX);
    if constexpr (swap_operands) {
        reconfig_data_format(dfb::in_scaler, dfb::in_data);
    }
    reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::in_data, dfb::in_scaler, dfb::out);

    dfb_in_scaler.wait_front(onetile);
    for (uint32_t nc = 0; nc < NC; nc++) {
        int reduce_dst_idx = 0;
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                dfb_in.wait_front(onetile);
#if (MATH_ONLY == 1)
#ifdef ARCH_QUASAR
                UNPACK((llk_unpack_AB_reduce(dfb::in_data, dfb::in_scaler, 0, 0)));
#else
                UNPACK((llk_unpack_AB_reduce<REDUCE_OP, REDUCE_DIM>(dfb::in_data, dfb::in_scaler, 0, 0)));
#endif
                // REDUCE_OP and REDUCE_DIM are expected to come from add_define
                reduce_tile_math<REDUCE_OP, REDUCE_DIM>(reduce_dst_idx);
#elif (MATH_ONLY == 0)
                // REDUCE_OP and REDUCE_DIM are expected to come from add_define
                reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::in_data, dfb::in_scaler, 0, 0, reduce_dst_idx);
#endif
                dfb_in.pop_front(onetile);
            }
            dfb_out.reserve_back(onetile);
            pack_tile(reduce_dst_idx, dfb::out);
            dfb_out.push_back(onetile);
            tile_regs_commit();
            tile_regs_release();
        }
    }
    reduce_uninit();
}
