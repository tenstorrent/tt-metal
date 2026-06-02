// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_out0(dfb::out);

    compute_kernel_hw_startup(dfb_in0.get_id(), dfb_out0.get_id());
    untilize_init(dfb_in0.get_id());

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in0.wait_front(per_core_block_tile_cnt);
        dfb_out0.reserve_back(per_core_block_tile_cnt);

        untilize_block(dfb_in0.get_id(), per_core_block_tile_cnt, dfb_out0.get_id());

        dfb_out0.push_back(per_core_block_tile_cnt);
        dfb_in0.pop_front(per_core_block_tile_cnt);
    }

    untilize_uninit(dfb_in0.get_id());
#else
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    untilize_init(tt::CBIndex::c_0);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb0.wait_front(per_core_block_tile_cnt);
        cb16.reserve_back(per_core_block_tile_cnt);

        untilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        cb16.push_back(per_core_block_tile_cnt);
        cb0.pop_front(per_core_block_tile_cnt);
    }

    untilize_uninit(tt::CBIndex::c_0);
#endif
}
