// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);

#ifndef FAST_TILIZE
    tilize_init(dfb::in, per_core_block_tile_cnt, dfb::out);
#else
    fast_tilize_init(dfb::in, per_core_block_tile_cnt, dfb::out);
#endif

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in.wait_front(per_core_block_tile_cnt);
        dfb_out.reserve_back(per_core_block_tile_cnt);

#ifndef FAST_TILIZE
        tilize_block(dfb::in, per_core_block_tile_cnt, dfb::out);
#else
        fast_tilize_block(dfb::in, per_core_block_tile_cnt, dfb::out);
#endif

        dfb_in.pop_front(per_core_block_tile_cnt);
        dfb_out.push_back(per_core_block_tile_cnt);
    }

#ifndef FAST_TILIZE
    tilize_uninit(dfb::in, dfb::out);
#else
    fast_tilize_uninit(dfb::in, dfb::out, per_core_block_tile_cnt);
#endif
}
