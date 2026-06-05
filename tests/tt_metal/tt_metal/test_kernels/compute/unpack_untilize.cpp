// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_out0(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    untilize_init(dfb::in);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in0.wait_front(per_core_block_tile_cnt);
        dfb_out0.reserve_back(per_core_block_tile_cnt);

        untilize_block(dfb::in, per_core_block_tile_cnt, dfb::out);

        dfb_out0.push_back(per_core_block_tile_cnt);
        dfb_in0.pop_front(per_core_block_tile_cnt);
    }

    untilize_uninit(dfb::in);
}
