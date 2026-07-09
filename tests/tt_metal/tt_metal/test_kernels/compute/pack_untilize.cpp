// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_out0(dfb::out);

    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    compute_kernel_hw_startup(dfb::in, dfb::out);
    pack_untilize_init<block_ct_dim, full_ct_dim>(dfb::in, dfb::out);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        dfb_out0.reserve_back(full_ct_dim);

        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            dfb_in0.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(dfb::in, 1, dfb::out, b);
            dfb_in0.pop_front(block_ct_dim);
        }
        dfb_out0.push_back(full_ct_dim);
    }

    pack_untilize_uninit(dfb::out);
}
