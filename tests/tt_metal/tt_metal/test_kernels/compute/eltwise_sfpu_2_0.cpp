// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    DataflowBuffer buff_in(dfb::in);
    DataflowBuffer buff_out(dfb::out);
    init_sfpu(dfb::in, dfb::out);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        buff_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            buff_in.wait_front(1);
            copy_tile(dfb::in, 0, 0);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            buff_in.pop_front(1);
            tile_regs_release();
        }
        buff_out.push_back(per_core_block_dim);
    }
}
