// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(3);
    experimental::DataflowBuffer buff_in(dfb_in_id);
    experimental::DataflowBuffer buff_out(dfb_out_id);
    const uint32_t in_id = buff_in.get_id();
    const uint32_t out_id = buff_out.get_id();
#else
    experimental::CircularBuffer buff_in(tt::CBIndex::c_0);
    experimental::CircularBuffer buff_out(tt::CBIndex::c_16);
    const uint32_t in_id = tt::CBIndex::c_0;
    const uint32_t out_id = tt::CBIndex::c_16;
#endif
    init_sfpu(in_id, out_id);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        buff_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            // Pop tile after tile, copy to DST and pack
            buff_in.wait_front(1);
            copy_tile(in_id, 0, 0);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_id);
            buff_in.pop_front(1);
            tile_regs_release();
        }
        buff_out.push_back(per_core_block_dim);
    }
}
