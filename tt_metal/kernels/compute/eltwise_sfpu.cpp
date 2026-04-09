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
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#ifdef ARCH_QUASAR
    constexpr uint32_t dfb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(3);
    experimental::DataflowBuffer dfb_in(dfb_in_id);
    experimental::DataflowBuffer dfb_out(dfb_out_id);
    init_sfpu(dfb_in.get_id(), dfb_out.get_id());
#else
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
#endif
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
#ifdef ARCH_QUASAR
        dfb_out.reserve_back(per_core_block_dim);
#else
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
#endif
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
#ifdef ARCH_QUASAR
            dfb_in.wait_front(1);
            copy_tile(dfb_in.get_id(), 0, 0);
#else
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);
#endif
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
#ifdef ARCH_QUASAR
            pack_tile(0, dfb_out.get_id());
            dfb_in.pop_front(1);
#else
            pack_tile(0, tt::CBIndex::c_16);
            cb_pop_front(tt::CBIndex::c_0, 1);
#endif
            tile_regs_release();
        }
#ifdef ARCH_QUASAR
        dfb_out.push_back(per_core_block_dim);
#else
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
#endif
    }
}
