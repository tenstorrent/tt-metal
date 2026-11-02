// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#ifdef DROPOUT_INPUT_FPU_ADD
#include "api/compute/eltwise_binary.h"
#endif
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    uint32_t seed = get_compile_time_arg_val(2);
    uint32_t int_probability = get_compile_time_arg_val(3);
    uint32_t int_scale_factor = get_compile_time_arg_val(4);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
#ifdef DROPOUT_INPUT_FPU_ADD
    // Fused FPU add + dropout: the tile dropout works on is produced in DST by the FPU (c_0 + c_0 = 2x),
    // so the FPU binary init, not a datacopy init, is the last ADDR_MOD_3 programmer before dropout_tile
    // (dropout_kernel_init below programs ADDR_MOD_7, the slot the SFPU op addresses Dest through).
    add_init(tt::CBIndex::c_0, tt::CBIndex::c_0);
#else
    copy_init(tt::CBIndex::c_0);
#endif
    dropout_kernel_init(seed);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb16.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Tile after tile: produce the tile in DST (datacopy, or FPU add of c_0 with itself),
            // apply dropout, pack, pop
            cb0.wait_front(1);

#ifdef DROPOUT_INPUT_FPU_ADD
            add_tiles(tt::CBIndex::c_0, tt::CBIndex::c_0, 0 /*itile0*/, 0 /*itile1*/, 0 /*idst*/);
#else
            copy_tile(tt::CBIndex::c_0, 0, 0);
#endif

            dropout_tile(0, int_probability, int_scale_factor);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, tt::CBIndex::c_16);

            cb0.pop_front(1);

            tile_regs_release();
        }
        cb16.push_back(per_core_block_dim);
    }
}
