// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t Tr = get_compile_time_arg_val(0);
constexpr uint32_t n_heads = get_compile_time_arg_val(1);

constexpr auto cb_dkpe_in = tt::CBIndex::c_3;
constexpr auto cb_dkpe_out = tt::CBIndex::c_4;

constexpr uint32_t onetile = 1U;

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_dkpe_in, cb_dkpe_in, cb_dkpe_out);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        // DST registers 0..Tr-1 hold the running head-axis sum across the block;
        // slot Tr is the working temp for each new head's incoming tile.
        tile_regs_acquire();

        // Head 0: copy the first head's Tr tiles into the accumulator (dst 0..Tr-1).
        cb_wait_front(cb_dkpe_in, Tr);
        copy_tile_init(cb_dkpe_in);
        for (uint32_t w = 0U; w < Tr; ++w) {
            copy_tile(cb_dkpe_in, w, /* dst_idx */ w);
        }
        cb_pop_front(cb_dkpe_in, Tr);

        // Heads 1..H-1: load each tile into dst[Tr] and add into dst[w].
        for (uint32_t h = 1U; h < n_heads; ++h) {
            cb_wait_front(cb_dkpe_in, Tr);
            copy_tile_init(cb_dkpe_in);
            for (uint32_t w = 0U; w < Tr; ++w) {
                constexpr uint32_t tmp_dst = Tr;
                copy_tile(cb_dkpe_in, w, /* dst_idx */ tmp_dst);
                add_binary_tile_init();
                add_binary_tile(/* dst_a */ w, /* dst_b */ tmp_dst, /* dst_dest */ w);
            }
            cb_pop_front(cb_dkpe_in, Tr);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_dkpe_out, Tr);
        pack_reconfig_data_format(cb_dkpe_out);
        for (uint32_t w = 0U; w < Tr; ++w) {
            pack_tile(/* dst_idx */ w, cb_dkpe_out);
        }
        cb_push_back(cb_dkpe_out, Tr);

        tile_regs_release();
    }
}
