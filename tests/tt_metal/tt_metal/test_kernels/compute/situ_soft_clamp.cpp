// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives one SiTU unary SFPU op over a tile at a time so a host test can compare
// the result against the torch reference. Both ops share a signature, so the host
// selects one by name and supplies beta as fp32 bit patterns. situ_tile and
// situ_tile_init are macros despite the lowercase spelling:
//   -Dsitu_tile=soft_clamp_tile -Dsitu_tile_init=soft_clamp_tile_init
//   compile_time_args = [num_tiles, beta_bits, beta_recip_bits]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/situ.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t beta_bits = get_compile_time_arg_val(1);
    constexpr uint32_t beta_recip_bits = get_compile_time_arg_val(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer in_cb(cb_in);
    CircularBuffer out_cb(cb_out);

    init_sfpu(cb_in, cb_out);
    // Hoisted out of the loop: this kernel runs a single SFPU op, so nothing else
    // reprograms the tanh constants between tiles.
    situ_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        in_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
        situ_tile(0, beta_bits, beta_recip_bits);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        in_cb.pop_front(1);
    }
}
