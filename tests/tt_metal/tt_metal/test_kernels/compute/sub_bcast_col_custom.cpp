// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SDPA blocked bcast-col SUB with SrcB reuse, driven through the Compute API.
//
// Per block, ONE held srcB tile is subtracted (column-broadcast) from CT_DIM srcA column tiles,
// each landing in its own dest slot. This reuse that distinguishes this op from
// calling the stock one-tile broadcast CT_DIM times.
//
// Mirrors the call sequence in sub_exp_block_bcast_cols() in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp.

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/dataflow/dataflow_buffer.h"

#ifndef CT_DIM
#define CT_DIM 1
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1
#endif

void kernel_main() {
    constexpr std::uint32_t ct_dim = CT_DIM;
    constexpr std::uint32_t num_blocks = NUM_BLOCKS;
    constexpr std::uint32_t onetile = 1;

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    constexpr std::uint32_t icb0 = dfb::in0;
    constexpr std::uint32_t icb1 = dfb::in1;
    constexpr std::uint32_t ocb = dfb::out;

    binary_op_init_common(icb0, icb1, ocb);
    sub_bcast_cols_init_short_custom(icb0, icb1, ct_dim);

    // Held for the whole run; never popped per block, so tile_index_b stays 0 below.
    dfb1.wait_front(onetile);

    for (std::uint32_t block = 0; block < num_blocks; block++) {
        // A whole block must be resident: the op unpacks all ct_dim srcA tiles in one call.
        dfb0.wait_front(ct_dim);
        dfb_out.reserve_back(ct_dim);

        tile_regs_acquire();
        // tile_index_a is relative to the buffer read pointer, which pop_front advances per block,
        // so it stays 0 too. The op writes dest slots [0, ct_dim).
        sub_tiles_bcast_cols_custom(icb0, icb1, 0, 0, 0, ct_dim);
        tile_regs_commit();

        tile_regs_wait();
        for (std::uint32_t i = 0; i < ct_dim; i++) {
            pack_tile(i, ocb);
        }
        tile_regs_release();

        dfb_out.push_back(ct_dim);
        dfb0.pop_front(ct_dim);
    }

    dfb1.pop_front(onetile);
}
