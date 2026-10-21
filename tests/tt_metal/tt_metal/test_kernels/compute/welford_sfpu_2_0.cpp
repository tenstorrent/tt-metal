// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/welford.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Mean and variance of each column over the rows of per_core_block_cnt input tiles stacked top to bottom: Welford's
// online update, or with WELFORD_TWO_PASS the shifted two-pass kernels, which read the inputs twice. The inputs sit
// in DST from tile 2 on; the mean and variance land in row 0 of DST tiles 0 and 1, which are packed.
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::per_core_block_cnt);
    constexpr uint32_t mean_dst = 0;
    constexpr uint32_t input_dst = 2;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in_id = dfb_in.get_id();
    const uint32_t out_id = dfb_out.get_id();

    compute_kernel_hw_startup(in_id, out_id);
    copy_init(in_id);

    dfb_in.wait_front(num_tiles);
    dfb_out.reserve_back(2);
    tile_regs_acquire();
    for (uint32_t t = 0; t < num_tiles; ++t) {
        copy_tile(in_id, /*tile_index=*/t, /*dst_index=*/input_dst + t);
    }

    const uint32_t rows = num_tiles * 32;
#ifdef WELFORD_TWO_PASS
    // FP32 bits of 1 / rows, from the host.
    constexpr uint32_t reciprocal_bits = WELFORD_TWO_PASS_RECIPROCAL_BITS;
    two_pass_stats_init_shifted();
    two_pass_stats_clear();
    two_pass_stats_update_shifted_rows<TwoPassAccumulation::ShiftedSum, TwoPassAnchor::Initialise>(input_dst, 0, 32);
    for (uint32_t t = 1; t < num_tiles; ++t) {
        two_pass_stats_update_shifted_rows<TwoPassAccumulation::ShiftedSum>(input_dst + t, 0, 32);
    }
    two_pass_stats_finish_shifted_mean(reciprocal_bits);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        two_pass_stats_update_rows(input_dst + t, 0, 32);
    }
    two_pass_stats_finalize_to_row(mean_dst, reciprocal_bits);
#else
    // An empty reciprocal table: the kernel divides by the row count itself.
    constexpr std::array<uint32_t, 0> no_lut{};
    welford_init();
    for (uint32_t t = 0; t < num_tiles; ++t) {
        welford_update<0>(input_dst + t, /*start_idx=*/t * 32, no_lut);
    }
    welford_finalize_to_row<0>(mean_dst, /*scale_idx=*/rows - 1, no_lut);
#endif

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(mean_dst, out_id);
    pack_tile(mean_dst + 1, out_id);
    tile_regs_release();

    dfb_out.push_back(2);
    dfb_in.pop_front(num_tiles);
}
