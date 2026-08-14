// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dst_untilize with a VARIABLE block width per row: each output row is packed
// as (width/w_main - 1) blocks of w_main tiles followed by two half-width
// blocks of w_main/2 tiles. This is the real shape of untilize kernels whose
// row width forces mixed block widths (untilize_with_unpadding-style tails):
// every width switch forces a pack_untilize_dest_init re-run — two switches
// per row (w_main -> w_tail, then back at the next row).
//
// LLK 1.0 constraint shaping the pattern: _llk_pack_untilize_init_ requires
// full_ct_dim % block_ct_dim == 0, so both widths must divide the row width
// (w_main = largest even divisor <= DST capacity, w_tail = w_main/2).
//
// Two baseline flavors, selected by VW_SKIP_REMAP:
//   default (production shape) — every re-init uses the default
//     pack_untilize_dest_init, which also re-runs the BH DEST remap configure:
//     a MATH-thread tensix_sync + cfg RMWs per width switch. This is what
//     in-tree kernels that re-init pack_untilize inside loops actually do
//     (transpose_wh_rm, bmm fused-bias untilize-out, pool compute).
//   VW_SKIP_REMAP (hand-tuned) — BH DEST remap configured once, re-inits pass
//     configure_remap=false; the per-switch cost is the API floor — pack
//     reconfig + untilize init + dest-offset registers on the PACK thread.

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t largest_even_divisor_leq(uint32_t n, uint32_t cap) {
    for (uint32_t d = cap; d >= 2; --d) {
        if ((d % 2 == 0) && (n % d == 0)) {
            return d;
        }
    }
    return 0;
}

// Largest even d <= cap where d, d/2 and (d + d/2) all divide n — enables the
// alternating [d, d/2, d, d/2, ...] block pattern across a row.
constexpr uint32_t largest_even_pair_divisor_leq(uint32_t n, uint32_t cap) {
    for (uint32_t d = cap; d >= 2; --d) {
        if ((d % 2 == 0) && (n % d == 0) && (n % (d / 2) == 0) && (n % (d + d / 2) == 0)) {
            return d;
        }
    }
    return 0;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);            // rows
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);  // row width, in tiles
    DataflowBuffer dfb_in0(dfb::in);
    DataflowBuffer dfb_out0(dfb::out);

    constexpr uint32_t dst_cap = DST_ACCUM_MODE ? 4 : 8;
    // Alternating mode ([w_main, w_tail] pairs across the row — the shape of
    // sort/permute-style kernels that re-init pack around every block): pick
    // the largest even w_main where both widths and the pair divide the row.
    constexpr uint32_t w_alt = largest_even_pair_divisor_leq(per_core_block_tile_cnt, dst_cap);
    constexpr bool alternating = (w_alt >= 2);
    constexpr uint32_t w_main = alternating ? w_alt : largest_even_divisor_leq(per_core_block_tile_cnt, dst_cap);
    static_assert(w_main >= 2, "row width needs an even divisor <= DST capacity");
    constexpr uint32_t w_tail = w_main / 2;
    // Alternating: num_pairs x [w_main, w_tail]. Otherwise: (width/w_main - 1)
    // main blocks plus two tail blocks.
    constexpr uint32_t num_pairs = alternating ? per_core_block_tile_cnt / (w_main + w_tail) : 1;
    constexpr uint32_t num_main = alternating ? 1 : per_core_block_tile_cnt / w_main - 1;
    constexpr uint32_t num_tail = alternating ? 1 : 2;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;
    static_assert(num_main > 0, "variable-width benchmark wants at least one full block plus the tail pair");

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_tile_to_dst_init_short(dfb::in);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        dfb_out0.reserve_back(full_ct_dim);

        // Main blocks (width w_main). Re-init needed on every row after the
        // first: the previous row ended at the tail width.
#ifdef VW_SKIP_REMAP
        if (r == 0) {
            pack_untilize_dest_init<w_main, full_ct_dim>(dfb::out);
        } else {
            pack_untilize_dest_init<w_main, full_ct_dim, false, TILE_C_DIM, false, false /*configure_remap*/>(
                dfb::out);
        }
#else
        pack_untilize_dest_init<w_main, full_ct_dim>(dfb::out);
#endif
        for (uint32_t b = 0; b < num_main; ++b) {
            dfb_in0.wait_front(w_main);
            tile_regs_acquire();
            for (uint32_t i = 0; i < w_main; ++i) {
                copy_tile(dfb::in, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<w_main, full_ct_dim>(dfb::out, 1, b);
            tile_regs_release();
            dfb_in0.pop_front(w_main);
        }

        // Tail blocks (width w_tail = w_main/2): width switch -> re-init.
#ifdef VW_SKIP_REMAP
        pack_untilize_dest_init<w_tail, full_ct_dim, false, TILE_C_DIM, false, false /*configure_remap*/>(dfb::out);
#else
        pack_untilize_dest_init<w_tail, full_ct_dim>(dfb::out);
#endif
        for (uint32_t t = 0; t < num_tail; ++t) {
            dfb_in0.wait_front(w_tail);
            tile_regs_acquire();
            for (uint32_t i = 0; i < w_tail; ++i) {
                copy_tile(dfb::in, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<w_tail, full_ct_dim>(dfb::out, 1, (num_main * w_main) / w_tail + t);
            tile_regs_release();
            dfb_in0.pop_front(w_tail);
        }

        dfb_out0.push_back(full_ct_dim);
    }

    pack_untilize_uninit(dfb::out);
}
