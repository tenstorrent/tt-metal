// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// STAGE 1 of the Quasar narrow-row pack-untilize (test id 918): plain whole-tile HW
// pack-untilize into a TILE-ALIGNED output.
//
// This is deliberately the stock, unmodified `pack_untilize_block` -- the fast path
// (one hardware-streamed pack per tile) that Quasar cannot make narrow. Its L1 row stride is
// `full_ct_dim` TILES wide, because the packer's untilize stride is face-granular and the
// compute API only exposes it in whole tiles (api/compute/pack_untilize.h static_asserts
// `narrow_row == false` / `row_num_datums == TILE_C_DIM` / `dense == false` for ARCH_QUASAR).
//
// So the output this kernel leaves in `pad` is 32 rows of `ct_dim * 32` datums, of which only
// the leading `matrix_w` datums of each row are wanted; the rest is the padding the caller
// asked to drop. Stage 2 (kernels/narrow_row_compact_dm.cpp) squeezes that padding out with
// one iDMA transaction, which is the whole point of the test.
//
// LOOP_FACTOR: 1 for every correctness run, and that is not a tuning choice. The Quasar
// tile-counter model consumes a tile per unpack, so re-packing the same resident tiles
// without re-streaming them (which is what a loop >1 does here) produces valid TIMING but
// undefined DATA from the second iteration on. `loop_factor > 1` is therefore only for the
// timing-only variants, which skip the data check.
//
// Zone name is PACK_UNTILIZE (not the data-movement harness's "RISCV1") so the CSV
// reconstruction in report_narrow_row_from_csv.py can tell stage 1 from stage 2 -- both
// stages live in the same program and stamp into the same log.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Compile-time: pack_untilize_init/_block take it as a template parameter (it sets the L1
    // row stride). Declared inside kernel_main -- args:: is only reliably in scope once the
    // generated header has been pulled in ahead of the kernel body.
    constexpr std::uint32_t CT_DIM = get_arg(args::ct_dim);
    constexpr std::uint32_t OUT_ROWS = get_arg(args::out_rows);
    const std::uint32_t loop_factor = get_arg(args::loop_factor);

    DataflowBuffer dfb_in(dfb::src);
    DataflowBuffer dfb_out(dfb::pad);

    compute_kernel_hw_startup(dfb::src, dfb::pad);
    pack_untilize_init<CT_DIM>(dfb::src, dfb::pad);

    // Make the tile-row resident and reserve the output slot OUTSIDE the timed region, so the
    // zone holds pack work only and no DFB handshake.
    dfb_in.wait_front(CT_DIM);
    // `pad` entries are untilized ROWS, not tiles (see the host-side dfb spec), so the
    // reserve/push counts are in rows.
    dfb_out.reserve_back(OUT_ROWS);

    // A compute kernel is compiled ONCE PER TRISC THREAD, so an unguarded DeviceZoneScopedN
    // would emit an identically-named zone from unpack, math AND pack -- three zone pairs the
    // CSV reconstruction would merge into nonsense. Guard it to the pack thread, which is the
    // one doing the pack_untilize work.
    {
#ifdef UCK_CHLKC_PACK
        DeviceZoneScopedN("PACK_UNTILIZE");
#endif
        for (std::uint32_t i = 0; i < loop_factor; i++) {
            // block_rt_dim = 1: one tile-row per call.
            pack_untilize_block<CT_DIM>(dfb::src, 1, dfb::pad);
        }
    }

    dfb_out.push_back(OUT_ROWS);
    dfb_in.pop_front(CT_DIM);

#ifdef UCK_CHLKC_PACK
    // Same guard: one stamp set, from the same thread as the zone above, so the CSV order
    // reconstruction sees exactly one zone followed by its own stamps.
    DeviceTimestampedData("Test id", get_arg(args::test_id));
    DeviceTimestampedData("Number of transactions", loop_factor);
    DeviceTimestampedData("Tile columns", CT_DIM);
#endif
}
