// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// STAGE 1 of the Quasar narrow-row pack-untilize (test id 918): plain whole-tile HW
// pack-untilize into a TILE-ALIGNED output.
//
// This is deliberately the stock, unmodified `pack_untilize_block` -- the fast hardware path
// that Quasar cannot make narrow. Its L1 row stride is `full_ct_dim` TILES wide, because the
// packer's untilize stride is face-granular and the compute API exposes it only in whole
// tiles (api/compute/pack_untilize.h static_asserts narrow_row/row_num_datums/dense away for
// ARCH_QUASAR).
//
// So the output left in `pad` is `out_rows` rows of `ct_dim * 32` datums, of which only the
// leading `matrix_w` are wanted. Stage 2 gathers those out into a dense buffer, which is the
// point of the test.

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

    DataflowBuffer dfb_in(dfb::src);
    DataflowBuffer dfb_out(dfb::pad);

    compute_kernel_hw_startup(dfb::src, dfb::pad);
    pack_untilize_init<CT_DIM>(dfb::src, dfb::pad);

    dfb_in.wait_front(CT_DIM);
    // `pad` entries are untilized ROWS, not tiles (see the host-side dfb spec), so the
    // reserve/push counts are in rows.
    dfb_out.reserve_back(OUT_ROWS);

    // block_rt_dim = 1: one tile-row. Called exactly once -- the Quasar tile-counter model
    // consumes a tile per unpack, so a second call without re-streaming the input would
    // untilize tiles that were never delivered.
    pack_untilize_block<CT_DIM>(dfb::src, 1, dfb::pad);

    dfb_out.push_back(OUT_ROWS);
    dfb_in.pop_front(CT_DIM);
}
