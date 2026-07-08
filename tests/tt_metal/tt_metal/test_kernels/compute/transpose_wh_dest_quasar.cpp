// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Quasar transpose-in-DEST kernel driven by the operand-based unpack-to-dest API.
//
// Unlike transpose_wh_dest.cpp (which routes to DEST via the program-wide unpack_to_dest_en flag
// and pairs with the mode-carrying tile_regs_commit<>/release<>), this kernel expresses the
// unpack-to-dest intent purely at the call site: copy_tile_to_dst (+ its init) is the signal, and
// tile_regs_acquire/commit/release are all mode-agnostic. The trisc threads discover the regime
// from section-local state seeded by copy_tile_to_dst_init and set by copy_tile_to_dst.
void kernel_main() {
    constexpr uint32_t NHtWt = get_arg(args::NHtWt);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    unary_op_init_common(dfb::in, dfb::out);
    // Once before the loop: configure UNP_DEST routing, init the unpack->math handshake, and mark
    // the pack thread unpack-to-dest.
    copy_tile_to_dst_init(dfb::in);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - unpack each tile straight into DEST, then transpose_dest it in place
    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        copy_tile_to_dst(dfb::in, 0, 0);

        transpose_dest_init<DST_ACCUM_MODE, true /* transpose_of_faces */>(dfb::in);
        transpose_dest<DST_ACCUM_MODE, true /* transpose_of_faces */>(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
