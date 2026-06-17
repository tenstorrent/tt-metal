// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// AXPY (y = a*x + y) compute kernel — chains two different FPU binary ops
// (mul_tiles followed by add_tiles).
//
// The point of this example is to show, in the simplest possible setting, two
// rules that trip up newcomers when they write their first multi-op kernel:
//
//   Rule 1: Call *_init every time the op or the input CBs change.
//           binary_op_init_common() is *not enough* — it only sets the initial
//           unpacker/math state. mul_tiles_init() and add_tiles_init() must be
//           called separately, and each time the op switches you must call the
//           new one again.
//
//   Rule 2: tile_regs_acquire() / tile_regs_release() bracket one math chain
//           plus its pack_tile(). You need a fresh acquire/release pair around
//           every pack — you cannot "save" the destination register across a
//           pack and reuse it.
//
// Simpler kernels like eltwise_binary call *_init once before the loop and
// take one acquire/release per iteration. That works because the op never
// changes — the same add_tiles fires every iteration. The moment you chain
// two *different* ops (here: mul_tiles -> add_tiles), the simple pattern
// breaks and you need the structure below.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_x = tt::CBIndex::c_0;     // input x[i]
    constexpr auto cb_y = tt::CBIndex::c_1;     // input y[i]
    constexpr auto cb_a = tt::CBIndex::c_2;     // scalar a (single broadcast tile, reused)
    constexpr auto cb_ax = tt::CBIndex::c_24;   // intermediate: ax[i] = a * x[i]
    constexpr auto cb_out = tt::CBIndex::c_16;  // output: y[i] += a * x[i]

    constexpr uint32_t dst_reg = 0;

    // The scalar 'a' tile is produced once by the reader. Wait for it once
    // and keep it pinned in the CB for the lifetime of the kernel — every
    // iteration reads the same tile slot.
    cb_wait_front(cb_a, 1);

    // Initial common config: tells the unpacker about the (cb_x, cb_a) input
    // pair and the packer about the cb_ax output. The format-dependent packer
    // setup happens here; subsequent pack_tile() calls to cb_out are fine as
    // long as cb_out shares the same data format as cb_ax.
    binary_op_init_common(cb_x, cb_a, cb_ax);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        // ===================== Phase 1: cb_ax[i] = a * x[i] =====================
        //
        // Why we need mul_tiles_init() here on every iteration:
        //   - On iteration 0: we just switched from "no math op" to MUL.
        //   - On iterations 1..N-1: the previous iteration ended in add_tiles,
        //     so the math engine is configured for ADD; we must switch it back.
        // Either way, the op (or its input CBs) changed, so init is required.
        mul_tiles_init(cb_x, cb_a);

        cb_wait_front(cb_x, 1);

        tile_regs_acquire();  // claim a fresh window of DST registers
        mul_tiles(cb_x, cb_a, /*itile0=*/0, /*itile1=*/0, /*idst=*/dst_reg);
        tile_regs_commit();  // hand DST off from MATH to PACK
        tile_regs_wait();    // wait until PACK can read DST

        cb_reserve_back(cb_ax, 1);
        pack_tile(dst_reg, cb_ax);
        cb_push_back(cb_ax, 1);

        tile_regs_release();  // release DST so the next acquire can claim it

        cb_pop_front(cb_x, 1);

        // ================== Phase 2: cb_out[i] = cb_ax[i] + y[i] =================
        //
        // Op switched (MUL -> ADD) and the input CB pair changed
        // (cb_x/cb_a -> cb_ax/cb_y). Both reasons independently require a new
        // *_init call — Rule 1.
        add_tiles_init(cb_ax, cb_y);

        cb_wait_front(cb_ax, 1);
        cb_wait_front(cb_y, 1);

        // A fresh acquire/release pair — Rule 2. We cannot reuse the previous
        // pair because the previous pack_tile() already consumed it.
        tile_regs_acquire();
        add_tiles(cb_ax, cb_y, /*itile0=*/0, /*itile1=*/0, /*idst=*/dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(dst_reg, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();

        cb_pop_front(cb_ax, 1);
        cb_pop_front(cb_y, 1);
    }

    cb_pop_front(cb_a, 1);
}
