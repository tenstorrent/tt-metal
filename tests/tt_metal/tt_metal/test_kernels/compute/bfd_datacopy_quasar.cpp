// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC kernel. Rotates over three input DFBs and calls
// unary_op_init_common before every tile copy, so each iteration bump-allocates a
// fresh buffer descriptor id from the per-TRISC partition (unpack: [0,16), pack:
// [16,24)) and programs its table entry. With num_cycles = 18 the unpack partition
// wraps once and the pack partition wraps twice, exercising id reuse across inits.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t num_cycles = get_arg(args::num_cycles);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out(dfb::out);

    for (std::uint32_t i = 0; i < num_cycles; ++i) {
        const std::uint32_t in_sel = i % 3;
        const std::uint32_t in_id = in_sel == 0 ? dfb::in0 : (in_sel == 1 ? dfb::in1 : dfb::in2);

        // Re-init every cycle: allocates fresh BFD ids on T0/T2 and reprograms the
        // table entries + MOPs for the newly selected input and the output.
        unary_op_init_common(in_id, dfb::out);

        tile_regs_acquire();
        tile_regs_wait();

        if (in_sel == 0) {
            dfb_in0.wait_front(1);
            dfb_out.reserve_back(1);
            copy_tile(dfb::in0, 0, 0);
            pack_tile(0, dfb::out);
            dfb_in0.pop_front(1);
        } else if (in_sel == 1) {
            dfb_in1.wait_front(1);
            dfb_out.reserve_back(1);
            copy_tile(dfb::in1, 0, 0);
            pack_tile(0, dfb::out);
            dfb_in1.pop_front(1);
        } else {
            dfb_in2.wait_front(1);
            dfb_out.reserve_back(1);
            copy_tile(dfb::in2, 0, 0);
            pack_tile(0, dfb::out);
            dfb_in2.pop_front(1);
        }
        dfb_out.push_back(1);

        tile_regs_commit();
        tile_regs_release();
    }
}
