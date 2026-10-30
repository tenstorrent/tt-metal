// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the fused moe_gate_sqrt_softplus_add_bias SFPU op over a tile pair at a time so a host
// test can compare against the torch reference. x arrives in c_0, bias in c_1; one of the two
// results (the score over x, or the key over bias) is packed to c_16.
//   compile_time_args = [num_tiles, dst_x_index, dst_bias_index, pack_key]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/moe_gate_sqrt_softplus.h"
#include "api/dataflow/dataflow_buffer.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#endif

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    // Both dst indices are host-supplied so the op is held to them. The gate calls it as (0, 1).
    constexpr uint32_t kDstX = get_compile_time_arg_val(1);
    constexpr uint32_t kDstBias = get_compile_time_arg_val(2);
    // The op writes two results in place; the host picks which one comes back. 0: the score
    // sqrt(softplus(x)) over kDstX, 1: the key score + bias over kDstBias.
    constexpr bool kPackKey = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t kDstOut = kPackKey ? kDstBias : kDstX;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_bias = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    DataflowBuffer x_cb(cb_x);
    DataflowBuffer bias_cb(cb_bias);
    DataflowBuffer out_cb(cb_out);

    // Both input CBs must share a data format: compute_kernel_hw_startup / copy_init configure the
    // unpacker from cb_x, and copy_tile only updates the L1 address per call, not the THCON
    // format registers.
    compute_kernel_hw_startup(cb_x, cb_out);
    copy_init(cb_x);
    // Stands in for an earlier op that owned the programmable constants. The init must load all
    // three the sqrt reads (Prgm0 as the inverse-sqrt seed, Prgm1 and Prgm2 as refinement
    // coefficients); one left stale fails the ULP bar.
    MATH((sfpi::vConstFloatPrgm0 = 0.125f));
    MATH((sfpi::vConstFloatPrgm1 = 0.125f));
    MATH((sfpi::vConstFloatPrgm2 = 0.125f));
    moe_gate_sqrt_softplus_add_bias_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        x_cb.wait_front(1);
        bias_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_x, 0, kDstX);
        copy_tile(cb_bias, 0, kDstBias);
        moe_gate_sqrt_softplus_add_bias_tile(kDstX, kDstBias);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(kDstOut, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        x_cb.pop_front(1);
        bias_cb.pop_front(1);
    }
}
