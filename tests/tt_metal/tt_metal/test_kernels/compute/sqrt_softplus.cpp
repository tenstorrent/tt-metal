// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the unary sqrt_softplus SFPU op one tile at a time so a host test can compare against the
// torch reference. x arrives in c_0, the result is packed to c_16.
//   compile_time_args = [num_tiles, beta_bits, beta_reciprocal_bits, threshold_bits]

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
    constexpr uint32_t beta = get_compile_time_arg_val(1);
    constexpr uint32_t beta_reciprocal = get_compile_time_arg_val(2);
    constexpr uint32_t threshold = get_compile_time_arg_val(3);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    DataflowBuffer x_cb(cb_x);
    DataflowBuffer out_cb(cb_out);

    compute_kernel_hw_startup(cb_x, cb_out);
    copy_init(cb_x);
    // Stands in for an earlier op that owned the programmable constants (the stock softplus loads
    // Prgm1/Prgm2). The init must load all three the sqrt reads; one left stale fails the ULP bar.
    MATH((sfpi::vConstFloatPrgm0 = 0.125f));
    MATH((sfpi::vConstFloatPrgm1 = 0.125f));
    MATH((sfpi::vConstFloatPrgm2 = 0.125f));
    sqrt_softplus_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        x_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_x, 0, 0);
        sqrt_softplus_tile(0, beta, beta_reciprocal, threshold);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        x_cb.pop_front(1);
    }
}
