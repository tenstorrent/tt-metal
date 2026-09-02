// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the fused clamped_silu_glu binary SFPU op over a tile pair at a time so a host
// test can compare against the torch reference. gate arrives in c_0, up in c_1;
// the result is packed to c_16.
//   compile_time_args = [num_tiles, dst_gate_index, dst_up_index, dst_out_index, skip_init]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/clamped_silu_glu.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#endif

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    // All three dst indices are host-supplied so the op is held to them. No production caller
    // exists yet; the shape mirrors how fused_swiglu.cpp drives its sibling activations --
    // BINARY_ACT_TILE(fp32, j, c + j, j) with a runtime chunk width c, so both the gate index
    // and the gate->up stride vary.
    constexpr uint32_t kDstGate = get_compile_time_arg_val(1);
    constexpr uint32_t kDstUp = get_compile_time_arg_val(2);
    // Output index: aliasing the gate operand is what the expert kernel does to save a dst slot;
    // a separate slot catches an implementation that ignores out_tile_idx.
    constexpr uint32_t kDstOut = get_compile_time_arg_val(3);
    // Skips clamped_silu_glu_tile_init() so a host case can show the init is load-bearing against
    // the polluted Prgm0 below, instead of that only being asserted by a comment.
    constexpr bool kSkipInit = get_compile_time_arg_val(4) != 0;

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer out_cb(cb_out);

    // Both input CBs must share a data format: compute_kernel_hw_startup / copy_init configure the
    // unpacker from cb_gate, and copy_tile only updates the L1 address per call, not the THCON
    // format registers.
    compute_kernel_hw_startup(cb_gate, cb_out);
    copy_init(cb_gate);
    // Leave Prgm0 in a state clamped_silu_glu_tile_init() has to repair, standing in for an earlier
    // op in the same kernel that owned it -- the condition a fused activation runs in, and what
    // holds that init under test. Any value but 2.0 does it: Prgm0 is read only as `x*y - Prgm0`
    // to gate a Newton step, and x*y is ~1, so anything below ~1 skips the refinement.
    MATH((sfpi::vConstFloatPrgm0 = 0.125f));
    if constexpr (!kSkipInit) {
        clamped_silu_glu_tile_init();
    }

    for (uint32_t t = 0; t < num_tiles; ++t) {
        gate_cb.wait_front(1);
        up_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, kDstGate);
        copy_tile(cb_up, 0, kDstUp);
        clamped_silu_glu_tile(kDstGate, kDstUp, kDstOut);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(kDstOut, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        gate_cb.pop_front(1);
        up_cb.pop_front(1);
    }
}
