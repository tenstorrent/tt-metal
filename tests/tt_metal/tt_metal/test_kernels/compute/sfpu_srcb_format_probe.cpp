// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experiment kernel for tt-llk#951: confirm that the SFPU interprets values it
// loads from DEST using the *SrcB* (ALU_FORMAT_SPEC_REG1_SrcB) data format.
//
// All CBs are Float16_b (E8M7) so the tt-metal exponent-family guardrail
// (check_consistent_format_across_buffers) is satisfied and the kernel compiles.
// We then *directly* override ALU_FORMAT_SPEC_REG1_SrcB to the format value
// given by compile arg 0, leaving SrcA = Float16_b. copy_tile (FPU datacopy)
// writes DEST from SrcA (Float16_b); square_tile's SFPLOAD/SFPSTORE reads/writes
// DEST using SrcB. If the output depends on the SrcB override, REG1_SrcB governs
// the SFPU's interpretation of DEST.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t srcb_format_override = get_compile_time_arg_val(0);

    // REG0_SrcA = REG1_SrcB = Float16_b (from c_0).
    unary_op_init_common(in0_cb, out_cb);

    // Force ONLY REG1_SrcB to the requested format value, leaving SrcA = Float16_b.
    MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE, false>(srcb_format_override)));

    square_tile_init();

    CircularBuffer cb0(in0_cb);
    CircularBuffer cb16(out_cb);

    tile_regs_acquire();
    tile_regs_wait();

    cb0.wait_front(1);
    cb16.reserve_back(1);

    copy_tile(in0_cb, 0, 0);  // SrcA (Float16_b) -> DEST[0]
    square_tile(0);           // SFPU load/store DEST via REG1_SrcB (override)
    pack_tile(0, out_cb);

    cb0.pop_front(1);
    cb16.push_back(1);

    tile_regs_commit();
    tile_regs_release();
}
