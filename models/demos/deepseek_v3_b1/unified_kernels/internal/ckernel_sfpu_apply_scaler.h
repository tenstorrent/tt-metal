// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// !!! INTERNAL — deepseek_v3_b1 only !!!
// This op is hand-tuned for the deepseek_v3_b1 dkv-matmul lane layout (face
// quadrant offsets 0/2/16/18, scaler in LReg1 after a Stage-2 SFPTRANSP).
// It is NOT a general-purpose SFPU activation. Do not include this header
// from outside models/demos/deepseek_v3_b1/. Use the standard sigmoid/silu
// wrappers under tt_metal/hw/ckernels/.../llk_sfpu/ instead.

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// ----------------------------------------------------------------------------
// apply_scaler (deepseek_v3_b1 internal)
//
// Multiplies dst_reg[0] by a scaler value pre-loaded into SFPU LReg0.
//
// The caller is expected to populate LReg0 before this op runs (e.g. via
// TT_SFPLOADI from L1 in the matmul kernel's CUSTOM_SFPU init block). The
// value in LReg0 is reinterpreted as fp32, so the caller is responsible for
// choosing the SFPLOADI mod that produces the desired bit pattern. For a
// bf16 source the canonical mode is SFPLOADI_MOD0_FLOATB (0), which lands
// the 16 source bits in LReg[31:16] and zeros LReg[15:0] — i.e. the fp32
// representation of the bf16 value.
// ----------------------------------------------------------------------------
// No template params: the op operates on a fixed lane layout (face quadrants
// 0/2/16/18) and a fixed scaler register (LReg0), so there are no compile-time
// knobs to expose.
inline void calculate_apply_scaler() {
    // Stage 1: pull 4 face quadrants of the dst tile into LReg4..7.
    //   offset 0  → Face 0/2, even columns
    //   offset 2  → Face 0/2, odd  columns
    //   offset 16 → Face 1/3, even columns
    //   offset 18 → Face 1/3, odd  columns
    TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_7, 16);
    TTI_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_7, 18);

    // Stage 2: transpose. SFPTRANSP shuffles {LReg0..3} and {LReg4..7} as two
    // independent 4-lane groups, putting the loaded face columns into a layout
    // where successive LRegs hold successive rows.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Stage 3: LReg4 *= LReg1, result in LReg4.
    // SFPMUL(srcA, srcB, srcC, dst, mod) — srcC = LCONST_0 disables the fused
    // add, so this is a pure multiply.
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);

    // Stage 4: inverse transpose to undo Stage 2's lane shuffle, putting the
    // multiplied values back into the face/column layout the dst tile expects.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Stage 5: store LReg4..7 back to the dst tile at the same face quadrants
    // they were loaded from in Stage 1 (mirrors the load offsets 0/2/16/18).
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 2);
    TTI_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_7, 16);
    TTI_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_7, 18);
}

template <bool APPROXIMATION_MODE>
inline void apply_scaler_init() {
    // No-op. The scaler value is loaded into LReg0 by the caller's SFPLOADI
    // sequence before this op runs; no SFPU state needs to be programmed here.
}

}  // namespace ckernel::sfpu
