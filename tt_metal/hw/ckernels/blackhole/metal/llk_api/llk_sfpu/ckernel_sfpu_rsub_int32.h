// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include <type_traits>

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_rsub_int(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");
    constexpr int sfpload_instr_mod = static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);

    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - int32/uint32/uint16
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);

        // operand B - int32/uint32/uint16 (offset by dst_offset * dest tile size)
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // Reverse subtraction is performed using 2's complement by adding B to the negation of A: LREG1 + (-LREG0)
        // Uses 6 as imod. Performs integer addition between LREG specified in lreg_c and the 2's complement (4) of LREG
        // specified in lreg_dest. The condition code register is not modified (2).
        TTI_SFPIADD(
            0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);

        // Store result from LREG_0 to dest
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_rsub_scalar_int32(uint32_t scalar) {
    int int_scalar = scalar;
    // Load scalar value param to lreg2
    _sfpu_load_imm32_(p_sfpu::LREG1, int_scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        // Uses 6 as imod. Performs integer addition between LREG specified in lreg_c and the 2's complement (4) of LREG
        // specified in lreg_dest. The condition code register is not modified (2).
        TTI_SFPIADD(
            0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
