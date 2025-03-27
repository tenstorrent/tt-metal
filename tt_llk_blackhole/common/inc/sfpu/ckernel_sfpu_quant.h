// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _quant_int32_(const uint dst_offset)
{
// Operand A is input (fp32)
// Operand B is scaling factor (fp32)
// Operand C is zero-point constant (fp32)
// Output is int32 scaled to int8 range
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A - fp32
        TTI_SFPLOAD(0, 3, ADDR_MOD_7, 0);
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, ADDR_MOD_7, dst_offset * 64);
        // D(A) = A*B+C, LREG[2] = zero_point
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // fp32->int8, descale value is zero (LREG_9)
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 3);
        // LREG_0 -> dest as int32
        if constexpr (SIGN_MAGNITUDE_FORMAT == false)
        {
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        TTI_SFPSTORE(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _requant_int32_(const uint dst_offset)
{
// Operand A is input to requant (int32)
// Operand B is scaling factor (fp32)
// Operand C is zero-point constant (fp32)
// Output is int32 scaled to int8 range
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A - int32
        TTI_SFPLOAD(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0);
        if constexpr (SIGN_MAGNITUDE_FORMAT == false)
        {
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, ADDR_MOD_7, dst_offset * 64);
        // cast int32->fp32
        TTI_SFPCAST(0, 0, 0);
        // D(A) = A*B+C, LREG[2] = zero_point
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // fp32->int8, descale value is zero (LREG_9)
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 3);
        // LREG_0 -> dest as int32
        if constexpr (SIGN_MAGNITUDE_FORMAT == false)
        {
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        TTI_SFPSTORE(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _dequant_int32_(const uint dst_offset)
{
// Operand A[LREG0] is input to dequant (int32)
// Operand B[LREG1] is scaling factor (fp32)
// Operand C[LREG2] is zero-point constant (fp32)
// Output = (A + (-C)) * B (fp32)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A - int32
        TTI_SFPLOAD(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0);
        if constexpr (SIGN_MAGNITUDE_FORMAT == false)
        {
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            // Required after cast due to a bug in Blackhole RTL.
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, ADDR_MOD_7, dst_offset * 64);
        // cast int32->fp32
        TTI_SFPCAST(0, 0, 0);
        // D(A)) = A+(-C), LREG[10] is 1, SFPADD = LREG_A*LREG_B+LREG_C
        TTI_SFPADD(0, 10, 2, 0, 0);
        TTI_NOP;
        // D(A)) = (A+(-C))*B, LREG[9] is zero
        TTI_SFPMUL(0, 1, 9, 0, 0);
        TTI_NOP;
        // LREG_0 -> dest as fp32
        TTI_SFPSTORE(0, 3, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_quant_zero_point_(const uint zero_point)
{
    _sfpu_load_imm32_(2, zero_point);
}

} // namespace sfpu
} // namespace ckernel
