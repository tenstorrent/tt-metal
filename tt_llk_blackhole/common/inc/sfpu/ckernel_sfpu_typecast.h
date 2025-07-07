// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_uint16_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPLOADI(0, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 2, 0, 1, 14);
        TTI_SFPSTORE(1, 6, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 6, ADDR_MOD_7, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 3, 1, 2, 1);
        TTI_SFPSTORE(2, 2, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // Modifies LOAD/STORE to work with INT32 2's complement, however
    // in Blackhole this has no effect and format remains INT32 sign-magnitude.
    constexpr auto INSTR_MOD_LOAD_STORE = InstrModLoadStore::INT32_2S_COMP;

    // LOAD/STORE have the value in INT sign magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0);
        TTI_SFPCAST(0, 2, INSTR_MOD_CAST);
        // Required after cast due to a bug in Blackhole RTL.
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFP_STOCH_RND(0, 0, 3, 1, 2, 1);
        TTI_SFPSTORE(2, 2, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_int32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        // extract exponent
        sfpi::vInt exp = exexp(in);

        v_if (exp < 0)
        {
            sfpi::dst_reg[0] = 0;
        }
        v_elseif (exp > 30)
        {
            // set to int32 max value in case of overflow
            sfpi::vInt tmp = std::numeric_limits<int32_t>::max();
            // check sign
            v_if (in < 0)
            {
                // 2's complement conversion
                tmp = (~tmp) + 1;
            }
            v_endif sfpi::dst_reg[0] = tmp;
        }
        v_else
        {
            // extract mantissa
            sfpi::vInt man = exman8(in);
            // shift the mantissa by (23-exponent) to the right
            sfpi::vInt shift = exp - 23;
            man              = sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);
            // check sign
            v_if (in < 0)
            {
                // 2's complement conversion
                man = (~man) + 1;
            }
            v_endif sfpi::dst_reg[0] = man;
        }
        v_endif

            sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
        TTI_SFP_STOCH_RND(0, 0, 2, 0, 1, 1);
        TTI_SFPSTORE(1, 0, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 6, ADDR_MOD_7, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFPSTORE(1, 3, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    // Modifies LOAD/STORE to work with INT32 2's complement, however
    // in Blackhole this has no effect and format remains INT32 sign-magnitude.
    constexpr auto INSTR_MOD_LOAD_STORE = InstrModLoadStore::INT32_2S_COMP;

    // LOAD/STORE have the value in INT sign magnitude format and SFPU needs it as 2's complement.
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, INSTR_MOD_LOAD_STORE, ADDR_MOD_7, 0);
        TTI_SFPCAST(0, 2, INSTR_MOD_CAST);
        // Required after cast due to a bug in Blackhole RTL.
        TTI_SFPSETSGN(0, 2, 0, 0);
        TTI_SFPCAST(0, 1, 0);
        TTI_SFPSTORE(1, 3, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp16b_to_uint32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        // check sign
        v_if (in <= 0)
        {
            sfpi::dst_reg[0] = 0;
        }
        v_else
        {
            // extract exponent
            sfpi::vInt exp = exexp(in);

            v_if (exp < 0)
            {
                sfpi::dst_reg[0] = 0;
            }
            v_elseif (exp > 31)
            {
                // set to uint32 max value in case of overflow
                sfpi::vInt tmp   = std::numeric_limits<int32_t>::max();
                sfpi::dst_reg[0] = sfpi::setsgn(sfpi::reinterpret<sfpi::vFloat>(tmp), 1);
            }
            v_elseif (exp == 31)
            {
                // extract mantissa without hidden bit
                sfpi::vInt man = exman9(in);
                // shift the mantissa by (23-exponent) to the right
                sfpi::vInt shift = exp - 23;
                man              = sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);
                // add hidden bit back (due to bug when shifting a 1 into MSB)
                sfpi::dst_reg[0] = sfpi::setsgn(sfpi::reinterpret<sfpi::vFloat>(man), 1);
            }
            v_else
            {
                // extract mantissa
                sfpi::vInt man = exman8(in);
                // shift the mantissa by (23-exponent) to the right
                sfpi::vInt shift = exp - 23;
                man              = sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), shift);
                sfpi::dst_reg[0] = man;
            }
            v_endif
        }
        v_endif

            sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 12, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPCAST(1, 2, 0);
        TTI_SFP_STOCH_RND(0, 0, 4, 2, 3, 1);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPADDI(0x4f00, 3, 0); // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(3, 2, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(0, 12, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, 0, 1, 1);
        TTI_SFPCAST(1, 2, 0);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFPADDI(0x4f00, 2, 0); // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(2, 3, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    // Packer will read HI16 bits from DEST if DEST is in 32bit mode but pck is configured for uint16
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0); // Load 32bit datums from DEST
        TTI_SFPLOADI(p_sfpu::LREG1, 2, 0xFFFF);                                      // Load Lo16 of LREG1 with 0xFFFF (65535)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 4);                                        // If sign bit of LREG0 is 0, set CC
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                             // Put smaller of LREG0 & 1 into LREG1
        TTI_SFPENCC(0, 0, 0, 0);                                                     // Clear CC bits
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_7, 0);         // Store output
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, 0); // Load 32bit datums from DEST
        TTI_SFPLOADI(p_sfpu::LREG1, 2, 0);                                           // Load Lo16 of LREG1 with 0
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                             // Put smaller of LREG0 & 1 into LREG1
        TTI_SFPLOADI(p_sfpu::LREG2, 2, 0xFFFF);                                      // Load Lo16 of LREG2 with 0xFFFF (65535)
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, 1);                             // Put smaller of LREG0 & 2 into LREG0
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);         // Store output
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
