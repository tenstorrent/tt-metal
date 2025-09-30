// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// C++17 compatible bit_cast replacement using union
template <typename To, typename From>
inline To _bit_cast_(const From& from) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "Types must have same size");
    static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");

    union
    {
        From f;
        To t;
    } u;

    u.f = from;
    return u.t;
}

// Optimized float to 16-bit parts conversion
struct FloatBits
{
    uint16_t high16;
    uint16_t low16;

    explicit FloatBits(float value)
    {
        const uint32_t bits = _bit_cast_<uint32_t>(value);
        high16              = static_cast<uint16_t>(bits >> 16);
        low16               = static_cast<uint16_t>(bits & 0xFFFF);
    }
};

template <std::size_t reciprocal_size>
sfpi_inline void _load_recip_current_sample_(const uint32_t current_sample, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    if constexpr (reciprocal_size > 0)
    {
        // Use LUT if current_sample is within bounds, otherwise fall back to float division
        if (current_sample < reciprocal_lut.size())
        {
            const auto reciprocal = reciprocal_lut[current_sample];
            TT_SFPLOADI(ckernel::p_sfpu::LREG7, 8, reciprocal >> 16);
            TT_SFPLOADI(ckernel::p_sfpu::LREG7, 10, reciprocal & 0xFFFF);
            return;
        }
    }

    // Fallback to float division
    const float reciprocal = 1.0f / static_cast<float>(current_sample + 1);
    const FloatBits reciprocal_bits(reciprocal);
    TT_SFPLOADI(ckernel::p_sfpu::LREG7, 8, reciprocal_bits.high16);
    TT_SFPLOADI(ckernel::p_sfpu::LREG7, 10, reciprocal_bits.low16);
}

sfpi_inline void _compute_welfords_math_()
{
    //=========================================
    // mean calculation start
    //=========================================
    // mean_{N_+1} = mean_{N} + ((1/N+1) * (x_{N+1} - mean_{N}))

    /*mean_{N+1}temp = 1 * (InputLREG + (-mean))*/
    TTI_SFPMAD(ckernel::p_sfpu::LREG11, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, 0);
    // Next cycle cannot read from LREG6 See tt-isa-documentation

    TTI_SFPNOP;

    /*mean_{N+1} = ((mean_{N+1} = (InputLREG-mean) * (1/N+1)) + mean_{N}*/
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG6, 0);
    // Next cycle cannot read from LREG6 See tt-isa-documentation

    //=========================================
    // mean calculation end
    //=========================================
    //
    //=========================================
    // var calculation start
    //=========================================

    // var_{N+1} = var_{N} + ...
    //...(1/N+1) * (((x_{N+1} - mean_{N}) * (x_{N+1} - mean_{N+1})) - var_{N})

    /*mean_{N} = (Input_LREG - mean_{N})*/
    TTI_SFPMAD(ckernel::p_sfpu::LREG11, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 0);
    // Next cycle cannot read from LREG4 See tt-isa-documentation

    /*inputLREG temp = (InputLREG + (-mean_{N+1}))*/
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*LREG11 = <-1>*/, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    // Next cycle cannot read from InputLREG See tt-isa-documentation

    TTI_SFPNOP;

    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);

    // Moves mean to LREG4 from LREG6 since it now is considered the past mean
    TTI_SFPMUL(ckernel::p_sfpu::LCONST_1 /*LREG11 = <-1>*/, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4, 0);
    // Next cycle cannot read from LREG4 See tt-isa-documentation
    TTI_SFPNOP;

    //=========================================
    // var calculation end
    //=========================================
    // Now past_mean (LREG4) is population
    // Now past_var (LREG5) is population
}

template <uint32_t I, uint32_t J>
sfpi_inline void welfords_load_data()
{
    constexpr uint32_t offset1 = (I * 32) + (4 * J);
    constexpr uint32_t offset2 = offset1 + 2;
    constexpr uint32_t offset3 = offset1 + 16;
    constexpr uint32_t offset4 = offset1 + 18;
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, ckernel::ADDR_MOD_3, offset1); /*row1*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, ckernel::ADDR_MOD_3, offset2); /*row2*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, ckernel::ADDR_MOD_3, offset3); /*row3*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, ckernel::ADDR_MOD_3, offset4); /*row4*/
    /*transposes raw mixed data to logical rows*/
    lltt::replay(18, 5);
}

sfpi_inline void _welfords_load_initial_data_()
{
    constexpr uint32_t offset1 = 0;
    constexpr uint32_t offset2 = offset1 + 2;
    constexpr uint32_t offset3 = offset1 + 16;
    constexpr uint32_t offset4 = offset1 + 18;
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, ckernel::ADDR_MOD_3, offset1); /*row1*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, ckernel::ADDR_MOD_3, offset2); /*row2*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, 0, ckernel::ADDR_MOD_3, offset3); /*row3*/
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, ckernel::ADDR_MOD_3, offset4); /*row4*/
    /*transposes raw mixed data to logical rows*/
    TTI_SFPTRANSP(0, 0, 0, 0);
    // Needed since LREGS can maintain state between calls/maybe kernels? So setting them to zero is needed
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, 0, 0);
    // wiping LREG 6 and 7 since they may be filled with garbage data
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, 0, 0);
}

// Macro to allow returns to exit main function

#define WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)                                                    \
    if (current_sample == final_sample)                                                                                                       \
    {                                                                                                                                         \
        return current_sample;                                                                                                                \
    }                                                                                                                                         \
    if (skip_n_rows == 0)                                                                                                                     \
    {                                                                                                                                         \
        _load_recip_current_sample_(current_sample, reciprocal_lut);                                                                          \
        lltt::replay(0, 9);                                                                                                                   \
        current_sample++;                                                                                                                     \
    }                                                                                                                                         \
    else                                                                                                                                      \
    {                                                                                                                                         \
        skip_n_rows--;                                                                                                                        \
    }                                                                                                                                         \
    if (current_sample == final_sample)                                                                                                       \
    {                                                                                                                                         \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);                                                                     \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);                                                                    \
        return current_sample;                                                                                                                \
    }                                                                                                                                         \
    if (skip_n_rows == 0)                                                                                                                     \
    {                                                                                                                                         \
        TTI_SFPADD(ckernel::p_sfpu::LCONST_1 /*LREG10 = <1>*/, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0); \
        _load_recip_current_sample_(current_sample, reciprocal_lut);                                                                          \
        lltt::replay(0, 9);                                                                                                                   \
        current_sample++;                                                                                                                     \
    }                                                                                                                                         \
    else                                                                                                                                      \
    {                                                                                                                                         \
        skip_n_rows--;                                                                                                                        \
    }                                                                                                                                         \
    if (current_sample == final_sample)                                                                                                       \
    {                                                                                                                                         \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);                                                                     \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);                                                                    \
        return current_sample;                                                                                                                \
    }                                                                                                                                         \
    if (skip_n_rows == 0)                                                                                                                     \
    {                                                                                                                                         \
        TTI_SFPADD(ckernel::p_sfpu::LCONST_1 /*LREG10 = <1>*/, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0); \
        _load_recip_current_sample_(current_sample, reciprocal_lut);                                                                          \
        lltt::replay(0, 9);                                                                                                                   \
        current_sample++;                                                                                                                     \
    }                                                                                                                                         \
    else                                                                                                                                      \
    {                                                                                                                                         \
        skip_n_rows--;                                                                                                                        \
    }                                                                                                                                         \
    if (current_sample == final_sample)                                                                                                       \
    {                                                                                                                                         \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);                                                                     \
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);                                                                    \
        return current_sample;                                                                                                                \
    }                                                                                                                                         \
    if (skip_n_rows <= 0)                                                                                                                     \
    {                                                                                                                                         \
        TTI_SFPADD(ckernel::p_sfpu::LCONST_1 /*LREG10 = <1>*/, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, 0); \
        _load_recip_current_sample_(current_sample, reciprocal_lut);                                                                          \
        lltt::replay(0, 9);                                                                                                                   \
        current_sample++;                                                                                                                     \
    }                                                                                                                                         \
    else                                                                                                                                      \
    {                                                                                                                                         \
        skip_n_rows--;                                                                                                                        \
    }                                                                                                                                         \
    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);                                                                         \
    TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);                                                                        \
    if (current_sample == final_sample)                                                                                                       \
    {                                                                                                                                         \
        return current_sample;                                                                                                                \
    }

template <std::size_t reciprocal_size>
sfpi_inline uint32_t
_welfords_main_(uint32_t current_sample, const uint32_t final_sample, uint32_t skip_n_rows, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    // I, J, LOAD_PREVIOUS, N, endN. N can only be zero in first iteration
    if (current_sample == 0)
    {
        lltt::replay(9, 9);
        WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    }
    else
    {
        welfords_load_data<0, 0>();
        WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    }
    welfords_load_data<0, 1>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<0, 2>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<0, 3>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<1, 0>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<1, 1>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<1, 2>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    welfords_load_data<1, 3>();
    WELFORDS_LOOP_ITERATION(current_sample, final_sample, skip_n_rows, reciprocal_lut)
    return current_sample;
}

#undef WELFORDS_LOOP_ITERATION

sfpi_inline void _save_data_(const bool reformat_dst)
{
    if (reformat_dst)
    {
        // This subroutine allows us to save the row of mean vals to the dstreg 1 and row of variance vals to dstreg 2
        TTI_SFPADD(ckernel::p_sfpu::LCONST_1 /*LREG10 = <1>*/, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, 0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0);

        TTI_SFPMUL(ckernel::p_sfpu::LREG7 /*LREG7 = 1/N*/, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, 0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, 0, 0);

        TTI_SFPTRANSP(0, 0, 0, 0);

        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, ckernel::ADDR_MOD_3, 64);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG1, 0, ckernel::ADDR_MOD_3, 64 + 2);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, ckernel::ADDR_MOD_3, 64 + 16);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, ckernel::ADDR_MOD_3, 64 + 18);

        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 128);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128 + 2);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG6, 0, ckernel::ADDR_MOD_3, 128 + 16);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG7, 0, ckernel::ADDR_MOD_3, 128 + 18);
    }
    else
    {
        // saves data raw to dst reg
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);
    }
}

namespace ckernel
{
namespace sfpu
{
sfpi_inline void _program_welfords_replay_()
{
    lltt::record(0, 23);
    _compute_welfords_math_();      // 9 TTI instructions
    _welfords_load_initial_data_(); // 9 TTI instructions
    TTI_SFPTRANSP(0, 0, 0, 0);
    /*past_mean = dst1*/ TTI_SFPLOAD(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);
    /*past_var = dst2*/ TTI_SFPLOAD(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);
    // wiping LREG 6 and 7 since they may be filled with garbage data
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, 0, 0);
}

template <std::size_t reciprocal_size>
void _calculate_welfords_online_(
    uint32_t current_sample,
    const uint32_t final_sample,
    uint32_t skip_n_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut,
    const bool reformat_dst_to_col_on_end,
    const bool convert_M2_to_var)
{
    // Pack the mean into the first face of the mean dst reg. Convert M2 to variance and pack into the first face of the var dst reg.
    if (convert_M2_to_var)
    {
        _load_recip_current_sample_(current_sample - 1, reciprocal_lut);
        TTI_SFPMUL(ckernel::p_sfpu::LREG7 /*LREG7 = 1/N*/, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, 64);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, 128);
        return;
    }

    const uint32_t sample_count = _welfords_main_(current_sample, final_sample, skip_n_rows, reciprocal_lut);
    if (sample_count == final_sample)
    {
        _save_data_(reformat_dst_to_col_on_end);
    }
}
} // namespace sfpu
} // namespace ckernel
