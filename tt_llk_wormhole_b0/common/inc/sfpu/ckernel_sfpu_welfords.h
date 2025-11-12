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

/**
 * @brief Loads the reciprocal of (idx + 1) into LREG7, using a lookup table if available.
 *
 * This function either loads a precomputed reciprocal value from the provided lookup table
 * (reciprocal_lut) into the LREG7 register, or, if the lookup table entry is not available,
 * computes the reciprocal at runtime as 1.0f/(idx + 1) and loads its bit representation
 * into the register.
 *
 * @tparam reciprocal_size The number of entries in the reciprocal lookup table.
 * @param idx The (zero-based) index (in the reciprocal lookup table) of the value to load.
 * @param reciprocal_lut Lookup table containing precomputed reciprocals packed as uint32_t.
 *
 * @note The reciprocal is written to ckernel::p_sfpu::LREG7.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _load_recip_of_idx_(const uint32_t idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    if constexpr (reciprocal_size > 0)
    {
        const auto reciprocal = reciprocal_lut[idx];
        TT_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, reciprocal >> 16);
        TT_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, reciprocal & 0xFFFF);
        return;
    }

    // Fallback to float division
    const float reciprocal = 1.0f / static_cast<float>(idx + 1);
    const FloatBits reciprocal_bits(reciprocal);
    TT_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, reciprocal_bits.high16);
    TT_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, reciprocal_bits.low16);
}

/*
 * @brief Loads the current input at row (I * 4) + J from the current tile in dst reg at offset 0.
 *
 * @tparam I
 * @tparam J
 * 4 inputs each from 32 columns at the current offset are loaded into the LREG0-3 registers
 * respectively from the current tile in dst 0.
 */
template <uint32_t I, uint32_t J>
sfpi_inline void _welfords_load_block_()
{
    constexpr uint32_t tile_offset    = 0; // offset for tile 0 in dst
    constexpr uint32_t dst_reg_offset = tile_offset + (I * 32) + (4 * J);
    constexpr uint32_t offset0        = dst_reg_offset;
    constexpr uint32_t offset1        = dst_reg_offset + 2;
    constexpr uint32_t offset2        = dst_reg_offset + 16;
    constexpr uint32_t offset3        = dst_reg_offset + 18;

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG2, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset2);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, offset3);
    TTI_SFPTRANSP(0, 0, 0, 0);
}

/**
 * @brief Computes a step of Welford's online algorithm for calculating the mean and m2.
 *
 * This function applies Welford's algorithm for a new sample. It updates the running mean and the
 * running (uncorrected) m2 based on the input value in the given LREG and the previous mean
 * and the running (uncorrected) m2.
 * The new input value, previous mean, running "population m2" (i.e., M2 sum), and the
 * reciprocal (1/N) for the sample count are assumed to be already loaded into their corresponding
 * LREGs as tabulated below.
 *
 * - input_lreg: Input value (new sample x_{N+1}) (Either LREG0, LREG1, LREG2, or LREG3)
 * - LREG4: Mean of previous samples (mean_{N})
 * - LREG5: Running sum of squared differences from the current mean (M2_{N})
 * - LREG6: Placeholder for the new mean (mean_{N+1})
 * - LREG7: Reciprocal of sample count (1/(N+1))
 *
 * The computation proceeds as:
 *   1. mean_{N+1} = mean_{N} + ((1/(N+1)) * (x_{N+1} - mean_{N}))
 *   2. M2_{N+1} = M2_{N} + (x_{N+1} - mean_{N}) * (x_{N+1} - mean_{N+1})
 *
 * The updated mean and M2 are left in LREG4 and LREG5, respectively.
 *
 * @tparam input_lreg The input LREG to use for the computation. Either LREG0, LREG1, LREG2, or LREG3.
 * @return None. The updated mean and M2 are left in LREG4 and LREG5, respectively.
 */
template <uint32_t input_lreg>
sfpi_inline void _compute_welfords_row_()
{
    // mean calculation
    // ----------------
    // mean_{N_+1} = mean_{N} + ((1/N+1) * (x_{N+1} - mean_{N}))
    // Let α = x_{N+1} - mean_{N} and β = 1/N+1
    // Then mean_{N+1} = mean_{N} + α * β

    // 1. Calculate α = x_{N+1} - mean_{N}
    // LREG6 = -1 * LREG4 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG6, 0);
    TTI_SFPNOP; // Next cycle cannot read from LREG6 (2-cycle operation)

    // 2. Calculate α * β + mean_{N}
    // LREG6 = LREG6 * LREG7 + LREG4
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG6, 0);

    // m2 calculation
    // ---------------
    // m2_{N+1} = m2_{N} + (x_{N+1} - mean_{N}) * (x_{N+1} - mean_{N+1})
    // Let α = x_{N+1} - mean_{N} and β = x_{N+1} - mean_{N+1}
    // Then m2_{N+1} = m2_{N} + α * β

    // 1. Re-calculate α in lREG4 since LREG6 now contains the new mean
    // LREG4 = -1 * LREG4 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG4, 0);

    // 2. Calculate β = x_{N+1} - mean_{N+1}
    // input_lreg = -1 * LREG6 + input_lreg
    TTI_SFPMAD(ckernel::p_sfpu::LREG11 /*-1*/, ckernel::p_sfpu::LREG6, input_lreg, input_lreg, 0);
    TTI_SFPNOP; // Next cycle cannot read from input_lreg (2-cycle operation)

    // 3. Calculate m2_{N+1} = α * β + m2_{N}
    // LREG5 = LREG4 * input_lreg + LREG5
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, input_lreg, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);

    // Moves mean to LREG4 from LREG6 since it now is considered the past mean
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG4, 0);
}

/**
 * @brief The number of instructions required to calculate the running mean and m2 for a single
 * row of 32 columns. If _compute_welfords_row_ is modified, this value must be updated.
 */
constexpr uint32_t WELFORD_INSTR_PER_ROW = 8;

/**
 * @brief Programs the replay buffer for the Welford's algorithm.
 *
 * This function programs the replay buffer for the Welford's algorithm. It places the
 * _compute_welfords_row_ tailored for each input LREG in the replay buffer.
 */
sfpi_inline void _program_welfords_replay_buffer_()
{
    lltt::record(0, WELFORD_INSTR_PER_ROW * 4);

    _compute_welfords_row_<ckernel::p_sfpu::LREG0>();
    _compute_welfords_row_<ckernel::p_sfpu::LREG1>();
    _compute_welfords_row_<ckernel::p_sfpu::LREG2>();
    _compute_welfords_row_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Executes the replay buffer for the Welford's algorithm.
 *
 * This function replays the instructions in the buffer for the input LREG.
 *
 * @tparam input_lreg The index of the input LREG to replay. LREG0-3.
 */
template <uint32_t input_lreg>
sfpi_inline void _execute_welfords_row_replay_buffer_()
{
    lltt::replay(WELFORD_INSTR_PER_ROW * input_lreg, WELFORD_INSTR_PER_ROW);
}

/**
 * @brief Calculates running mean and m2 for a single block of 4 rows and 32 columns.
 *
 * @tparam I
 * @tparam J
 * @param start_idx The index of the first element in the block; used to index the reciprocal lookup table.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 *
 * This is a helper function that performs all three steps for a single block:
 * load inputs, load reciprocal and compute running mean and m2. Each block has 4 rows of 32 columns.
 */
template <std::size_t reciprocal_size, uint32_t I, uint32_t J>
sfpi_inline void _calculate_welfords_block_(uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    _welfords_load_block_<I, J>();

    _load_recip_of_idx_<reciprocal_size>(start_idx, reciprocal_lut);
    _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG0>();

    _load_recip_of_idx_<reciprocal_size>(start_idx + 1, reciprocal_lut);
    _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG1>();

    _load_recip_of_idx_<reciprocal_size>(start_idx + 2, reciprocal_lut);
    _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG2>();

    _load_recip_of_idx_<reciprocal_size>(start_idx + 3, reciprocal_lut);
    _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG3>();
}

/**
 * @brief Calculates running mean and m2 for a single block of 4 rows and 32 columns on a subset of
 *        rows starting from start_row.
 *
 * @tparam I
 * @tparam J
 * @param start_idx The index of the first element in the block; used to index the reciprocal lookup table.
 * @param start_row The offset of the row to start from. Only rows starting from this offset are
 *                   processed in the block. Should be 0 <= start_row < 4.
 * @param end_row The offset of the row to end at. Only rows up to this offset are
 *                processed in the block (not including the row at this offset).
 *                Should comply with the condition: start_row <= end_row <= 4.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 *
 * This is a helper function that performs all three steps for a single block:
 * load inputs, load reciprocal and compute running mean and m2. Each block has 4 rows of 32 columns.
 */
template <std::size_t reciprocal_size, uint32_t I, uint32_t J>
sfpi_inline void _calculate_welfords_block_w_offset_(
    uint32_t& start_idx, uint32_t start_row, uint32_t end_row, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    // These are the row indices of the block in the tile.
    constexpr uint32_t block_min_row_idx = I * 16 + J * 4;
    constexpr uint32_t block_max_row_idx = block_min_row_idx + 4;

    // If the start_row and end_row don't intersect with this block, we don't need to process this.
    if (((start_row >= block_max_row_idx) || (end_row <= block_min_row_idx)))
    {
        return;
    }

    // Trim the start_row and end_row so we only look at rows in this window
    const uint32_t block_start_row = std::max(block_min_row_idx, start_row);
    const uint32_t block_end_row   = std::min(block_max_row_idx, end_row);

    // Make the start_row and end_row relative to this block
    start_row = block_start_row - block_min_row_idx;
    end_row   = block_end_row - block_min_row_idx;

    _welfords_load_block_<I, J>();

    if ((start_row == 0) && (end_row > 0))
    {
        _load_recip_of_idx_<reciprocal_size>(start_idx, reciprocal_lut);
        _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG0>();
        ++start_idx;
    }
    if ((start_row <= 1) && (end_row > 1))
    {
        _load_recip_of_idx_<reciprocal_size>(start_idx, reciprocal_lut);
        _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG1>();
        ++start_idx;
    }
    if ((start_row <= 2) && (end_row > 2))
    {
        _load_recip_of_idx_<reciprocal_size>(start_idx, reciprocal_lut);
        _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG2>();
        ++start_idx;
    }
    if ((start_row <= 3) && (end_row > 3))
    {
        _load_recip_of_idx_<reciprocal_size>(start_idx, reciprocal_lut);
        _execute_welfords_row_replay_buffer_<ckernel::p_sfpu::LREG3>();
        ++start_idx;
    }
}

namespace ckernel
{
namespace sfpu
{
/**
 * @brief Clears the previous Welford's mean and m2 stored in registers LREG4 and LREG5.
 *
 * This function zeroes out the registers LREG4 and LREG5 used for storing the previous Welford's
 * values, preparing for a new calculation cycle. Typically invoked at the beginning of the
 * calculation for a new group of tiles over which the mean and m2 are calculated.
 */
sfpi_inline void _clear_previous_mean_and_m2_()
{
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
}

/*
 * @brief Calculates the Welford's online algorithm for a tile in the dst reg.
 *
 * This function calculates the Welford's online algorithm for a tile in the dst reg.
 * It assumes that the current mean and m2 values are placed in LREG4 and LREG5, respectively.
 * It operates on 32 inputs each from 32 columns, present in a 32x32 tile in dst 0. At the end,
 * the updated mean and m2 values are stored in LREG4 and LREG5, respectively.
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param start_idx The index of the first element in the tile; used to index the reciprocal lookup table.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_welfords_tile_(uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    // We load 4 rows of a tile (with 32 columns each) at a time and process them.
    // To finish the entire tile, we need to repeat this process 8 times.

    // Process the first block (4 rows of 32 columns)
    _calculate_welfords_block_<reciprocal_size, 0, 0>(start_idx, reciprocal_lut);

    // Repeat this 7 more times to process the remaining blocks
    _calculate_welfords_block_<reciprocal_size, 0, 1>(start_idx + 4, reciprocal_lut);
    _calculate_welfords_block_<reciprocal_size, 0, 2>(start_idx + 8, reciprocal_lut);
    _calculate_welfords_block_<reciprocal_size, 0, 3>(start_idx + 12, reciprocal_lut);

    _calculate_welfords_block_<reciprocal_size, 1, 0>(start_idx + 16, reciprocal_lut);
    _calculate_welfords_block_<reciprocal_size, 1, 1>(start_idx + 20, reciprocal_lut);
    _calculate_welfords_block_<reciprocal_size, 1, 2>(start_idx + 24, reciprocal_lut);
    _calculate_welfords_block_<reciprocal_size, 1, 3>(start_idx + 28, reciprocal_lut);
}

/*
 * @brief Calculates the Welford's online algorithm for a tile in the dst reg on a subset of rows.
 *
 * This function calculates the Welford's online algorithm for a tile in the dst reg.
 * It assumes that the current mean and m2 values are placed in LREG4 and LREG5, respectively.
 * It operates on 32 inputs each from 32 columns, present in a 32x32 tile in dst 0. At the end,
 * the updated mean and m2 values are stored in LREG4 and LREG5, respectively.
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param start_idx The index of the first element in the tile; used to index the reciprocal lookup
 *                  table.
 * @param start_row The offset of the row to start from. Only rows starting from this offset are
 *                   processed in the tile. Should be 0 <= start_row <= 31.
 * @param num_rows The number of rows to process. Should be 0 <= num_rows <= 32. Also,
 *                 0 <= start_row + num_rows <= 32.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _calculate_welfords_partial_tile_(
    uint32_t start_idx, uint32_t start_row, uint32_t num_rows, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    if (num_rows == 0)
    {
        return;
    }

    const uint32_t end_row = start_row + num_rows;

    // We load 4 rows of a tile (with 32 columns each) at a time and process them.
    // To finish the entire tile, we need to repeat this process 8 times.
    _calculate_welfords_block_w_offset_<reciprocal_size, 0, 0>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 0, 1>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 0, 2>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 0, 3>(start_idx, start_row, end_row, reciprocal_lut);

    _calculate_welfords_block_w_offset_<reciprocal_size, 1, 0>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 1, 1>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 1, 2>(start_idx, start_row, end_row, reciprocal_lut);
    _calculate_welfords_block_w_offset_<reciprocal_size, 1, 3>(start_idx, start_row, end_row, reciprocal_lut);
}

/*
 * @brief Stores the mean and m2 values to the tile in the dst reg at offset 0 and 1 respectively.
 *
 * This function stores the mean and m2 values to the tile in the dst reg at offset 0 and 1
 * respectively. The values are stored in "raw" format. i.e. a total of 32 mean values are stored in
 * the first face of the tile in dst at offset 0. These values are stored at even indices only.
 * Thus, the values take up 4 rows in the first face with 8 values per row. The m2 values are stored
 * in the same way but at tile offset 1.
 */
sfpi_inline void _store_mean_m2_to_dst_()
{
    constexpr uint32_t mean_tile_offset = 0;  // offset for the mean tile in dst
    constexpr uint32_t m2_tile_offset   = 64; // offset for the m2 tile in dst

    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, m2_tile_offset);
}

/*
 * @brief Stores the mean and m2 values to the tile in the dst reg at offset 0 and 1 respectively
 *        for a given group.
 *
 * This function does the same as _store_mean_m2_to_dst_ but allows for the data to be stored at an
 * offset that is dependent on the group id. This allows for data of multiple groups to be stored in
 * the same tile.
 * @note Since group_id is known at runtime, we use TT_SFPSTORE instead of TTI_SFPSTORE.
 * @param group_id The group id to store the data for.
 */
sfpi_inline void _store_mean_m2_to_dst_group_(uint32_t group_id)
{
    constexpr uint32_t mean_tile_offset = 0;  // offset for the mean tile in dst
    constexpr uint32_t m2_tile_offset   = 64; // offset for the m2 tile in dst

    TT_SFPSTORE(ckernel::p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + (group_id << 2));
    TT_SFPSTORE(ckernel::p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, m2_tile_offset + (group_id << 2));
}

/*
 * @brief Loads the mean and m2 values from the tile in the dst reg into LREGs.
 *
 * This function loads the mean and m2 values from the tiles in the dst reg into LREGs.
 * The mean is loaded into LREG4, and the m2 is loaded into LREG5.
 * It assumes that the mean and m2 values were stored by _store_mean_m2_to_dst_
 * (i.e., they are each stored in the "raw" format).
 */
sfpi_inline void _load_mean_m2_from_dst_()
{
    constexpr uint32_t mean_tile_offset = 0;  // offset for the mean tile in dst
    constexpr uint32_t m2_tile_offset   = 64; // offset for the m2 tile in dst

    TTI_SFPLOAD(ckernel::p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, m2_tile_offset);
}

/*
 * @brief Loads the mean and m2 values from the tile in the dst reg into LREGs for a given group.
 *
 * This function does the same as _load_mean_m2_from_dst_ but allows for the data to be loaded from
 * an offset that is dependent on the group id. This allows for data of multiple groups to be loaded
 * from the same tile.
 * @note Since group_id is known at runtime, we use TT_SFPLOAD instead of TTI_SFPLOAD.
 * @param group_id The group id to load the data for.
 */
sfpi_inline void _load_mean_m2_from_dst_group_(uint32_t group_id)
{
    constexpr uint32_t mean_tile_offset = 0;  // offset for the mean tile in dst
    constexpr uint32_t m2_tile_offset   = 64; // offset for the m2 tile in dst

    TT_SFPLOAD(ckernel::p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + (group_id << 2));
    TT_SFPLOAD(ckernel::p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, m2_tile_offset + (group_id << 2));
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg.
 *
 * This function stores the mean and variance values to the tile in the dst reg.
 * It assumes that the mean and m2 values are placed in LREG4 and LREG5, respectively.
 * These values are placed in the first row of the tile in dst. The reciprocal LUT, if provided,
 * is used to load the reciprocal of the sample count.
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_row_(uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(scale_idx, reciprocal_lut);
    // Move mean to LREG0
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG1, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, 0, 0);
    // Convert M2 to variance and move to LREG4
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, 0, 0);

    // Move all the values to a single row
    TTI_SFPTRANSP(0, 0, 0, 0);

    constexpr uint32_t offset0 = 0;
    constexpr uint32_t offset1 = 2;
    constexpr uint32_t offset2 = 16;
    constexpr uint32_t offset3 = 18;

    constexpr uint32_t mean_tile_offset = 0; // offset for the mean tile in dst

    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + offset0);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + offset1);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + offset2);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, mean_tile_offset + offset3);

    constexpr uint32_t var_tile_offset = 64; // offset for the var tile in dst

    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, var_tile_offset + offset0);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG5, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, var_tile_offset + offset1);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG6, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, var_tile_offset + offset2);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_3, var_tile_offset + offset3);
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg.
 *
 * This function stores the mean and variance values to the tile in the dst reg.
 * It assumes that the mean and m2 values are placed in LREG4 and LREG5, respectively.
 * These values are placed in the first face of the tile in dst. The reciprocal LUT, if provided,
 * is used to load the reciprocal of the sample count.
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_raw_(uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(scale_idx, reciprocal_lut);

    // Convert M2 to variance in LREG5
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 0);

    constexpr uint32_t mean_tile_offset = 0; // offset for the mean tile in dst
    TTI_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, mean_tile_offset);

    constexpr uint32_t var_tile_offset = 64; // offset for the var tile in dst
    TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, var_tile_offset);
}

/*
 * @brief Stores the mean and variance values to the tile in the dst reg for a given group.
 *
 * This function does the same as _store_mean_var_to_dst_raw_ but allows for the data to be stored
 * at an offset that is dependent on the group id.
 * @note Since group_id is known at runtime, we use TT_SFPSTORE instead of TTI_SFPSTORE.
 * @tparam reciprocal_size The size of the reciprocal lookup table.
 * @param group_id The group id to store the data for.
 * @param scale_idx The index of the scale value to use for the variance calculation.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 */
template <std::size_t reciprocal_size>
sfpi_inline void _store_mean_var_to_dst_raw_group_(uint32_t group_id, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)
{
    _load_recip_of_idx_<reciprocal_size>(scale_idx, reciprocal_lut);

    // Convert M2 to variance in LREG5
    TTI_SFPMAD(ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 0);

    constexpr uint32_t mean_tile_offset = 0; // offset for the mean tile in dst
    TT_SFPSTORE(ckernel::p_sfpu::LREG4, 0, ckernel::ADDR_MOD_3, mean_tile_offset + (group_id << 2));

    constexpr uint32_t var_tile_offset = 64; // offset for the var tile in dst
    TT_SFPSTORE(ckernel::p_sfpu::LREG5, 0, ckernel::ADDR_MOD_3, var_tile_offset + (group_id << 2));
}
} // namespace sfpu
} // namespace ckernel
