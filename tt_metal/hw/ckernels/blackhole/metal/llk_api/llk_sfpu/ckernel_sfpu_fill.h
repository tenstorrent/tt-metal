// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_(const float value) {
    // SFPU microcode
    sfpi::vFloat fill_val = value;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void _calculate_fill_int_(const std::uint32_t value) {
    // SFPU microcode
    if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32) {
        // Materialize the 32-bit immediate once, outside the loop
        sfpi::vInt fill_val = value;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = fill_val;
            sfpi::dst_reg++;
        }
    } else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::UINT16) {
        // Materialize the 16-bit immediate once, outside the loop
        sfpi::vUInt fill_val = static_cast<std::uint16_t>(value & 0xFFFF);
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = fill_val;
            sfpi::dst_reg++;
        }
    } else {
        static_assert(false, "INSTRUCTION_MODE must be one of: INT32, UINT16.");
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_bitcast_(const std::uint32_t value_bit_mask) {
    // SFPU microcode
    sfpi::vFloat fill_val = Converter::as_float(value_bit_mask);

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}

// Rows [row_begin, row_end) of the 32x32 tile at the DEST base <- value; every other row is left as it is. Called once
// per tile (VectorMode::None). The tile is 32 sfpi vectors: face f (rows 16 * (f / 2) .., columns 16 * (f % 2) ..) is
// vectors 8f .. 8f + 7, and vector 2b + g of a face holds the face's rows 4b .. 4b + 3 at the columns of parity g,
// lane 8j + k being (row 4b + j, column 2k + g). A band with all four rows in the range is stored whole; a band the
// range cuts is masked by the lane's row, vConstTileId / 16 (the tile id is twice the lane index).
template <bool APPROXIMATION_MODE>
inline void _calculate_fill_rows_(const std::uint32_t row_begin, const std::uint32_t row_end, const float value) {
    const sfpi::vFloat fill_val = value;
    for (std::uint32_t band = 0; band < 8; band++) {  // tile rows 4 * band .. 4 * band + 3
        const std::uint32_t r0 = 4 * band;
        const std::uint32_t lo = row_begin > r0 ? row_begin - r0 : 0;  // first lane row of the band in the range
        const std::uint32_t hi = row_end >= r0 + 4 ? 4 : (row_end > r0 ? row_end - r0 : 0);  // one past the last
        if (lo >= hi) {
            continue;
        }
        const std::uint32_t left = 8 * (2 * (band / 4)) + 2 * (band % 4);    // the band's even-column vector, left face
        const std::uint32_t vecs[4] = {left, left + 1, left + 8, left + 9};  // both parities, left and right face
        if (lo == 0 && hi == 4) {
            for (std::uint32_t v : vecs) {
                sfpi::dst_reg[v] = fill_val;
            }
        } else {
            const sfpi::vInt id_lo = static_cast<int>(16 * lo);
            const sfpi::vInt id_hi = static_cast<int>(16 * hi);
            for (std::uint32_t v : vecs) {
                sfpi::vFloat x = sfpi::dst_reg[v];
                v_if(sfpi::vConstTileId >= id_lo && sfpi::vConstTileId < id_hi) { x = fill_val; }
                v_endif;
                sfpi::dst_reg[v] = x;
            }
        }
    }
}

// The same with the rows as template constants. One function per band: a band outside the range compiles to
// nothing, one inside to its four stores or four lane-masked stores with immediate bounds, so the whole fill is
// the stores alone. (Unrolling the run-time loop above instead does not compile: its run-time bounds would have to
// live across the unrolled bands, and sfpi cannot spill a vector object.)
template <std::uint32_t ROW_BEGIN, std::uint32_t ROW_END, std::uint32_t BAND>
inline void _fill_rows_band_ct_(const sfpi::vFloat fill_val) {
    constexpr std::uint32_t r0 = 4 * BAND;
    constexpr std::uint32_t lo = ROW_BEGIN > r0 ? ROW_BEGIN - r0 : 0;
    constexpr std::uint32_t hi = ROW_END >= r0 + 4 ? 4 : (ROW_END > r0 ? ROW_END - r0 : 0);
    if constexpr (lo < hi) {
        constexpr std::uint32_t left = 8 * (2 * (BAND / 4)) + 2 * (BAND % 4);
        constexpr std::uint32_t vecs[4] = {left, left + 1, left + 8, left + 9};
        if constexpr (lo == 0 && hi == 4) {
            sfpi::dst_reg[vecs[0]] = fill_val;
            sfpi::dst_reg[vecs[1]] = fill_val;
            sfpi::dst_reg[vecs[2]] = fill_val;
            sfpi::dst_reg[vecs[3]] = fill_val;
        } else {
#pragma GCC unroll 4
            for (std::uint32_t i = 0; i < 4; i++) {
                sfpi::vFloat x = sfpi::dst_reg[vecs[i]];
                v_if(
                    sfpi::vConstTileId >= static_cast<int>(16 * lo) && sfpi::vConstTileId < static_cast<int>(16 * hi)) {
                    x = fill_val;
                }
                v_endif;
                sfpi::dst_reg[vecs[i]] = x;
            }
        }
    }
}
template <bool APPROXIMATION_MODE, std::uint32_t ROW_BEGIN, std::uint32_t ROW_END>
inline void _calculate_fill_rows_ct_(const float value) {
    const sfpi::vFloat fill_val = value;
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 0>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 1>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 2>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 3>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 4>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 5>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 6>(fill_val);
    _fill_rows_band_ct_<ROW_BEGIN, ROW_END, 7>(fill_val);
}
}  // namespace ckernel::sfpu
