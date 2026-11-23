// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_converter.h"

// Quasar: no ckernel_sfpu_load_config.h, and nothing in this file uses it.

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

// The row-range fills (fill_rows_tile) are not ported to Quasar. The Compute API keeps their entry points on every
// arch so fill.h stays includable; a kernel that calls one fails to build.
[[gnu::error("fill_rows_tile is not supported on Quasar")]] void fill_rows_is_not_supported_on_quasar();

template <bool APPROXIMATION_MODE>
inline void _calculate_fill_rows_(
    [[maybe_unused]] const std::uint32_t row_begin,
    [[maybe_unused]] const std::uint32_t row_end,
    [[maybe_unused]] const float value) {
    fill_rows_is_not_supported_on_quasar();
}

template <bool APPROXIMATION_MODE, std::uint32_t ROW_BEGIN, std::uint32_t ROW_END>
inline void _calculate_fill_rows_ct_([[maybe_unused]] const float value) {
    fill_rows_is_not_supported_on_quasar();
}
}  // namespace ckernel::sfpu
