// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Add top row operation for a 32x32 tile.
 *        Automatically chooses between integer and floating-point implementations based on the data format.
 *        Takes the top row of tile 0 (first 16 datums of face 0 and first 16 of face 1) and adds them
 *        with the top row of tile 1 (first 16 datums of face 2 and first 16 of face 3).
 * @tparam format The data format that determines which implementation to use.
 *                Supported formats:
 *                - DataFormat::Int32: Use integer implementation with INT32 instruction mode
 *                  (Quasar's DataFormat has no UInt32, so Blackhole's UInt32 path is not ported)
 *                - DataFormat::Float32: Uses floating-point implementation with FP32 instruction mode
 * @param tile_idx_0 The index of the first tile in the Dest register to operate on.
 * @param tile_idx_1 The index of the second tile in the Dest register to operate on.
 * @param tile_idx_dst The index of the result tile in the Dest register where the result will be stored.
 */
template <DataFormat format>
inline void calculate_add_top_row(
    const std::uint32_t tile_idx_0 = 0, const std::uint32_t tile_idx_1 = 0, const std::uint32_t tile_idx_dst = 0) {
    static_assert(
        format == DataFormat::Int32 || format == DataFormat::Float32,
        "Unsupported data format. Supported formats are: DataFormat::Int32, DataFormat::Float32");

    // sfpi dst_reg[] indexes in SFPU passes, 8 per face and 32 per tile, and a pass reads the same
    // lanes as on Blackhole (4 Dest rows x 8 even or odd columns). The Blackhole indices
    // {0, +1, +8, +9} therefore cover the same datums here: rows 0-3 of face 0 (index 0 even and
    // 1 odd columns) and of face 1 (index 8, 9), i.e. the top four rows of the tile.
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    const std::uint32_t off0 = tile_idx_0 * dst_tile_size_sfpi;
    const std::uint32_t off1 = tile_idx_1 * dst_tile_size_sfpi;
    const std::uint32_t offd = tile_idx_dst * dst_tile_size_sfpi;

    constexpr std::uint32_t sub[4] = {0, 1, 8, 9};

    if constexpr (format == DataFormat::Float32) {
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            sfpi::vFloat a = sfpi::dst_reg[off0 + sub[i]];
            sfpi::vFloat b = sfpi::dst_reg[off1 + sub[i]];
            sfpi::dst_reg[offd + sub[i]] = a + b;
        }
    } else {
        // Int32. The INT32 layout loads the raw 32 bits; Int32 unpacked to Dest is 2's
        // complement on Quasar.
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            sfpi::vInt a = sfpi::dst_reg[off0 + sub[i]].mode<sfpi::DataLayout::I32>();
            sfpi::vInt b = sfpi::dst_reg[off1 + sub[i]].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[offd + sub[i]].mode<sfpi::DataLayout::I32>() = a + b;
        }
    }
}

/**
 * @brief Init for the add-top-row kernel. The kernel is pure sfpi and addresses Dest from the
 *        tile indices, so only the Dest counters need resetting. Run it as one call per tile
 *        (VectorMode::None): it reaches all four rows itself and must not be walked per face.
 */
inline void init_add_top_row() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

}  // namespace sfpu
}  // namespace ckernel
