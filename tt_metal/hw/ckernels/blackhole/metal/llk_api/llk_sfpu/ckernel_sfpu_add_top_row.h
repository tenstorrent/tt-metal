// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
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
 *                - DataFormat::UInt32: Use integer implementation with INT32 instruction mode
 *                  (UInt32 is stored as raw unsigned bits — no sign-magnitude conversion needed)
 *                - DataFormat::Float32: Uses floating-point implementation with FP32 instruction mode
 * @param tile_idx_0 The index of the first tile in the Dest register to operate on.
 * @param tile_idx_1 The index of the second tile in the Dest register to operate on.
 * @param tile_idx_dst The index of the result tile in the Dest register where the result will be stored.
 */
template <DataFormat format>
inline void calculate_add_top_row(
    const std::uint32_t tile_idx_0 = 0, const std::uint32_t tile_idx_1 = 0, const std::uint32_t tile_idx_dst = 0) {
    static_assert(
        format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32,
        "Unsupported data format. Supported formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::Float32");

    // sfpi dst_reg[] indexes in row units where a full tile is 32 (SFP_DESTREG_STRIDE),
    // i.e. half the raw TT_SFPLOAD immediate (tile = 64). The four raw sub-face offsets
    //   {0, +2, +16, +18}  ->  sfpi indices  {0, +1, +8, +9}
    // pick face0/face2 (even/odd cols) and face1/face3 (even/odd cols) of the top rows.
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
        // Int32 / UInt32. INT32 layout leaves UInt32 raw and (on Blackhole) is a no-op
        // sign-magnitude conversion, matching the original INSTRUCTION_MODE == INT32 path.
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            sfpi::vInt a = sfpi::dst_reg[off0 + sub[i]].mode<sfpi::DataLayout::I32>();
            sfpi::vInt b = sfpi::dst_reg[off1 + sub[i]].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[offd + sub[i]].mode<sfpi::DataLayout::I32>() = a + b;
        }
    }
}

/**
 * @brief Initialize SFPU configuration register for the add-top-row kernel.
 *        The compute path is now pure sfpi (see calculate_add_top_row), so the integer/float
 *        replay buffers the raw implementation programmed here are no longer needed.
 */
inline void init_add_top_row() { _init_sfpu_config_reg(); }

}  // namespace sfpu
}  // namespace ckernel
