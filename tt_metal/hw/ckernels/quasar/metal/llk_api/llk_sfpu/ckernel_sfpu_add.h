// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/// Float ADD math policy, shared by Dest and SrcS.
struct AddFloatMath {
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat a, sfpi::vFloat b) { return a + b; }
};

/// Int32 ADD math policy. SIGN_MAGNITUDE_FORMAT casts sign-magnitude Dest values to 2's complement around the add.
template <bool SIGN_MAGNITUDE_FORMAT>
struct AddIntMath {
    sfpi_inline static sfpi::vInt apply(sfpi::vInt a, sfpi::vInt b) {
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            a = sfpi::convert<sfpi::vInt>(sfpi::as<sfpi::vSMag>(a));
            b = sfpi::convert<sfpi::vInt>(sfpi::as<sfpi::vSMag>(b));
        }
        sfpi::vInt sum = a + b;
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            sum = sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag>(sum));
        }
        return sum;
    }
};

/// Int32 ADD over Dest. Explicit I32 layout because implied integer formats are broken on Quasar (TEN-4674).
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    DataFormat FMT = DataFormat::Int32,
    [[maybe_unused]] int INSTRUCTION_MODE = 0,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_add_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(FMT == DataFormat::Int32, "Only Int32 currently supported for SFPU integer add on Quasar");

    using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::I32, sfpi::vInt>>;
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    calculate_binary_operands<AddIntMath<SIGN_MAGNITUDE_FORMAT>, ITERATIONS>(
        Operand{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
        Operand{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
        Operand{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
}

}  // namespace sfpu
}  // namespace ckernel
