// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/// Math policy for ADD: a + b. Shared by the Dest and SrcS paths via @ref calculate_binary_operands;
/// any rounding is the output operand's store policy.
struct AddMath {
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat a, sfpi::vFloat b) { return a + b; }
};

/// Math policy for Int32 ADD: a + b on the integer adder. Dest holds Int32 as 2's complement
/// (UNP_DEST / Int32 L1) or sign-magnitude (copy_tile Int8 + fp32_dest_acc FPU); with
/// SIGN_MAGNITUDE_FORMAT the operands are cast to 2's complement around the add.
template <bool SIGN_MAGNITUDE_FORMAT>
struct AddIntMath {
    sfpi_inline static sfpi::vInt apply(sfpi::vInt a, sfpi::vInt b) {
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            a = sfpi::impl_::smag_to_int(sfpi::as<sfpi::vSMag>(a));
            b = sfpi::impl_::smag_to_int(sfpi::as<sfpi::vSMag>(b));
        }
        sfpi::vInt sum = a + b;
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            sum = sfpi::as<sfpi::vInt>(sfpi::impl_::int_to_smag(sum));
        }
        return sum;
    }
};

/**
 * @brief Int32 ADD over Dest tiles: dest[out] = dest[in0] + dest[in1], one face per call.
 *
 * Loads and stores use the explicit I32 layout: implied formats with unpack-to-Dest are broken
 * for integers on Quasar (TEN-4674).
 *
 * @tparam ITERATIONS: Number of SFPU passes (each covers 2 rows).
 * @tparam FMT: Dest format, values = <Int32>.
 * @tparam SIGN_MAGNITUDE_FORMAT: See @ref AddIntMath.
 * @tparam TILE_SHAPE: Destination tile shape used to calculate operand offsets.
 */
template <
    bool APPROXIMATION_MODE /*maybe_unused*/,
    int ITERATIONS = SFPU_ITERATIONS,
    DataFormat FMT = DataFormat::Int32,
    int INSTRUCTION_MODE /*maybe_unused*/ = 0,
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
