// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/** @brief Float ADD math policy, shared by Dest and SrcS. */
struct AddFloatMath {
    using value_type = sfpi::vFloat;
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat a, sfpi::vFloat b) { return a + b; }
};

/**
 * @brief Int32 ADD math policy.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Operands and result are sign-magnitude; cast to 2's complement around the add.
 */
template <bool SIGN_MAGNITUDE_FORMAT>
struct AddIntMath {
    using value_type = sfpi::vInt;
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

/**
 * @brief Int32 ADD over Dest: out = in0 + in1.
 *
 * @tparam SIGN_MAGNITUDE_FORMAT: Dest holds sign-magnitude Int32.
 * @param dst_index_in0: Dest tile index of the first operand.
 * @param dst_index_in1: Dest tile index of the second operand.
 * @param dst_index_out: Dest tile index of the result.
 */
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

    // SFPI costs ~8-10 cycles/tile over a raw TT_SFPLOAD/TT_SFPSTORE loop (emulator, perf_eltwise_binary_sfpu_quasar
    // int ADD: L1_TO_L1 97.6 vs 89.8, MATH_ISOLATE 97.0 vs 86.7). SFPI emits 5 RISC instructions per runtime-indexed
    // Dest access vs 3 hand-written: a per-instruction opcode placeholder GCC cannot CSE, plus a 10-bit address mask.
    using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::I32, sfpi::vInt>>;
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
    calculate_binary_operands<AddIntMath<SIGN_MAGNITUDE_FORMAT>, ITERATIONS>(
        Operand{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
        Operand{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
        Operand{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
}

// ADD math for a SrcS layout: float layouts add floats; I32 adds sign-magnitude Int32, which is how SrcS holds it.
template <sfpi::DataLayout LAYOUT>
using AddMath =
    std::conditional_t<LAYOUT == sfpi::DataLayout::I32, AddIntMath<true /*SIGN_MAGNITUDE_FORMAT*/>, AddFloatMath>;

/** @brief SrcS ADD op type (float or Int32 by LAYOUT); Sfpi only. */
template <sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
using AddSrcs = SrcsBinary<AddMath<LAYOUT>, LAYOUT, resolve_sfpu_impl<IMPL, false /*HAS_LOADMACRO*/>()>;

}  // namespace sfpu
}  // namespace ckernel
