// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

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

    // Raw TT_ instructions instead of SFPI: SFPI builds each runtime-address SFPLOAD/SFPSTORE word with a
    // compiler call that GCC will not hoist out of a loop past a runtime branch, so the words get rebuilt
    // every tile (~10 cycles/tile). Load/store use explicit INT32 because implied formats are broken (TEN-4674).
    constexpr auto instr_mod = p_sfpu::sfpmem::INT32;
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, in1_offset + (d << 1));

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }

        TTI_SFPIADD(0x0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }

        TT_SFPSTORE(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
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
