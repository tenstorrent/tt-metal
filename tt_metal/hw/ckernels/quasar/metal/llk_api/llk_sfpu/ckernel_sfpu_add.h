// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief ADD on independently located floating-point operands: output = input0 + input1.
 *
 * Any rounding is the output operand's store policy. Advances explicit indices only; the caller
 * owns setup and synchronization. Each input range must coincide with the output or be disjoint.
 */
template <int ITERATIONS, typename Input0, typename Input1, typename Output>
sfpi_inline void calculate_add_operands(const Input0& input0, const Input1& input1, const Output& output) {
    static_assert(ITERATIONS > 0, "ADD requires at least one SFPI access");
    static_assert(
        std::is_same_v<typename Input0::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Input1::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Output::value_type, sfpi::vFloat>,
        "ADD requires floating-point operands");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = input0.load(d);
        sfpi::vFloat in1 = input1.load(d);
        sfpi::vFloat result = in0 + in1;
        output.store(d, result);
    }
}

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    DataFormat FMT = DataFormat::Int32,
    int INSTRUCTION_MODE = 0,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_add_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(FMT == DataFormat::Int32, "Only Int32 currently supported for SFPU integer add on Quasar");

    constexpr bool is_int = (FMT == DataFormat::Int32);
    // There is a Quasar bug with implied formats + unpack to dest, so use explicit types for
    // integer SFPULOAD/SFPSTORE (TEN-4674).
    constexpr auto instr_mod = is_int ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, in1_offset + (d << 1));

        // Dest layout depends on how operands reached dest:
        //   UNP_DEST / Int32 L1 with 2's-comp tiles -> 2's-comp Int32
        //   copy_tile Int8 + fp32_dest_acc FPU -> sign-mag Int32
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

// Op class for an elementwise integer add of two tiles. Same name and leading template parameters as on
// Wormhole/Blackhole; Quasar supports Int32 only. Int8 copy_tile + fp32_dest_acc FPU (through ELWADD) writes
// sign-magnitude Int32 into Dest, so the kernel runs with SIGN_MAGNITUDE_FORMAT=true, as the compute API did.
template <
    bool APPROXIMATION_MODE,
    DataFormat data_format = DataFormat::Int32,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct AddInt : SfpuBinaryOp<AddInt<APPROXIMATION_MODE, data_format, ITERATIONS, SLOT>, SLOT> {
    static_assert(data_format == DataFormat::Int32, "Unsupported data format for add_int on Quasar. Supported: Int32");
    static constexpr auto& calculate =
        calculate_add_int<APPROXIMATION_MODE, ITERATIONS, data_format, 0, true /* SIGN_MAGNITUDE_FORMAT */, SLOT>;
};

}  // namespace sfpu
}  // namespace ckernel
