// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel::sfpu {

// Int32 comparison via SFPSWAP:
//   Dest stores Int32 in sign-magnitude format; SFPSWAP mod VEC_MIN_MAX compares sign-magnitude
//   integers directly (no two's-complement conversion needed and no overflow on mixed signs).
//
//   Destination Index Tracking Mode is enabled (bit [2] of SFPU_CONTROL_REG): with VC/VD taken
//   from LREG[0..3], SFPSWAP additionally performs the same conditional swap on LREG[4..7]
//   (LREG(4+i) mirrors LREG(i)). We exploit this to turn the compare-and-swap into a boolean
//   0/1 selection without any predicated control flow (no SETCC/ENCC/COMPC).
//
//   Layout per iteration (for LT: A < B):
//     VD=LREG0=B, VC=LREG1=A, mirror LREG4=0 (paired with VD), LREG5=1 (paired with VC)
//     SFPSWAP(VEC_MIN_MAX) swaps iff SignMagIsSmaller(VC,VD) = (A<B):
//       - A<B  -> swap -> LREG4 <- old LREG5 = 1
//       - A>=B -> no swap -> LREG4 = 0
//     Store LREG4 -> 1 if A<B else 0.
//   Other ops (GT/LE/GE) use the same pattern with operand positions and mirror values adjusted
//   so the desired boolean always ends up in LREG4.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Enable Destination Index Tracking Mode so SFPSWAP mirrors LREG[0..3] swaps onto LREG[4..7].
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    // Operand placement:
    //   vd_is_in1: true  -> VD=in1 (B), VC=in0 (A) ; used by LT and GE
    //             false -> VD=in0 (A), VC=in1 (B) ; used by GT and LE
    // Mirror initial values (so LREG4 ends up with the desired result after swap):
    //   LT:  LREG4=0, LREG5=1    GE:  LREG4=1, LREG5=0
    //   GT:  LREG4=0, LREG5=1    LE:  LREG4=1, LREG5=0
    constexpr bool vd_is_in1 = (RELATIONAL_OP == SfpuType::lt) || (RELATIONAL_OP == SfpuType::ge);
    constexpr uint32_t lreg4_init = (RELATIONAL_OP == SfpuType::ge || RELATIONAL_OP == SfpuType::le) ? 0x01 : 0x00;
    constexpr uint32_t lreg5_init = (RELATIONAL_OP == SfpuType::ge || RELATIONAL_OP == SfpuType::le) ? 0x00 : 0x01;

    constexpr uint dst_tile_size = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (vd_is_in1) {
            TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
            TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        } else {
            TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
            TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        }

        TTI_SFPLOADI(p_sfpu::LREG4, SFPLOADI_MOD0_USHORT, lreg4_init);
        TTI_SFPLOADI(p_sfpu::LREG5, SFPLOADI_MOD0_USHORT, lreg5_init);

        // VEC_MIN_MAX: puts min in VD=LREG0, max in VC=LREG1; index-tracked swap mirrors on LREG4/5.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        TT_SFPSTORE(p_sfpu::LREG4, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }

    // Restore default SFPU config (disable Destination Index Tracking Mode).
    _init_sfpu_config_reg();
}

// IEEE-754 float32 NaN: exponent all 1s and nonzero mantissa => |bits| > 0x7F800000.
sfpi_inline auto is_nan(sfpi::vInt abs_bits) { return abs_bits > 0x7F800000; }
// IEEE-754 float32 infinity: exponent all 1s, mantissa zero => |bits| == 0x7F800000.
sfpi_inline auto is_inf(sfpi::vInt abs_bits) { return abs_bits == 0x7F800000; }

template <SfpuType Op>
inline constexpr bool is_fp32_equal_compare_v = Op == SfpuType::eq || Op == SfpuType::ne;

template <SfpuType Op>
inline constexpr bool is_fp32_ordered_compare_v =
    Op == SfpuType::lt || Op == SfpuType::gt || Op == SfpuType::le || Op == SfpuType::ge;

template <SfpuType Op>
inline constexpr bool is_fp32_compare_v = is_fp32_equal_compare_v<Op> || is_fp32_ordered_compare_v<Op>;

template <SfpuType RELATIONAL_OP, std::enable_if_t<is_fp32_equal_compare_v<RELATIONAL_OP>, int> = 0>
sfpi_inline sfpi::vFloat binary_comp_fp32_equal_mask(sfpi::vFloat in0, sfpi::vFloat in1) {
    constexpr bool is_eq = (RELATIONAL_OP == SfpuType::eq);
    const sfpi::vFloat positive = is_eq ? sfpi::vConst1 : sfpi::vConst0;
    const sfpi::vFloat negative = is_eq ? sfpi::vConst0 : sfpi::vConst1;

    sfpi::vFloat mask = negative;
    sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0);
    sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);
    sfpi::vInt in0_abs = in0_bits & 0x7FFFFFFF;
    sfpi::vInt in1_abs = in1_bits & 0x7FFFFFFF;

    // Equality (±0 / identical +inf/-inf bit patterns) then strip NaN lanes. For ne, NaN strip must
    // follow so unordered NaN lanes stay "not equal" (1) and are not confused with same-infinity eq.
    v_if(
        (in0 == in1) ||                                 // Handle equal values
        (in0_abs == 0 && in1_abs == 0) ||               // Handle ±0.0
        ((in0_bits == in1_bits) && is_inf(in0_abs))) {  // Handle +inf/-inf bit patterns
        mask = positive;
    }
    v_endif;
    v_if(is_nan(in0_abs) || is_nan(in1_abs)) { mask = negative; }
    v_endif;
    return mask;
}

// Float32 ordered comparison (lt/gt/le/ge) via SFPSWAP.
//
//   IEEE-754 fp32 bit patterns form a valid sign-magnitude int32 total order for every non-NaN
//   value (including +/-inf and +/-0 which compare equal). SFPSWAP mod VEC_MIN_MAX compares the
//   operands as sign-magnitude integers, which therefore delivers the correct IEEE ordering for
//   all non-NaN inputs in a single instruction -- with no SETCC/ENCC/COMPC predication.
//
//   Destination Index Tracking Mode is enabled so the conditional swap on LREG[0..1] is mirrored
//   onto LREG[4..5]. We seed LREG4/LREG5 with fp32 0.0 and 1.0 (via LCONST_0 / LCONST_1) so the
//   correct boolean value (as fp32) ends up in LREG4 and is stored directly. NaN lanes violate
//   IEEE (all ordered compares with NaN must return false) and are fixed up with one v_if after.
//
//   Layout per iteration (mirrors the int32 path):
//     LT:  VD=LREG0=in1, VC=LREG1=in0, LREG4=0.0f, LREG5=1.0f
//     GT:  VD=LREG0=in0, VC=LREG1=in1, LREG4=0.0f, LREG5=1.0f
//     LE:  VD=LREG0=in0, VC=LREG1=in1, LREG4=1.0f, LREG5=0.0f
//     GE:  VD=LREG0=in1, VC=LREG1=in0, LREG4=1.0f, LREG5=0.0f
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_ordered_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_float_ordered(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Enable Destination Index Tracking Mode so SFPSWAP mirrors LREG[0..3] swaps onto LREG[4..7].
    _sfpu_load_config32_(0xF, 0x0, 0x4);

    constexpr bool vd_is_in1 = (RELATIONAL_OP == SfpuType::lt) || (RELATIONAL_OP == SfpuType::ge);
    constexpr bool lreg4_is_one = (RELATIONAL_OP == SfpuType::ge) || (RELATIONAL_OP == SfpuType::le);

    constexpr uint dst_tile_size = 64;
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (vd_is_in1) {
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        } else {
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
            TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        }

        // Seed mirror registers with fp32 0.0 and 1.0 from the SFPU hardware constants.
        if constexpr (lreg4_is_one) {
            TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
        } else {
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);
            TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG5, 0);
        }

        // Conditional swap of LREG0/LREG1 (as sign-magnitude) mirrored onto LREG4/LREG5.
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

        TT_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        // NaN fixup: IEEE requires every ordered compare with a NaN operand to return false.
        // Sign-magnitude compare ranks NaNs outside +/-inf so the SFPSWAP result is arbitrary.
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vInt in0_abs = sfpi::reinterpret<sfpi::vInt>(in0) & 0x7FFFFFFF;
        sfpi::vInt in1_abs = sfpi::reinterpret<sfpi::vInt>(in1) & 0x7FFFFFFF;
        v_if(is_nan(in0_abs) || is_nan(in1_abs)) { sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = sfpi::vConst0; }
        v_endif;

        sfpi::dst_reg++;
    }

    // Restore default SFPU config.
    _init_sfpu_config_reg();
}

// Float32 equality comparison (eq/ne) -- sfpi-based; SFPSWAP doesn't compute equality directly.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_equal_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_float_equal(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = binary_comp_fp32_equal_mask<RELATIONAL_OP>(in0, in1);
        sfpi::dst_reg++;
    }
}

// Float32 binary comparisons (single entry point used by the LLK wrappers).
// Dispatches:
//   - lt/gt/le/ge -> SFPSWAP-based ordered path.
//   - eq/ne       -> sfpi-based equality path.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_float(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    if constexpr (is_fp32_ordered_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_float_ordered<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP>(
            dst_index_in0, dst_index_in1, dst_index_out);
    } else {
        calculate_binary_comp_float_equal<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_uint16(
    const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    static_assert(RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt, "Supported operation types: lt, gt");
    constexpr uint32_t dst_tile_size = 64;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (RELATIONAL_OP == SfpuType::lt) {
            // Load operand A as uint16 for lt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
            // Load operand B as uint16 for lt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
            // Load operand A as uint16 for gt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
            // Load operand B as uint16 for gt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        }
        // LREG1 = LREG0 - LREG1 = A - B (imod=6 does dst = src - dst)
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        // Extract sign bit: logical right shift by 31 -> 1 if negative (A < B), 0 otherwise
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Store the result in the destination register
        TT_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
}  //  namespace ckernel::sfpu
