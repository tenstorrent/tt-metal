// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_binary_pow.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#else
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_binary_comp.h"
#include "ckernel_sfpu_binary_pow.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise binop operation with the two floating point inputs: y = binop(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 *
 * add_binary_tile, sub_binary_tile and mul_binary_tile also take the extent of the tile they touch:
 *
 * | Argument       | Description                                                           | Type       | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | ITERATIONS     | Template: 4-row groups per visited face. Iteration k writes face rows 4*(k/2)..4*(k/2)+3 of one column parity (even columns for even k, odd for odd k), so 8 is the whole face and 2 under VectorMode::R is rows 0..3 of all 32 columns. The body advances the DEST counter per iteration and the vector-mode loop re-bases every face on the CR counter, so a partial count composes with any vector mode. Fixed at 8 on Quasar. | int | 1 to 8 | False (8) |
 * | vector_mode    | The faces to visit: RC all four, R faces 0 and 1 (rows 0..15), C faces 0 and 2 (columns 0..15) | VectorMode | RC, R, C | False (RC) |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#ifdef ARCH_QUASAR
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::DIV, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary_div,
        (APPROX, ckernel::BinaryOp::DIV, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
#endif
}

/**
 * @tparam is_fp32_dest_acc_en Enables fp32 DEST accumulation. Defaults to DST_ACCUM_MODE.
 * @tparam ITERATIONS, vector_mode The extent of the tile the op touches; see the table above and add_binary_tile.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, int ITERATIONS = 8>
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    static_assert(ITERATIONS == 8, "mul_binary_tile: ITERATIONS is fixed at 8 on Quasar");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::MUL, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, ITERATIONS, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        vector_mode)));
#endif
}

/**
 * @tparam dst_rounding_mode Controls bf16 narrowing for the result. Default truncates (native SFPSTORE
 *         behavior); NearestEven applies IEEE 754 round-to-nearest-even in software before the store.
 *         Ignored when fp32 DEST accumulation is enabled. Note: in WH and BH, mul_binary_tile and div_binary_tile
 *         apply RNE rounding when narrowing to bf16. Kept as the first template parameter for source compatibility
 *         with add_binary_tile<DstRoundingMode::...>(...).
 * @tparam is_fp32_dest_acc_en Enables fp32 DEST accumulation. Defaults to DST_ACCUM_MODE.
 * @tparam ITERATIONS 4-row groups per visited face (1..8, default 8 = the whole face): iteration k writes face rows
 *         4*(k/2)..4*(k/2)+3 of one column parity (even columns for even k, odd for odd k), so 2 under VectorMode::R
 *         is rows 0..3 of all 32 columns. The body advances the DEST counter per iteration and the vector-mode loop
 *         re-bases every face on the CR counter, so a partial count composes with any vector mode. Fixed at 8 on
 *         Quasar.
 * @param vector_mode The faces to visit: RC (default) all four, R faces 0 and 1, C faces 0 and 2. The elements
 *        outside the visited faces and iterations keep their contents.
 */
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    int ITERATIONS = 8>
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    static_assert(ITERATIONS == 8, "add_binary_tile: ITERATIONS is fixed at 8 on Quasar");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, is_fp32_dest_acc_en, dst_rounding_mode),
        idst0,
        idst1,
        odst,
        vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, ITERATIONS, is_fp32_dest_acc_en, dst_rounding_mode),
        idst0,
        idst1,
        odst,
        vector_mode)));
#endif
}

/// @tparam dst_rounding_mode, is_fp32_dest_acc_en, ITERATIONS, vector_mode See add_binary_tile.
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    int ITERATIONS = 8>
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    static_assert(ITERATIONS == 8, "sub_binary_tile: ITERATIONS is fixed at 8 on Quasar");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, is_fp32_dest_acc_en, dst_rounding_mode),
        idst0,
        idst1,
        odst,
        vector_mode)));
#else
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, ITERATIONS, is_fp32_dest_acc_en, dst_rounding_mode),
        idst0,
        idst1,
        odst,
        vector_mode)));
#endif
}

#ifndef ARCH_QUASAR
/// @tparam dst_rounding_mode, is_fp32_dest_acc_en See add_binary_tile.
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::RSUB, 8 /* ITERATIONS */, is_fp32_dest_acc_en, dst_rounding_mode),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}
#endif  // !ARCH_QUASAR

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary_pow,
        (APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

#ifndef ARCH_QUASAR

ALWI void eq_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::eq),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void ne_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::ne),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void lt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::lt),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void gt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::gt),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void le_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::le),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void ge_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8 /* ITERATIONS */, SfpuType::ge),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}
#endif

/**
 * Please refer to documentation for any_init.
 */
ALWI void div_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::DIV))));
}

ALWI void mul_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
}

ALWI void add_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::ADD))));
}

ALWI void sub_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::SUB))));
}

#ifndef ARCH_QUASAR
ALWI void rsub_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::RSUB))));
}

#endif  // !ARCH_QUASAR

ALWI void power_binary_tile_init() { MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_pow_init, (APPROX)))); }

#ifndef ARCH_QUASAR

ALWI void eq_binary_tile_init() { MATH((SFPU_BINARY_INIT(eq))); }

ALWI void ne_binary_tile_init() { MATH((SFPU_BINARY_INIT(ne))); }

ALWI void lt_binary_tile_init() { MATH((SFPU_BINARY_INIT(lt))); }

ALWI void gt_binary_tile_init() { MATH((SFPU_BINARY_INIT(gt))); }

ALWI void le_binary_tile_init() { MATH((SFPU_BINARY_INIT(le))); }

ALWI void ge_binary_tile_init() { MATH((SFPU_BINARY_INIT(ge))); }
#endif

}  // namespace ckernel
