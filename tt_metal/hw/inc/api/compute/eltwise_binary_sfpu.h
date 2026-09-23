// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_binary_comp.h"
#include "ckernel_sfpu_binary_pow.h"
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
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::DIV, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::MUL, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * @tparam dst_rounding_mode Controls bf16 narrowing for the result. Default truncates (native SFPSTORE
 *         behavior); NearestEven applies IEEE 754 round-to-nearest-even in software before the store.
 *         Ignored when fp32 DEST accumulation is enabled. Note: in WH and BH, mul_binary_tile and div_binary_tile
 *         apply RNE rounding when narrowing to bf16. Kept as the first template parameter for source compatibility
 *         with add_binary_tile<DstRoundingMode::...>(...).
 * @tparam is_fp32_dest_acc_en Enables fp32 DEST accumulation. Defaults to DST_ACCUM_MODE.
 */
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::ADD, DST_SYNC_MODE, is_fp32_dest_acc_en, dst_rounding_mode>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

/// @tparam dst_rounding_mode, is_fp32_dest_acc_en See add_binary_tile.
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::SUB, DST_SYNC_MODE, is_fp32_dest_acc_en, dst_rounding_mode>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

#ifndef ARCH_QUASAR
/// @tparam dst_rounding_mode, is_fp32_dest_acc_en See add_binary_tile.
template <
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::RSUB, DST_SYNC_MODE, is_fp32_dest_acc_en, dst_rounding_mode>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinaryPow<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eq_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Eq, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ne_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ne, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Lt, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Gt, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void le_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Le, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ge_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ge, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst0, idst1, odst, VectorMode::RC)));
}
#endif

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void div_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::DIV, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::MUL, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::ADD, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::SUB, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, ckernel::BinaryOp::RSUB, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void power_binary_tile_init() {
    MATH((sfpu::SfpuBinaryPow<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eq_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Eq, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ne_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ne, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lt_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Lt, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gt_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Gt, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void le_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Le, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ge_binary_tile_init() {
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ge, DataFormat::Float32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}
#endif

}  // namespace ckernel
