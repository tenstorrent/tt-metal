// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary_comp.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise comparison operation with two integer inputs: y = comparison_op(x0,x1)
 * Supports Int32, UInt32 and UInt16 data formats (selected via the data_format template parameter).
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Template Param | Description                                                           | Valid Values                             | Required |
 * |----------------|-----------------------------------------------------------------------|------------------------------------------|----------|
 * | data_format    | Data format of the integer operands                                   | DataFormat::Int32/UInt32/UInt16          | True     |
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on

#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eq_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Eq, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
}

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ne_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ne, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
}
#endif

#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Lt, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
}
#endif

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#if defined(ARCH_QUASAR)
    // Int8 copy_tile + fp32_dest_acc FPU writes sign-magnitude Int32 into dest.
    // Native Int32 tiles use 2's-comp dest and keep SIGN_MAGNITUDE_FORMAT=false.
    MATH((sfpu::BinaryComp<
          APPROX,
          sfpu::BinaryCompMode::Gt,
          data_format,
          DST_SYNC_MODE,
          is_fp32_dest_acc_en,
          true /*SIGN_MAGNITUDE_FORMAT*/>::calculate(idst0, idst1, odst, VectorMode::RC)));
#else
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Gt, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
#endif
}

#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void le_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Le, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
}

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ge_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    MATH(
        (sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ge, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst0, idst1, odst, VectorMode::RC)));
}
#endif

/**
 * The following functions initialize the relational operations. They should be invoked prior to calling the execution
 * API. Please refer to execution API documentation to find out more about the relational operations.
 */
#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eq_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for eq_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Eq, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ne_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for ne_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ne, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif

#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lt_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Lt, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gt_int_tile_init() {
#if defined(ARCH_QUASAR)
    static_assert(data_format == DataFormat::Int32, "Unsupported data format for gt_int on Quasar. Supported: Int32");
#else
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for gt_int. Supported data formats are: Int32, UInt32, UInt16");
#endif
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Gt, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

#ifndef ARCH_QUASAR
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void le_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Le, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ge_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((sfpu::BinaryComp<APPROX, sfpu::BinaryCompMode::Ge, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif

}  // namespace ckernel
