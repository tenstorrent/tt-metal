// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "llk_math_eltwise_binary_sfpu_binary_comp.h"
#else
#include "ckernel_sfpu_binary_comp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
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

#if defined(TRISC_MATH) && !defined(ARCH_QUASAR)
namespace detail {

// Dispatches the integer relational compare to the appropriate ckernel functor based on the
// runtime DataFormat. This was previously the body of `llk_math_eltwise_binary_sfpu_rel_int_impl`
// in the (now-deleted) BH wrapper.
//
// Guarded by TRISC_MATH because the template signature references SfpuType, which is only
// brought into scope on the math thread. All callers wrap the invocation in MATH((...)) so the
// function is never reached on unpack/pack threads.
template <SfpuType OP, DataFormat data_format>
ALWI void rel_int_tile_dispatch(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::Int32) {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_comp_int32,
            (APPROX, 8 /* ITERATIONS */, OP),
            idst0,
            idst1,
            odst,
            VectorMode::RC);
    } else {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_comp_uint,
            (APPROX, 8 /* ITERATIONS */, OP, data_format),
            idst0,
            idst1,
            odst,
            VectorMode::RC);
    }
}

}  // namespace detail
#endif

#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void lt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((detail::rel_int_tile_dispatch<SfpuType::lt, data_format>(idst0, idst1, odst)));
}
#endif

template <DataFormat data_format>
ALWI void gt_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#if defined(ARCH_QUASAR)
    // Int8 copy_tile + fp32_dest_acc FPU writes sign-magnitude Int32 into dest.
    // Native Int32 tiles use 2's-comp dest and keep SIGN_MAGNITUDE_FORMAT=false.
    MATH((llk_math_eltwise_binary_sfpu_gt_int<APPROX, data_format, 8 /*ITERATIONS*/, true /*SIGN_MAGNITUDE_FORMAT*/>(
        idst0, idst1, odst)));
#else
    MATH((detail::rel_int_tile_dispatch<SfpuType::gt, data_format>(idst0, idst1, odst)));
#endif
}

#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void le_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((detail::rel_int_tile_dispatch<SfpuType::le, data_format>(idst0, idst1, odst)));
}

template <DataFormat data_format>
ALWI void ge_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((detail::rel_int_tile_dispatch<SfpuType::ge, data_format>(idst0, idst1, odst)));
}
#endif

/**
 * The following functions initialize the relational operations. They should be invoked prior to calling the execution
 * API. Please refer to execution API documentation (lt_int_tile/gt_int_tile/le_int_tile/ge_int_tile) to find out more
 * about the relational operations.
 */
#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void lt_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((SFPU_BINARY_INIT(lt_int)));
}
#endif

template <DataFormat data_format>
ALWI void gt_int_tile_init() {
#if defined(ARCH_QUASAR)
    MATH((llk_math_eltwise_binary_sfpu_gt_int_init<APPROX, data_format>()));
#else
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for gt_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((SFPU_BINARY_INIT(gt_int)));
#endif
}

#ifndef ARCH_QUASAR
template <DataFormat data_format>
ALWI void le_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((SFPU_BINARY_INIT(le_int)));
}

template <DataFormat data_format>
ALWI void ge_int_tile_init() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    MATH((SFPU_BINARY_INIT(ge_int)));
}
#endif

}  // namespace ckernel
