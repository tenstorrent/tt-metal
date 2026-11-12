// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "experimental/ckernel_sfpu_fill.h"
#else
#include "sfpu/ckernel_sfpu_fill.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Value to fill tile with.                                                   | float    |                                                       | True     |
 */
// clang-format on
ALWI void fill_tile(uint32_t idst, float param0) {
#if defined(ARCH_QUASAR)
    // Quasar's _calculate_fill_ is templated on ITERATIONS only; it takes no APPROX parameter.
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
#endif
}

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Value to fill tile with (unsigned integer)                                 | uint32_t |                                                       | True     |
 */
template <DataFormat DATA_FORMAT>
ALWI void fill_tile_int(uint32_t idst, uint32_t param0) {
#if defined(ARCH_QUASAR)
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::Int16 || DATA_FORMAT == DataFormat::Int8 ||
            DATA_FORMAT == DataFormat::UInt8,
        "Unsupported data format for fill_tile_int on Quasar. Supported: Int32, Int16, Int8, UInt8");
    // Quasar picks the SFPMEM store mode from the DataFormat itself; it has neither
    // DataFormat::UInt32 nor the InstrModLoadStore enum used on the Tensix-1xx path.
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_int_, (DATA_FORMAT, 8 /*ITERATIONS*/), idst, VectorMode::RC,
        param0));
#else
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for fill_tile_int. Supported: Int32, UInt32, UInt16");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_int_,
        (APPROX, INSTRUCTION_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
#endif
}

// clang-format off
/**
 * Performs element-wise fill operation. The value to be filled in the tile is provided as const param0, which is
 * interpreted as a bit-cast representation of a floating-point value. The DST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The bit-cast representation of a floating-point value to be used as output | uint32_t | Must represent a valid bit-cast float value           | True     |
 */
// clang-format on
ALWI void fill_tile_bitcast(uint32_t idst, uint32_t param0) {
#if defined(ARCH_QUASAR)
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_bitcast_, (8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_bitcast_,
        (APPROX, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
#endif
}
/**
 * Please refer to documentation for any_init.
 */
ALWI void fill_tile_init() { MATH(SFPU_UNARY_INIT(fill)); }

}  // namespace ckernel
