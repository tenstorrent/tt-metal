// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifndef ARCH_QUASAR
#include "sfpu/ckernel_sfpu_fill.h"
#else
#include "llk_sfpu/ckernel_sfpu_fill.h"
#endif
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
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_fill,
        (DataFormat::Float32, DST_ACCUM_MODE, SFPU_ITERATIONS),
        idst,
        VectorMode::RC,
        param0));
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
#ifndef ARCH_QUASAR
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
#else
    // Quasar's int fill path (_calculate_fill_int_ in ckernel_sfpu_fill.h) only supports
    // Int32/Int16/Int8/UInt8 — UInt32 and UInt16 (valid on WH/BH above) are not yet supported here.
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::Int16 || DATA_FORMAT == DataFormat::Int8 ||
            DATA_FORMAT == DataFormat::UInt8,
        "Unsupported data format for fill_tile_int on Quasar. Supported: Int32, Int16, Int8, UInt8");
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_fill,
        (DATA_FORMAT, DST_ACCUM_MODE, SFPU_ITERATIONS),
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
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_fill_bitcast_,
        (APPROX, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _fill_store_,
        (p_sfpu::sfpmem::DEFAULT, SFPU_ITERATIONS),
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
