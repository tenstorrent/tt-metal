// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logical_not.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
// clang-format off
/**
 * Performs element-wise computation of the logical not operation on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp4_b.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat DATA_FORMAT>
ALWI void logical_not_tile(uint32_t idst) {
    static_assert(
        DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||
            DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 ||
            DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::Bfp8_b || DATA_FORMAT == DataFormat::Bfp4_b,
        "Unsupported data format for logical_not_tile. Supported data formats are: Float32, Float16_b, Int32, UInt32, "
        "UInt16, Bfp8_b, Bfp4_b.");
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||
         DATA_FORMAT == DataFormat::Bfp8_b || DATA_FORMAT == DataFormat::Bfp4_b)
            ? InstrModLoadStore::DEFAULT
        : (DATA_FORMAT == DataFormat::UInt16)                                     ? InstrModLoadStore::LO16
        : (DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32) ? InstrModLoadStore::INT32
                                                                                  : InstrModLoadStore::DEFAULT;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_logical_not,
        (APPROX, INSTRUCTION_MODE, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void logical_not_tile_init() { MATH(SFPU_UNARY_INIT(logical_not_unary)); }

}  // namespace ckernel
