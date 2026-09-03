// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#ifndef ARCH_QUASAR
#ifdef TRISC_MATH
#include "ckernel_sfpu_reduce.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs reduce operation (sum, average, max, min) on a 32x32 tile for column reduction and multiple tiles for row reduction, placing output values into the first row.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Only 32x32 tile dimensions are supported
 *  - This kernel is optimized for 32x32 tile dimensions and uses VectorMode::RC_custom for customized reduction
 *  - Column reduction (REDUCE_COL) is supported for all pool types; row reduction (REDUCE_ROW) is supported for SUM, MAX and MIN only.
 *  - REDUCE_COL operates on a single tile only (ct_dim = 1, rt_dim = 1).
 *  - REDUCE_ROW supports multiple tiles: ct_dim and rt_dim specify the tile block dimensions to reduce over.
 *
 * | Argument        | Description                                                                     | Type      | Valid Range
 * |-----------------|---------------------------------------------------------------------------------|-----------|-------------------------------------------------------
 * | pool_type       | The type of reduction operation                                                 | PoolType  | SUM, AVG, MAX, MIN
 * | format          | The data format for the reduction operation                                     | DataFormat| Float32, Int32, UInt32, UInt16, Float16_b
 * | reduce_dim      | The reduction dimension                                                         | ReduceDim | REDUCE_COL or REDUCE_ROW (REDUCE_ROW only for SUM, MAX and MIN)
 * | idst            | The index of the tile in DST register containing the data to be reduced         | uint32_t  | Must be less than the size of the DST register buffer
 * | ct_dim          | Tile dimension along columns (runtime); must be 1 when reduce_dim is REDUCE_COL | uint32_t  | >= 1; default 1
 * | rt_dim          | Tile dimension along rows (runtime); must be 1 when reduce_dim is REDUCE_COL    | uint32_t  | >= 1; default 1
 */
// clang-format on
template <
    PoolType pool_type,
    DataFormat format,
    ReduceDim reduce_dim = ReduceDim::REDUCE_COL,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_reduce(uint32_t idst, uint32_t ct_dim = 1, uint32_t rt_dim = 1) {
    static_assert(
        reduce_dim == ReduceDim::REDUCE_COL ||
            (reduce_dim == ReduceDim::REDUCE_ROW &&
             (pool_type == PoolType::SUM || pool_type == PoolType::MAX || pool_type == PoolType::MIN)),
        "Only column reduction (REDUCE_COL) is supported for all pool types; row reduction (REDUCE_ROW) is only "
        "supported for SUM, MAX and MIN");
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32 ||
            format == DataFormat::UInt16 || format == DataFormat::Float16_b,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32, UInt16, Float16_b");
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool type. Supported pool types: SUM, AVG, MAX, MIN");

    // This kernel is optimized for 32x32 tiles and uses RC_custom vector mode for custom reduction
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_reduce,
        (pool_type, reduce_dim, format, is_fp32_dest_acc_en),
        idst,
        VectorMode::RC_custom,
        ct_dim,
        rt_dim));
}

/**
 * @brief Initialization for SFPU reduce kernel.
 *        Must be called before sfpu_reduce() to set up the necessary configurations for reduction operations.
 *        The same init is used for both REDUCE_COL and REDUCE_ROW; it does not take tile dimensions.
 * @tparam pool_type The reduction operation, currently supported: (SUM, AVG, MAX, MIN)
 * @tparam format The data format, currently supported: (Float32, Int32, UInt32, UInt16, Float16_b)
 */
template <PoolType pool_type, DataFormat format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_reduce_init() {
    static_assert(
        pool_type == PoolType::SUM || pool_type == PoolType::AVG || pool_type == PoolType::MAX ||
            pool_type == PoolType::MIN,
        "Unsupported pool type. Supported pool types: SUM, AVG, MAX, MIN");
    static_assert(
        format == DataFormat::Float32 || format == DataFormat::Int32 || format == DataFormat::UInt32 ||
            format == DataFormat::UInt16 || format == DataFormat::Float16_b,
        "Unsupported data format. Supported formats: Float32, Int32, UInt32, UInt16, Float16_b");

    MATH(SFPU_UNARY_INIT_FN_ARGS(
        reduce, sfpu::init_reduce, (pool_type, format, is_fp32_dest_acc_en), 1 /* block_ct_dim */));
}

}  // namespace ckernel

#endif  // !ARCH_QUASAR
