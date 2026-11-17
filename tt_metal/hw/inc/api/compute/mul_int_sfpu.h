// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifdef ARCH_QUASAR
#include "llk_math_eltwise_binary_sfpu_mul_int.h"
#else
#include "ckernel_sfpu_mul_int32.h"
#include "ckernel_sfpu_mul_int.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise mul operation with the two integer inputs: y = mul(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void mul_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#if defined(ARCH_QUASAR)
    static_assert(data_format == DataFormat::Int32, "Unsupported data format for mul_int on Quasar. Supported: Int32");
    // Int8 copy_tile + fp32_dest_acc FPU writes sign-magnitude Int32 into dest.
    // Native Int32 tiles use 2's-comp dest and keep SIGN_MAGNITUDE_FORMAT=false.
    MATH((llk_math_eltwise_binary_sfpu_mul_int<APPROX, data_format, 8 /*ITERATIONS*/, true /*SIGN_MAGNITUDE_FORMAT*/>(
        idst0, idst1, odst)));
#else
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::UInt16) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _mul_int_,
            (APPROX, 8 /* ITERATIONS */),
            idst0,
            idst1,
            odst,
            VectorMode::RC)));
    } else {
        MATH(
            (SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mul_int32, (APPROX), idst0, idst1, odst, VectorMode::RC)));
    }
#endif
}

/**
 * Please refer to documentation for mul_int_tile.
 */
template <DataFormat data_format>
ALWI void mul_int_tile_init() {
#if defined(ARCH_QUASAR)
    static_assert(data_format == DataFormat::Int32, "Unsupported data format for mul_int on Quasar. Supported: Int32");
    MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, data_format>()));
#else
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::UInt16) {
        MATH((SFPU_BINARY_INIT_FN(mul_uint16, sfpu::_init_mul_int_, (APPROX))));
    } else {
        MATH((SFPU_BINARY_INIT_FN(mul_int32, sfpu::mul_int32_init, (APPROX))));
    }
#endif
}

// clang-format off
/**
 * Performs an elementwise sign-magnitude fixed-point multiply of two integer inputs:
 * y = sign * ((|x0| * |x1|) >> 23). Output overwrites odst in DST.
 *
 * x1 is a Q0.23 fixed-point scale, so y is x0 rescaled by x1 / 2^23. This is the quantization
 * requant primitive. It is not mul_int_tile followed by a shift: a full multiply keeps the low
 * 32 bits of the product and discards exactly the bits this operation returns.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine. Quasar only.
 *
 * Operands are sign-magnitude Int32, Quasar's native integer DST layout. INT32_MIN has no
 * sign-magnitude encoding and is not supported. The result is exact when each retained cross
 * product fits in 23 bits and at most one operand reaches 2^23; for requant that holds whenever
 * the result itself fits in 23 bits. Neither the multiply nor the accumulation saturates.
 *
 * The ensure_* flags are preconditions the caller asserts, each dropping instructions: a
 * guaranteed-non-negative operand skips its sign handling, and an operand guaranteed below 2^23
 * skips one cross product. Their order matches the Blackhole kernel's template parameters.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <
    bool ensure_pos_a = false,
    bool ensure_pos_b = false,
    bool ensure_a_23bit = false,
    bool ensure_b_23bit = false,
    DataFormat data_format = DataFormat::Int32>
ALWI void mul_int_tile_sm_and_shift23(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#if defined(ARCH_QUASAR)
    static_assert(
        data_format == DataFormat::Int32,
        "Unsupported data format for mul_int_sm_and_shift23 on Quasar. Supported: Int32");
    MATH((llk_math_eltwise_binary_sfpu_mul_int_sm_shift23<
          APPROX,
          data_format,
          8 /*ITERATIONS*/,
          true /*SIGN_MAGNITUDE_FORMAT*/,
          ensure_pos_a,
          ensure_pos_b,
          ensure_a_23bit,
          ensure_b_23bit>(idst0, idst1, odst)));
#else
    static_assert(data_format == DataFormat::Invalid, "mul_int_tile_sm_and_shift23 is implemented on Quasar only");
#endif
}

/**
 * Please refer to documentation for mul_int_tile_sm_and_shift23.
 *
 * The ensure_* flags must match the ones passed to mul_int_tile_sm_and_shift23: they decide whether
 * the SFPLOADMACRO sequence that the fully-constrained shape runs on gets programmed. The argument
 * order is the same, so the same template argument list works for both calls.
 */
template <
    bool ensure_pos_a = false,
    bool ensure_pos_b = false,
    bool ensure_a_23bit = false,
    bool ensure_b_23bit = false,
    DataFormat data_format = DataFormat::Int32>
ALWI void mul_int_tile_sm_and_shift23_init() {
#if defined(ARCH_QUASAR)
    static_assert(
        data_format == DataFormat::Int32,
        "Unsupported data format for mul_int_sm_and_shift23 on Quasar. Supported: Int32");
    MATH((llk_math_eltwise_binary_sfpu_mul_int_sm_shift23_init<
          APPROX,
          data_format,
          true /*SIGN_MAGNITUDE_FORMAT*/,
          ensure_pos_a,
          ensure_pos_b,
          ensure_a_23bit,
          ensure_b_23bit>()));
#else
    static_assert(data_format == DataFormat::Invalid, "mul_int_tile_sm_and_shift23 is implemented on Quasar only");
#endif
}

}  // namespace ckernel
