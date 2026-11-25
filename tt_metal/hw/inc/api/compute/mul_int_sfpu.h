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
 * Performs an elementwise fixed-point multiply of two integer inputs: y = sign * ((|x0| * |x1|) >> 23).
 * Output overwrites odst in DST.
 *
 * x1 is a Q0.23 fixed-point scale, so y is x0 rescaled by x1 / 2^23, rounded toward zero. This is
 * the quantization requant primitive; it is not mul_int_tile followed by a shift, which would drop
 * the high bits of the product that this operation returns.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine. Quasar only.
 *
 * The result is exact whenever |y| < 2^23. Outside that range neither the multiply nor the
 * accumulation saturates. A zero result is +0.
 *
 * @tparam data_format Template argument specifying the data type. Supported data formats are: DataFormat::Int32
 * @tparam sign_magnitude_format Encoding of the operands and result in DST: false for two's complement
 * (Int32 tiles unpacked to DST), true for sign-magnitude (Int8 promoted by the FPU). Two's-complement
 * INT32_MIN is not supported.
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
template <DataFormat data_format, bool sign_magnitude_format = false>
ALWI void mul_int_tile_sm_and_shift23(uint32_t idst0, uint32_t idst1, uint32_t odst) {
#if defined(ARCH_QUASAR)
    static_assert(
        data_format == DataFormat::Int32,
        "Unsupported data format for mul_int_tile_sm_and_shift23 on Quasar. Supported: Int32");
    MATH((llk_math_eltwise_binary_sfpu_mul_int_sm_shift23<APPROX, data_format, 8 /*ITERATIONS*/, sign_magnitude_format>(
        idst0, idst1, odst)));
#else
    static_assert(data_format == DataFormat::Invalid, "mul_int_tile_sm_and_shift23 is implemented on Quasar only");
#endif
}

/**
 * Please refer to documentation for mul_int_tile_sm_and_shift23.
 */
template <DataFormat data_format>
ALWI void mul_int_tile_sm_and_shift23_init() {
#if defined(ARCH_QUASAR)
    static_assert(
        data_format == DataFormat::Int32,
        "Unsupported data format for mul_int_tile_sm_and_shift23 on Quasar. Supported: Int32");
    MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, data_format>()));
#else
    static_assert(data_format == DataFormat::Invalid, "mul_int_tile_sm_and_shift23 is implemented on Quasar only");
#endif
}

}  // namespace ckernel
