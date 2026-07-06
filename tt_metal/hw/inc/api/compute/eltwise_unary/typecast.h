// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_typecast.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#ifdef ARCH_QUASAR
namespace detail {
// MX / block-float typecasts are a pure unpack/pack format conversion (a datacopy) on Quasar:
// the unpacker converts MX -> float into Dest and the packer converts float -> MX on the way out,
// so no SFPU op runs. This mirrors the BH "handled by unpacker/packer" no-op bfp arms.
inline constexpr bool _typecast_is_mx_format_(DataFormat fmt) {
    return fmt == DataFormat::MxFp8R || fmt == DataFormat::MxFp8P || fmt == DataFormat::MxFp6R ||
           fmt == DataFormat::MxFp6P || fmt == DataFormat::MxFp4 || fmt == DataFormat::MxInt8 ||
           fmt == DataFormat::MxInt4 || fmt == DataFormat::MxInt2;
}
}  // namespace detail
#endif

// clang-format off
/**
 * Performs an elementwise typecast operation on the input.
 * Supports following typecasts:
 *  Float16_b <-> Float32
 *  Float16_b <-> Int32
 *  Float16_b <-> UInt16
 *  Float16_b <-> UInt32
 *  Float16_b <-> UInt8
 *  Float32 <-> Int32
 *  Float32 <-> UInt16
 *  Float32 <-> UInt32
 *  Float32 <-> UInt8
 *  Bfp8_b <-> Int32
 *  Bfp8_b <-> UInt16
 *  Bfp8_b <-> UInt32
 *  Bfp8_b <-> UInt8
 *  Bfp8_b <-> Float16_b
 *  Bfp8_b <-> Float32
 *  Bfp4_b <-> Int32
 *  Bfp4_b <-> UInt16
 *  Bfp4_b <-> UInt32
 *  Bfp4_b <-> UInt8
 *  Bfp4_b <-> Bfp8_b
 *  Bfp4_b <-> Float16_b
 *  Bfp4_b <-> Float32
 *  UInt16 <-> UInt32
 *  UInt16 <-> Int32
 *  UInt16 <-> UInt8
 *
 * For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | IN_DTYPE       | Input data format                                                          | uint32_t | Must be valid tt::DataFormat                          | True     |
 * | OUT_DTYPE      | Desired output data format                                                 | uint32_t | Must be valid tt::DataFormat                          | True     |
 */
// clang-format on
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile(uint32_t idst) {
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);

#ifdef ARCH_QUASAR
    // An MX endpoint is unpacked to / packed from Float16_b by the format, so at the SFPU level an MX
    // format behaves as Float16_b. Route through that effective format: MX <-> Float16_b (and MX <-> MX)
    // collapse to a pure format no-op, while MX <-> {Float32, Int32, ...} run the Float16_b <-> X SFPU
    // conversion on top of the format (X -> MX runs X -> Float16_b, then the packer emits MX).
    constexpr DataFormat effective_input_format =
        detail::_typecast_is_mx_format_(in_format) ? DataFormat::Float16_b : in_format;
    constexpr DataFormat effective_output_format =
        detail::_typecast_is_mx_format_(out_format) ? DataFormat::Float16_b : out_format;
    if constexpr (effective_input_format != effective_output_format) {
        // Single unified Quasar typecast kernel, templated on the effective source/destination formats.
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast,
            (effective_input_format, effective_output_format, SFPU_ITERATIONS),
            idst,
            VectorMode::RC));
    }
#else
    if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint16,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Float16_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::Float16_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_int32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::Int32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_int32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::Float32) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Float16_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint16,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_fp32,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Int32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_int32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_int32_to_fp32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint16,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Bfp8_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Int32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_int32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::Bfp8_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_int32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::UInt32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::Float16_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::UInt32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::UInt32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::Bfp8_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::UInt32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_uint32,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Int32) {
        // Calls same kernel as UInt32 case
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_uint32,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_uint16,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_int32_to_uint16,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Float16_b) {
        // no SFPU kernel needed, handled by unpacker
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::Bfp8_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Float32) {
        // no SFPU kernel needed, handled by unpacker/packer
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Bfp8_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint16,
            (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Bfp4_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint16_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::Int32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_int32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::Bfp4_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_int32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::UInt32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::Bfp4_b) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::Float16_b) {
        // no SFPU kernel needed, handled by unpacker
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::Bfp4_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::Bfp8_b) {
        // no SFPU kernel needed, handled by unpacker
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Bfp4_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::Float32) {
        // no SFPU kernel needed, handled by unpacker/packer
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Bfp4_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (
        (in_format == DataFormat::Float32 || in_format == DataFormat::Float16_b || in_format == DataFormat::Bfp8_b ||
         in_format == DataFormat::Bfp4_b) &&
        out_format == DataFormat::UInt8) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_fp32_to_uint8,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (
        (in_format == DataFormat::Int32 || in_format == DataFormat::UInt32 || in_format == DataFormat::UInt16) &&
        out_format == DataFormat::UInt8) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint_to_uint8,
            (APPROX, 8 /* ITERATIONS */, (in_format == DataFormat::UInt16)),
            idst,
            VectorMode::RC));
    } else if constexpr (in_format == DataFormat::UInt8 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp32,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (
        in_format == DataFormat::UInt8 &&
        (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp4_b)) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_fp16b,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    } else if constexpr (
        in_format == DataFormat::UInt8 && (out_format == DataFormat::Int32 || out_format == DataFormat::UInt32)) {
        // No SFPU kernel needed.
    } else if constexpr (in_format == DataFormat::UInt8 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_typecast_uint32_to_uint16,
            (APPROX, 8 /* ITERATIONS */),
            idst,
            VectorMode::RC));
    }
#endif  // ARCH_QUASAR
}

/**
 * Please refer to documentation for any_init.
 */
template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
ALWI void typecast_tile_init() {
    constexpr DataFormat in_format = static_cast<DataFormat>(IN_DTYPE);
    constexpr DataFormat out_format = static_cast<DataFormat>(OUT_DTYPE);

#ifdef ARCH_QUASAR
    // Mirror typecast_tile: an MX endpoint behaves as Float16_b at the SFPU level, so only a
    // non-trivial effective conversion needs the SFPU init (MX <-> Float16_b is a format no-op).
    constexpr DataFormat effective_input_format =
        detail::_typecast_is_mx_format_(in_format) ? DataFormat::Float16_b : in_format;
    constexpr DataFormat effective_output_format =
        detail::_typecast_is_mx_format_(out_format) ? DataFormat::Float16_b : out_format;
    if constexpr (effective_input_format != effective_output_format) {
        MATH(SFPU_UNARY_INIT(typecast, sfpu::init_typecast));
    }
#else
    if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Float16_b) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_fp16b, (APPROX)));
    } else if constexpr (
        in_format == DataFormat::UInt16 && (out_format == DataFormat::UInt32 || out_format == DataFormat::Int32)) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_uint32, (APPROX)));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX)));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_uint16, (APPROX)));
    } else if constexpr (in_format == DataFormat::UInt32 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX)));
    } else if constexpr (in_format == DataFormat::Int32 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp32, (APPROX)));
    } else if constexpr (in_format == DataFormat::UInt16 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp32, (APPROX)));
    } else if constexpr (
        in_format == DataFormat::UInt16 &&
        (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp4_b)) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp16b, (APPROX)));
    } else if constexpr (
        in_format == DataFormat::Int32 &&
        (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp4_b)) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp16b, (APPROX)));
    } else if constexpr (
        in_format == DataFormat::UInt32 &&
        (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp4_b)) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX)));
    } else if constexpr (
        (in_format == DataFormat::Float32 || in_format == DataFormat::Float16_b || in_format == DataFormat::Bfp8_b ||
         in_format == DataFormat::Bfp4_b) &&
        out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint16, (APPROX)));
    } else if constexpr (
        (in_format == DataFormat::Float32 || in_format == DataFormat::Float16_b || in_format == DataFormat::Bfp8_b ||
         in_format == DataFormat::Bfp4_b) &&
        out_format == DataFormat::UInt8) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint8, (APPROX)));
    } else if constexpr (
        (in_format == DataFormat::Int32 || in_format == DataFormat::UInt32 || in_format == DataFormat::UInt16) &&
        out_format == DataFormat::UInt8) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint_to_uint8, (APPROX)));
    } else if constexpr (in_format == DataFormat::UInt8 && out_format == DataFormat::Float32) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX)));
    } else if constexpr (
        in_format == DataFormat::UInt8 &&
        (out_format == DataFormat::Float16_b || out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp4_b)) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX)));
    } else if constexpr (in_format == DataFormat::UInt8 && out_format == DataFormat::UInt16) {
        MATH(SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX)));
    } else {
        MATH(SFPU_UNARY_INIT(typecast));
    }
#endif  // ARCH_QUASAR
}

}  // namespace ckernel
