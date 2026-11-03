// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Compile-time format classification for the typecast datapath. In the SFPU register file a
// float loads to fp32 and an int loads to sign-magnitude int32, so the conversion sequence is
// picked from these classes, not from individual formats.
template <DataFormat FMT>
inline constexpr bool _typecast_is_fp16_() {
    return FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_float_() {
    return _typecast_is_fp16_<FMT>() || FMT == DataFormat::Float32 || FMT == DataFormat::Tf32;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_int8_() {
    return FMT == DataFormat::UInt8 || FMT == DataFormat::Int8;
}

template <DataFormat FMT>
inline constexpr bool _typecast_is_unsigned_int_() {
    return FMT == DataFormat::UInt8 || FMT == DataFormat::UInt16;
}

/**
 * @brief Program the address mode the typecast op walks Dest with.
 *
 * Sets ADDR_MOD_6 to post-increment Dest by one SFPU pass (Quasar writes SFP_ROWS = 2 rows per
 * pass), so the per-pass store advances Dest and the execute loop needs no separate increment.
 *
 * @note Call before @ref calculate_typecast, whose store walks Dest through ADDR_MOD_6.
 */
inline void init_typecast() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::math::SFP_ROWS},
    }
        .set(ADDR_MOD_6);
}

// Load one SFPU pass worth of rows from Dest as VTYPE (vFloat for floats, vSMag/vInt for ints —
// ints load as sign-magnitude int32, and unsigned sources zero-extend so the sign-mag view stays
// non-negative).
template <typename VTYPE, DataFormat FMT>
inline VTYPE _typecast_load_() {
    if constexpr (
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
        FMT == DataFormat::Tf32) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::Float16 ? sfpi::DataLayout::F16a
                                    : FMT == DataFormat::Float16_b ? sfpi::DataLayout::F16b
                                    : FMT == DataFormat::Float32 || FMT == DataFormat::Tf32
                                    ? sfpi::DataLayout::F32 : sfpi::DataLayout::None;
        return sfpi::as<VTYPE>(sfpi::vFloat(sfpi::dst_reg[0].mode<layout>()));
    } else if constexpr (FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::Int32 ? sfpi::DataLayout::SM32
                                    : FMT == DataFormat::Int16 ? sfpi::DataLayout::SM16
                                    : FMT == DataFormat::Int8  ? sfpi::DataLayout::SM8
            : sfpi::DataLayout::None;
        return sfpi::as<VTYPE>(sfpi::vSMag(sfpi::dst_reg[0].mode<layout>()));
    } else if constexpr (FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::UInt16 ? sfpi::DataLayout::U16
                                    : FMT == DataFormat::UInt8 ? sfpi::DataLayout::U8
                                    : sfpi::DataLayout::None;
        return sfpi::as<VTYPE>(sfpi::vUInt(sfpi::dst_reg[0].mode<layout>()));
    } else {
        // No dedicated SFPU mode (fp8, MX block formats, 4-bit ints): fall back to the implied/
        // default register-file format. Matches the runtime overload's default case.
        return sfpi::as<VTYPE>(sfpi::vFloat(sfpi::dst_reg[0].mode<sfpi::DataLayout::FSrcB>()));
    }
}

// ADDR_MOD_6 post-increments Dest by one SFPU pass (SFP_ROWS), so the store both writes the result
// and advances to the next pair of rows — replacing the per-iteration _incr_counters_. Requires
// init_typecast to have programmed ADDR_MOD_6.
template <DataFormat FMT, typename TYPE>
inline void _typecast_store_(TYPE value) {
    if constexpr (
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
        FMT == DataFormat::Tf32) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::Float16 ? sfpi::DataLayout::F16a
                                    : FMT == DataFormat::Float16_b ? sfpi::DataLayout::F16b
                                    : FMT == DataFormat::Float32 || FMT == DataFormat::Tf32
                                    ? sfpi::DataLayout::F32 : sfpi::DataLayout::None;
        sfpi::dst_reg[0].mode<layout>(ADDR_MOD_6) = sfpi::as<sfpi::vFloat>(value);
    } else if constexpr (FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::Int32 ? sfpi::DataLayout::SM32
                                    : FMT == DataFormat::Int16 ? sfpi::DataLayout::SM16
                                    : FMT == DataFormat::Int8  ? sfpi::DataLayout::SM8
                                    : sfpi::DataLayout::None;
        sfpi::dst_reg[0].mode<layout>(ADDR_MOD_6) = sfpi::as<sfpi::vSMag>(value);
    } else if constexpr (FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8) {
        constexpr sfpi::DataLayout layout = FMT == DataFormat::UInt16  ? sfpi::DataLayout::U16
                                    : FMT == DataFormat::UInt8 ? sfpi::DataLayout::U8
                                    : sfpi::DataLayout::None;
        sfpi::dst_reg[0].mode<layout>(ADDR_MOD_6) = sfpi::as<sfpi::vUInt>(value);
    } else {
        // No dedicated SFPU mode (fp8, MX block formats, 4-bit ints): fall back to the implied/
        // default register-file format. Matches the runtime overload's default case.
        sfpi::dst_reg[0].mode<sfpi::DataLayout::FSrcB>(ADDR_MOD_6) = sfpi::as<sfpi::vFloat>(value);
    }
}

// fp32 -> fp16 narrow, picking the fp16a/fp16b variant from the destination format. Round-nearest-
// even matches the validated TTI encoding (the immediate-descale path is int-only and unused here).
template <DataFormat DST_FMT>
inline sfpi::vFloat _typecast_narrow_to_fp16_(sfpi::vFloat value) {
    using fp16_t = std::conditional_t<DST_FMT == DataFormat::Float16, sfpi::vFloat16a, sfpi::vFloat16b>;
    return sfpi::convert<fp16_t>(value, sfpi::RoundMode::NearestEven);
}

// sign-mag int32 -> 8-bit int round-and-narrow (the proven INT32_TO_(U)INT8 path), picking the
// unsigned/signed variant from the destination format. Round-nearest-even matches the TTI baseline.
template <DataFormat DST_FMT>
inline auto _typecast_narrow_to_int8_(sfpi::vInt value) {
    if constexpr (_typecast_is_unsigned_int_<DST_FMT>()) {
        return sfpi::int32_to_uint8(value, 0u, sfpi::RoundMode::NearestEven);
    } else {
        return sfpi::int32_to_int8(value, 0u, sfpi::RoundMode::NearestEven);
    }
}

// Convert one SFPU pass worth of rows from SRC_FMT to DST_FMT. The (src,dst) classes pick the
// sequence (load to fp32 for floats, to sign-mag int32 for ints; store sfpmem mode sets the output):
//   float->float : narrow via convert<fp16> (-> fp16), else store the loaded fp32 as-is.
//   float->int   : one SFP_STOCH_RND to the narrow int (it rounds the full float and saturates).
//                  Int32 has no such mode. Below 32768, SFPCAST keeps only 8 fraction bits, so
//                  that range uses the int16 convert. Above it a float32 has no finer bit.
//   int->float   : convert<vFloat> int32 -> fp32, then narrow via convert<fp16> for an fp16 dst.
//   int->int     : narrow to 8-bit if needed, else the store sfpmem mode widens/narrows.
template <DataFormat SRC_FMT, DataFormat DST_FMT>
inline void _calculate_typecast_arith_sfp_rows_() {
    constexpr bool src_float = _typecast_is_float_<SRC_FMT>();
    constexpr bool dst_float = _typecast_is_float_<DST_FMT>();

    if constexpr (src_float && dst_float) {
        sfpi::vFloat value = _typecast_load_<sfpi::vFloat, SRC_FMT>();
        if constexpr (_typecast_is_fp16_<DST_FMT>()) {
            value = _typecast_narrow_to_fp16_<DST_FMT>(value);
        }
        _typecast_store_<DST_FMT>(value);
    } else if constexpr (src_float && !dst_float) {
        // SFPCAST to int32 keeps 8 fraction bits, so a value in (k+0.5, k+0.5+1/256) for even k
        // rounds to k. The narrow SFP_STOCH_RND modes round the whole float and saturate.
        // The unsigned modes drop the sign, so the ReLU stays in front of them.
        sfpi::vFloat value = _typecast_load_<sfpi::vFloat, SRC_FMT>();

        if constexpr (_typecast_is_unsigned_int_<DST_FMT>()) {
            value = sfpi::rectified_linear_unit(value);
        }

        if constexpr (DST_FMT == DataFormat::Int8) {
            _typecast_store_<DST_FMT>(sfpi::convert<sfpi::vSMag8>(value, sfpi::RoundMode::NearestEven));
        } else if constexpr (DST_FMT == DataFormat::UInt8) {
            _typecast_store_<DST_FMT>(sfpi::convert<sfpi::vUInt8>(value, sfpi::RoundMode::NearestEven));
        } else if constexpr (DST_FMT == DataFormat::Int16) {
            _typecast_store_<DST_FMT>(sfpi::convert<sfpi::vSMag16>(value, sfpi::RoundMode::NearestEven));
        } else if constexpr (DST_FMT == DataFormat::UInt16) {
            _typecast_store_<DST_FMT>(sfpi::convert<sfpi::vUInt16>(value, sfpi::RoundMode::NearestEven));
        } else if constexpr (DST_FMT == DataFormat::Int32) {
            // Only |x| < 32768 can have a fraction bit finer than 1/256, and that is the range
            // mode 7 covers. Above it SFPCAST already sees every float32 bit.
            sfpi::vSMag int_value = sfpi::convert<sfpi::vSMag>(value, sfpi::RoundMode::NearestEven);
            v_if(sfpi::abs(value) < 32767.5f) {
                int_value = sfpi::as<sfpi::vSMag>(sfpi::convert<sfpi::vSMag16>(value, sfpi::RoundMode::NearestEven));
            }
            v_endif;
            _typecast_store_<DST_FMT>(int_value);
        } else {
            sfpi::vSMag int_value = sfpi::convert<sfpi::vSMag>(value, sfpi::RoundMode::NearestEven);
            _typecast_store_<DST_FMT>(int_value);
        }
    } else if constexpr (!src_float && dst_float) {
        sfpi::vSMag int_value = _typecast_load_<sfpi::vSMag, SRC_FMT>();
        sfpi::vFloat float_value = sfpi::convert<sfpi::vFloat>(int_value, sfpi::RoundMode::NearestEven);
        if constexpr (_typecast_is_fp16_<DST_FMT>()) {
            float_value = _typecast_narrow_to_fp16_<DST_FMT>(float_value);
        }
        _typecast_store_<DST_FMT>(float_value);
    } else {
        // 8-bit dst rounds-and-narrows; a 16-bit dst is narrowed by the store sfpmem mode.
        sfpi::vInt value = _typecast_load_<sfpi::vInt, SRC_FMT>();
        if constexpr (_typecast_is_int8_<DST_FMT>()) {
            _typecast_store_<DST_FMT>(_typecast_narrow_to_int8_<DST_FMT>(value));
        } else {
            _typecast_store_<DST_FMT>(value);
        }
    }
}

/**
 * @brief Cast a Dest tile in place from SRC_FMT to DST_FMT, element by element.
 *
 * Each loop pass converts one SFPU pass worth of rows (SFP_ROWS) and stores the result back to
 * Dest through ADDR_MOD_6, which advances the Dest pointer — so no explicit increment is emitted.
 * MX / block-float typecasts are a pure unpack/pack gasket conversion (a datacopy) and never reach
 * this op; instantiating it with an MX endpoint is unsupported by design.
 *
 * @tparam SRC_FMT: Source data format (the format currently in Dest).
 * @tparam DST_FMT: Destination data format to convert each element to.
 * @tparam ITERATIONS: Number of SFPU passes (each covers SFP_ROWS rows) needed to span the tile.
 * @note Call @ref init_typecast first to program the ADDR_MOD_6 every path stores through,
 *       including the dedicated UInt16 → Float32 kernel.
 */
template <DataFormat SRC_FMT, DataFormat DST_FMT, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_typecast() {
    if constexpr (SRC_FMT == DataFormat::UInt16 && DST_FMT == DataFormat::Float32) {
        // UInt16 in a 32-bit Dest word: load the low 16 bits zero-extended and store fp32.
        _calculate_typecast_uint16_to_fp32_<ITERATIONS>();
        return;
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_typecast_arith_sfp_rows_<SRC_FMT, DST_FMT>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
