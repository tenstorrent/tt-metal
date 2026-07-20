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
// non-negative). The raw load builtin is used directly (rather than sfpi::dst_reg[].mode<>())
// because the DataLayout abstraction has no encoding for UInt8 (sfpmem 0b1011) and no 8-bit Dest
// operator on Quasar; the _sfpu_sfpmem_type_<FMT>() selector covers every typecast endpoint. The
// load uses ADDR_MOD_7 (all-zeroes, no increment) — the paired store advances Dest via ADDR_MOD_6.
// TODO: once DataLayout gains Int8/UInt8, load via sfpi::dst_reg[0].mode<...>() instead of the raw builtin.
template <typename VTYPE, DataFormat FMT>
inline VTYPE _typecast_load_() {
    return VTYPE(__builtin_rvtt_sfpload(0 /* dest_reg */, _sfpu_sfpmem_type_<FMT>(), ADDR_MOD_7));
}

// ADDR_MOD_6 post-increments Dest by one SFPU pass (SFP_ROWS), so the store both writes the result
// and advances to the next pair of rows — replacing the per-iteration _incr_counters_. Requires
// init_typecast to have programmed ADDR_MOD_6.
// TODO: once DataLayout gains Int8/UInt8, store via sfpi::dst_reg[0].mode<...>() instead of the raw builtin.
template <DataFormat FMT, typename TYPE>
inline void _typecast_store_(TYPE value) {
    __builtin_rvtt_sfpstore(value.get(), 0 /* dest_reg */, _sfpu_sfpmem_type_<FMT>(), ADDR_MOD_6);
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
//   float->int   : convert<vSMag> fp32 -> sign-mag int32, then narrow to 8-bit / 16-bit if needed.
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
        // The direct fp32 -> narrow-int convert modes round to all-zeros on Quasar, so compose two
        // proven steps: fp32 -> sign-mag int32, then narrow from int32. Clamp negatives to 0 first
        // for unsigned targets (CC sign test predicates the zero-write) so the int32 is non-negative.
        sfpi::vFloat value = _typecast_load_<sfpi::vFloat, SRC_FMT>();

        if constexpr (_typecast_is_unsigned_int_<DST_FMT>()) {
            value = sfpi::rectified_linear_unit(value);
        }

        // fp32 -> sign-magnitude int32 (SFPCAST, round-nearest-even)
        sfpi::vSMag int_value = sfpi::convert<sfpi::vSMag>(value, sfpi::RoundMode::NearestEven);
        if constexpr (_typecast_is_int8_<DST_FMT>()) {
            _typecast_store_<DST_FMT>(_typecast_narrow_to_int8_<DST_FMT>(sfpi::as<sfpi::vInt>(int_value)));
        } else {
            // Int32/16-bit dst: the store sfpmem mode narrows a 16-bit dst; int32 stores as-is.
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
 * @note Call @ref init_typecast first to program the ADDR_MOD_6 it stores through.
 */
template <DataFormat SRC_FMT, DataFormat DST_FMT, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_typecast() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_typecast_arith_sfp_rows_<SRC_FMT, DST_FMT>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
