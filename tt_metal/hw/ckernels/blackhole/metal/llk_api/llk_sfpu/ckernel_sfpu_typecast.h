// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint16() {
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
    _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() {
    _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
    _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
    _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    _init_typecast_fp32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    _init_typecast_uint16_to_uint32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
    _init_typecast_uint32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
    _init_typecast_int32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() {
    _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
    _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() {
    _init_typecast_int32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() {
    _init_typecast_uint32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {
    _init_typecast_fp32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {
    _init_typecast_uint32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
    _init_typecast_int32_to_uint16_<APPROXIMATION_MODE>();
}

// UINT8 typecast support functions

// FP32 -> UINT8: Clamp to [0, 255], convert to unsigned integer, store as INT32.
// The packer (configured for UInt8 output) reads the low 8 bits from the dest register.
// Cannot reuse the UINT16 SFPLOADMACRO pipeline directly because its init configures
// the store format as LO16, which the UInt8 packer cannot read correctly (produces all zeros).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load FP32 value from dest register
        vFloat v = dst_reg[0];

        // Clamp to [0, 255] range
        v_if(v < 0.0f) { v = 0.0f; }
        v_endif;
        v_if(v > 255.0f) { v = 255.0f; }
        v_endif;

        // Convert the clamped positive float to a uint32 integer using
        // the same mantissa-extraction approach as _float_to_int32_positive_.
        // For values in [0, 255] this produces the correct integer in [0, 255].
        vInt result;
        vInt exp = exexp(v);
        v_if(exp < 0) { result = 0; }
        v_else {
            vInt man = exman8(v);
            vInt shift = exp - 23;
            man = reinterpret<vInt>(shft(reinterpret<vUInt>(man), shift));
            result = man;
        }
        v_endif;

        // Store the integer result; the packer for UInt8 output will
        // take the low 8 bits (which are guaranteed to be in [0, 255]).
        dst_reg[0] = result;
        dst_reg++;
    }
}

// UINT8 -> FP32: The unpacker zero-extends UInt8 to the dest register width.
// We then use the UINT16->FP32 cast (SFPCAST: unsigned int to float) which
// handles the zero-extended value correctly.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

// UINT8 -> FP16B: Same rationale as UINT8 -> FP32; zero-extended by unpacker,
// then UINT16->FP16B cast handles the conversion.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

// Init functions for UINT8 typecast operations
// FP32 -> UINT8: No SFPLOADMACRO pipeline needed because the compute function
// uses dst_reg directly. Fall through to the default init.
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    // Default init — no SFPLOADMACRO templates required.
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp32() {
    _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp16b() {
    _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
