// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"

#include "sfpi.h"
#include "ckernel_sfpu_conversions.h"


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

// UINT32 -> UINT8: Clamp to [0, 255], store as INT32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        // Clamping logic for UINT32 -> UINT8. Since v is reinterpreted as vInt,
        // we handle both large positive values and potential negative bits (if vInt is signed).
        v_if(v < 0 || v > 255) { v = 255; }
        v_endif;
        dst_reg[0] = reinterpret<vFloat>(v);
        dst_reg++;
    }
}

// INT32 -> UINT8: Clamp to [0, 255], store as INT32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        v_if(v < 0) { v = 0; }
        v_endif;
        v_if(v > 255) { v = 255; }
        v_endif;
        dst_reg[0] = reinterpret<vFloat>(v);
        dst_reg++;
    }
}

// FP32 -> UINT8: Clamp to [0, 255] range, convert to integer using a robust rounding trick,
// and store as INT8. The packer (configured for UInt8 output) expects the data
// in the low 8 bits of the destination register.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load FP32 value from dest register
        vFloat v = dst_reg[0];

        // Clamp to [0, 255] range.
        v_if(v < 0.0f) { v = 0.0f; }
        v_endif;
        v_if(v > 255.0f) { v = 255.0f; }
        v_endif;

        // Convert the clamped positive float to a uint32 integer.
        // We use the 2**23 rounding trick: adding 2**23 to [0, 255] puts the integer
        // value in the mantissa bits (0-22).
        vFloat v_rounded = v + 8388608.0f;
        vInt result = reinterpret<vInt>(v_rounded) & 0x7FFFFF;

        // Store the integer result.
        dst_reg[0] = reinterpret<vFloat>(result);
        dst_reg++;
    }
}

// UINT8 -> FP32: The unpacker zero-extends UInt8 to the dest register width (as an integer).
// We then convert the integer bits to Float32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        dst_reg[0] = int32_to_float(v, 0);
        dst_reg++;
    }
}

// UINT8 -> FP16B: Same rationale as UINT8 -> FP32; zero-extended by unpacker,
// then converted to Float32. The packer (configured for FP16B output)
// handles the conversion to FP16B.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp16b() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        dst_reg[0] = int32_to_float(v, 0);
        dst_reg++;
    }
}

// UINT32 -> UINT32: Identity (no-op) used for UINT8 -> UINT32/INT32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg++;
    }
}

// FP32 -> UINT8: Uses INT8 store format (spec value 5).
// The packer for UInt8 output expects the data in the low 8 bits.
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT8, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp32() {
    // No specific SFPLOADMACRO pipeline needed; uses dst_reg directly.
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp16b() {
    // No specific SFPLOADMACRO pipeline needed; uses dst_reg directly.
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint32() {
    // Identity copy.
}

}  // namespace sfpu
}  // namespace ckernel
