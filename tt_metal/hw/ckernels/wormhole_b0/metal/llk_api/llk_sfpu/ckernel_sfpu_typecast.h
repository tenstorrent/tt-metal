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

// UINT32 -> UINT8: Clamp to [0, 255], store as INT32.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = reinterpret<vUInt>(dst_reg[0]);
        v_if(v > 255u) { v = 255u; }
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

        // Convert the clamped positive float to a uint32 integer.
        // vInt(vFloat) uses the SFPCAST hardware instruction for full-precision conversion.
        vInt result = vInt(v);

        // Store the integer result; use reinterpret to ensure bits are preserved
        // without float conversion. The packer for UInt8 output will
        // take the low 8 bits.
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
        vUInt v = reinterpret<vUInt>(dst_reg[0]);
        dst_reg[0] = vFloat(v);
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
        vUInt v = reinterpret<vUInt>(dst_reg[0]);
        dst_reg[0] = vFloat(v);
        dst_reg++;
    }
}

// Init functions for UINT8 typecast operations
// FP32 -> UINT8: Uses INT32 store format. The packer for UInt8 output
// will take the low 8 bits from the dest register.
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp32() {
    // No specific SFPLOADMACRO pipeline needed; uses dst_reg directly.
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp16b() {
    // No specific SFPLOADMACRO pipeline needed; uses dst_reg directly.
}

}  // namespace sfpu
}  // namespace ckernel
