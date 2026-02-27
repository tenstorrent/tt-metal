// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "ckernel_sfpu_conversions.h"


using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Standard typecast internal implementations (Restored)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_fp16b_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_uint32_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp32_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp32_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp32_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_fp16b_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_fp16b_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_fp16b_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_uint16_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint32_to_uint16_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_int32_to_uint16_() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

// Public wrappers
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

// UINT32 -> UINT8: Clamp to [0, 255], then OR into bit pattern 0x4B000000.
// This preserves the integer bits in the mantissa (bits 0-7) and avoids subnormal flushing.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        // Clamping logic for UINT32 -> UINT8. Values > 2^31 appear as negative.
        v_if(v < 0 || v > 255) {
            v = 255;
        }
        v_endif;
        // OR into 2^23 bit pattern (0x4B000000)
        vInt bits = v | vInt(0x4B000000);
        dst_reg[0] = reinterpret<vFloat>(bits);
        dst_reg++;
    }
}

// INT32 -> UINT8: Clamp to [0, 255], then bitwise OR into 2^23 pattern.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        v_if(v < 0) { v = 0; }
        v_elseif(v > 255) { v = 255; }
        v_endif;
        vInt bits = v | vInt(0x4B000000);
        dst_reg[0] = reinterpret<vFloat>(bits);
        dst_reg++;
    }
}

// FP32 -> UINT8: Clamp to [0, 255], then use the 2**23 rounding trick.
// Instead of reinterpreting to subnormal (which flushes to 0), we leave it
// as a float 2**23 + integer_value (0x4B0000XX). The packer reads the low byte.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v < 0.0f) { v = 0.0f; }
        v_endif;
        v_if(v > 255.0f) { v = 255.0f; }
        v_endif;

        // Standard 2^23 trick for float to integer conversion bits.
        dst_reg[0] = v + 8388608.0f;
        dst_reg++;
    }
}

// UINT8 -> FP32: Zero-extended integer bits to float value.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        dst_reg[0] = int32_to_float(v, 0);
        dst_reg++;
    }
}

// UINT8 -> FP16B: Same as above.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint8_to_fp16b() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = reinterpret<vInt>(dst_reg[0]);
        dst_reg[0] = int32_to_float(v, 0);
        dst_reg++;
    }
}

// IDENTITY for UINT8 -> UINT32/INT32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp32() {
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint8_to_fp16b() {
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint32() {
}

}  // namespace sfpu
}  // namespace ckernel
