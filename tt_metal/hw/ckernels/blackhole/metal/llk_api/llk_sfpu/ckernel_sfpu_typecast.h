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
inline void calculate_typecast_fp16b_to_uint16() {
    _calculate_typecast_fp16b_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
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
inline void calculate_typecast_fp16b_to_int32() {
    _calculate_typecast_fp16b_to_int32_<APPROXIMATION_MODE, ITERATIONS>();
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
inline void calculate_typecast_fp16b_to_uint32() {
    _calculate_typecast_fp16b_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
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

}  // namespace sfpu
}  // namespace ckernel
