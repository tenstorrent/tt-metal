// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"
#include "llk_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_uint16() {
    _calculate_typecast_fp16b_to_uint16_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
    _calculate_typecast_int32_to_fp16b_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_int32() {
    _calculate_typecast_fp16b_to_int32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    _calculate_typecast_fp32_to_fp16b_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
    _calculate_typecast_int32_to_fp32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_fp16b_to_uint32() {
    _calculate_typecast_fp16b_to_uint32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    _calculate_typecast_uint32_to_fp16b_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    _calculate_typecast_uint32_to_fp32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    _calculate_typecast_uint16_to_uint32_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    _calculate_typecast_uint32_to_uint16_<APPROX_MODE, ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    _calculate_typecast_int32_to_uint16_<APPROX_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
