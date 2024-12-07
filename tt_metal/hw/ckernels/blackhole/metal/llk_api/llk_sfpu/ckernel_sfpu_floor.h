// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"
#include "limits.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline vInt float_to_int32(vFloat in)
{
    vInt result;
    vInt exp = exexp(in); // extract exponent
    v_if (exp < 0) {
        result = 0;
    } v_elseif (exp > 30) {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<int32_t>::max();
        // check sign
        v_if (in < 0) {
            result = reinterpret<vInt>(setsgn(reinterpret<vFloat>(result), 1));
        } v_endif
    } v_else {
        // extract mantissa
        vInt man = exman8(in);
        // shift the mantissa by (23-exponent) to the right
        vInt shift = exp - 23;
        man = shft(reinterpret<vUInt>(man), shift);
        // check sign
        v_if (in < 0) {
            man = reinterpret<vInt>(setsgn(reinterpret<vFloat>(man), 1));
        } v_endif
        result = man;
    } v_endif
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_floor() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = dst_reg[0];
        vFloat v = result;
        vInt tmp = float_to_int16(result, 0);
        result = int32_to_float(tmp, 0);
        v_if(result > v) { result = result - 1; }
        v_endif;
        v_if(v <= SHRT_MIN || v >= SHRT_MAX) { result = v; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_floor_float32() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = dst_reg[0];
        vFloat v = result;
        vInt tmp = float_to_int32(result);
        result = int32_to_float(tmp, 0);
        v_if(result > v) { result = result - 1; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
