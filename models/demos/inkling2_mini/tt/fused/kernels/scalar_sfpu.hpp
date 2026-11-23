// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// SFPU ops on a scalar held at [0, 0] of a DST tile, the element the bcast-scalar FPU ops read: the stock
// per-element SFPU functions on face 0 only (VectorMode::None), IT 4-row groups of it (1 = rows 0..3 of the even
// columns, which holds [0, 0]; 8 = the whole face). The other elements keep whatever they held.
#pragma once

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"

namespace scal {
constexpr VectorMode V = VectorMode::None;

template <int IT>
FORCE_INLINE void mul_unary(uint32_t d, uint32_t p) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, MUL_UNARY, IT, DST_ACCUM_MODE), d, V, p));
}
template <int IT>
FORCE_INLINE void add_unary(uint32_t d, uint32_t p) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, ADD_UNARY, IT, DST_ACCUM_MODE), d, V, p));
}
template <int IT>
FORCE_INLINE void max_unary(uint32_t d, uint32_t p) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min, (true, APPROX, IT), d, V, p));
}
template <int IT>
FORCE_INLINE void rsqrt(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, IT, DST_ACCUM_MODE, false), d, V));
}
template <int IT>
FORCE_INLINE void recip(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (APPROX, DST_ACCUM_MODE, IT), d, V));
}
template <int IT>
FORCE_INLINE void exp(uint32_t d) {
    exp_tile<false, false, InputClamping::ClampToNegative, IT>(d, V);
}
template <int IT>
FORCE_INLINE void bmax(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (true, IT), a, b, o, V)));
}
template <int IT>
FORCE_INLINE void sub(uint32_t a, uint32_t b, uint32_t o) {
    sub_binary_tile<ckernel::DstRoundingMode::Default, DST_ACCUM_MODE, IT>(a, b, o, V);
}
template <int IT>
FORCE_INLINE void add(uint32_t a, uint32_t b, uint32_t o) {
    add_binary_tile<ckernel::DstRoundingMode::Default, DST_ACCUM_MODE, IT>(a, b, o, V);
}
template <int IT>
FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t o) {
    mul_binary_tile<DST_ACCUM_MODE, IT>(a, b, o, V);
}
}  // namespace scal
