// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SFPU ops of the GDN step split 4 on the live rows only (QWEN38_GDN_ROWTRIM).  The step has rows = 1: only row 0 of
// every tile is live.  With ITERS = 2 an op runs VectorMode::R with 2 iterations per face: rows 0..3 of faces 0 and 1
// (one iteration is 4 rows x 8 columns of one column parity), which holds the live row at all 32 columns.  ITERS = 0 is
// the stock op (VectorMode::RC, 8 iterations): kernels that never define the iteration counts compile today's code.
#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "../../kernels/source_reciprocal.h"

// SFPU iterations per face of the ops whose rows 4.. never reach a live row (row-local consumers), 0 = stock.
#ifndef QWEN38_GDN_ROWTRIM_ITERS
#define QWEN38_GDN_ROWTRIM_ITERS 0
#endif
// ... and of the ops whose rows 4.. reach the state only through the exactly zero rows of the mask (k^T delta_b), 0 =
// stock.
#ifndef QWEN38_GDN_ROWTRIM_MASKED_ITERS
#define QWEN38_GDN_ROWTRIM_MASKED_ITERS 0
#endif

namespace rowtrim {
constexpr int LOCAL = QWEN38_GDN_ROWTRIM_ITERS;
constexpr int MASKED = QWEN38_GDN_ROWTRIM_MASKED_ITERS;

template <int ITERS>
ALWI void add_unary(uint32_t idst, uint32_t param) {
    if constexpr (ITERS == 0) {
        add_unary_tile(idst, param);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, ITERS /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::R,
            param));
    }
}

template <int ITERS>
ALWI void mul_unary(uint32_t idst, uint32_t param) {
    if constexpr (ITERS == 0) {
        mul_unary_tile(idst, param);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, ITERS /* ITERATIONS */, DST_ACCUM_MODE),
            idst,
            VectorMode::R,
            param));
    }
}

template <int ITERS>
ALWI void sqrt(uint32_t idst) {
    if constexpr (ITERS == 0) {
        sqrt_tile(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sqrt,
            (APPROX, ITERS /* ITERATIONS */, DST_ACCUM_MODE, false),
            idst,
            VectorMode::R));
    }
}

template <int ITERS>
ALWI void recip(uint32_t idst) {
    if constexpr (ITERS == 0) {
        qwen38_recip_tile(idst);
    } else {
        qwen38_recip_tile(idst, VectorMode::R, ITERS);
    }
}

template <int ITERS>
ALWI void sigmoid(uint32_t idst) {
    if constexpr (ITERS == 0) {
        sigmoid_tile<VectorMode::RC, false>(idst);
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sigmoid,
            (false, DST_ACCUM_MODE, ITERS /* ITERATIONS */),
            idst,
            VectorMode::R));
    }
}

template <int ITERS>
ALWI void sub_binary(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    if constexpr (ITERS == 0) {
        sub_binary_tile(idst0, idst1, odst);
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::SUB, ITERS /* ITERATIONS */, DST_ACCUM_MODE, DstRoundingMode::Default),
            idst0,
            idst1,
            odst,
            VectorMode::R)));
    }
}

template <int ITERS>
ALWI void mul_binary(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    if constexpr (ITERS == 0) {
        mul_binary_tile(idst0, idst1, odst);
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, ITERS /* ITERATIONS */, DST_ACCUM_MODE),
            idst0,
            idst1,
            odst,
            VectorMode::R)));
    }
}
}  // namespace rowtrim
