// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_bf16_poly_common.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Wormhole BF16 erfinv kernel: same route family and coefficient table as
// Blackhole (`ckernel_sfpu_bf16_poly_common.h`).
//
//     t = ln(1 - x^2);   erfinv(x) ~= x * P3(|t|)   on |x| < 1
//
// Specials: x = +/-1 -> +/-Inf; |x| > 1 / +/-Inf / NaN -> +Inf; zeros /
// BF16 subnormals -> +0. Both architecture paths are covered by
// tests/ttnn/unit_tests/operations/eltwise/test_erfinv_bf16_exhaustive.py.
//
// This path serves the BF16 destination-register case only; fp32 dest
// (is_fp32_dest_acc_en) keeps the pre-existing Wormhole implementation.

struct ErfinvBf16Config {
    static constexpr float log1p_c0 = -0x1.00cefep-1f;
    static constexpr float log1p_c1 = 0x1.617f6ap-2f;
    static constexpr float log1p_c2 = -0x1.a0ed2ep-3f;

    static constexpr int poly_degree = 3;
    static constexpr float poly[poly_degree + 1] = {
        0x1.c5bf8ap-1f,
        0x1.e690d2p-3f,
        0x1.8b17cap-8f,
        -0x1.2db84cp-10f,
    };
};

template <int ITERATIONS = 8>
inline void calculate_erfinv_bf16() {
    calculate_log_square_factorized_odd_bf16<ErfinvBf16Config, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
