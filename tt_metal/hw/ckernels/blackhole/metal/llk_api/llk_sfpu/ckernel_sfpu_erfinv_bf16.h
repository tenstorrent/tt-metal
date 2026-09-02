// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_bf16_poly_common.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// BF16 erfinv kernel (Blackhole): route family "log_square_factorized_odd"
// from ckernel_sfpu_bf16_poly_common.h,
//
//     t = ln(1 - x^2);   erfinv(x) ~= x * P3(|t|)   on |x| < 1
//
// with x = +/-1 -> +/-Inf, |x| > 1 / +/-Inf / NaN -> +Inf, and zeros /
// BF16 subnormals -> +0 (see the family header for why the conversion
// pipeline yields +Inf / +0 there).  On the open interval the result matches
// torch.erfinv to below 1 ULP.  Only the certified coefficient table
// below is specific to erfinv.
//
// Provenance (tt-polynomial-fitter, erfinv_p3_s1_uniform_lsq_ulp_block-whole_log_ratio.csv):
//   P3 least-squares fit, ULP-selected, on |t| in [0, 5.55];
//   coefficients sha256 0da6b90dcc523c25;
//   silicon validation: all 65,536 BF16 encodings on Blackhole,
//   max pure ULP 0.8324 / mean 0.2492
//   (tt-polynomial-fitter exabox job 75313, shard18, main d6368841a0); the replaced
//   implementation measures 255.2 max / 168.4 mean on the same sweep.
//
// This path serves the BF16 destination-register case only; fp32 dest
// (is_fp32_dest_acc_en) keeps the pre-existing implementation.

struct ErfinvBf16Config {
    // log1p correction on the anchored reduction (see log1p_anchored_bf16).
    static constexpr float log1p_c0 = -0x1.00cefep-1f;  // 0xBF00677F = -0.501579225  (c0)
    static constexpr float log1p_c1 = 0x1.617f6ap-2f;   // 0x3EB0BFB5 = 0.345212609  (c1)
    static constexpr float log1p_c2 = -0x1.a0ed2ep-3f;  // 0xBE507697 = -0.203577384  (c2)

    // P3(|t|) coefficients, lowest power first.
    static constexpr int poly_degree = 3;
    static constexpr float poly[poly_degree + 1] = {
        0x1.c5bf8ap-1f,    // 0x3F62DFC5 = 0.886226952  (|t|^0)
        0x1.e690d2p-3f,    // 0x3E734869 = 0.23758091  (|t|^1)
        0x1.8b17cap-8f,    // 0x3BC58BE5 = 0.00602863962  (|t|^2)
        -0x1.2db84cp-10f,  // 0xBA96DC26 = -0.00115097011  (|t|^3)
    };
};

template <int ITERATIONS = 8>
inline void calculate_erfinv_bf16() {
    calculate_log_square_factorized_odd_bf16<ErfinvBf16Config, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
