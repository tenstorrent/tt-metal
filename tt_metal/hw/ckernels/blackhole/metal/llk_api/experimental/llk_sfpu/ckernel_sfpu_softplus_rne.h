// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_softplus.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

// RNE-rounded softplus body: ``is_fp32_dest_acc_en=false`` forces an
// explicit RNE round to bf16 via ``convert<vFloat16b>`` after the SFPU
// compute (fp32 in LRegs; the default SFPSTORE truncates and is
// downward-biased ~0.5 ULP vs torch's ``F.softplus(bf16(x)).to(bf16)``).
// Empirically more accurate for the DeepSeek-V4 router gate-MM's
// bf16-tied top-k picks than the fp32-dst store path.
template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body_rne(const float beta, const float beta_reciprocal, const float threshold) {
    calculate_softplus_body<APPROXIMATION_MODE, /*is_fp32_dest_acc_en=*/false>(beta, beta_reciprocal, threshold);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus_rne(std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    const auto beta = Converter::as_float(param0);
    const auto beta_reciprocal = Converter::as_float(param1);
    const auto threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body_rne<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
