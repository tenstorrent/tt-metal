// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `smoothstep-fresh` coverage row (metal
// experimental ckernel_sfpu_smoothstep.h smoothstep_tile_face, corpus
// manifest class D-ABSENT — zero dispatch anywhere).  Mathematical definition
// (GLSL/torch smoothstep): with t = clamp((x - edge0) / (edge1 - edge0), 0, 1),
//   y = t^2 * (3 - 2t)
// The kernel contract passes edge0 and the precomputed 1/(edge1-edge0)
// (inv_delta) as fp32 scalars; the clamp is stated with typed min/max.
// bf16 RNE store per the fresh float-body convention.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_smoothstep_fresh_cpp(const std::uint32_t edge0_bits, const std::uint32_t inv_delta_bits)
{
    const sfpi::vFloat edge0     = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(edge0_bits)));
    const sfpi::vFloat inv_delta = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(inv_delta_bits)));
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t       = (x - edge0) * inv_delta;
        t                    = sfpi::max(sfpi::min(t, 1.0f), 0.0f);
        const sfpi::vFloat y = t * t * (3.0f - 2.0f * t);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
