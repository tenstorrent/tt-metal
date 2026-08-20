// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `addcdiv` corpus row (metal
// calculate_addcdiv).  Mathematical definition (torch.addcdiv):
//
//   out = a + value * (b / c)
//
// evaluated in fp32 (the golden's statement), with the scalar `value`
// arriving as raw fp32 bits exactly as the production dispatch sends it.
// The division is stated as multiplication by the divisor's reciprocal
// (fresh_common.h shared hardware-seed statement, sign-agnostic — the
// Newton identity holds for either sign, so no sign split is needed); the
// suite's stimuli hold c away from zero, the statement's only exclusion.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, DataFormat FORMAT, int ITERATIONS>
inline void calculate_addcdiv_fresh_cpp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out,
    const std::uint32_t value)
{
    static_assert(FORMAT == DataFormat::Float32 || FORMAT == DataFormat::Float16_b || FORMAT == DataFormat::Bfp8_b);
    constexpr std::uint32_t tile_rows = 32;
    const sfpi::vFloat scale          = Converter::as_float(value);

#pragma GCC unroll 8
    for (int row = 0; row < ITERATIONS; ++row)
    {
        const sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * tile_rows];
        const sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * tile_rows];
        const sfpi::vFloat c = sfpi::dst_reg[dst_index_in2 * tile_rows];

        const sfpi::vFloat c_recip = fresh_recip_hwseed(c);
        sfpi::vFloat result        = (scale * b) * c_recip + a;
        if constexpr (!DST_ACCUM_MODE)
        {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[dst_index_out * tile_rows] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
