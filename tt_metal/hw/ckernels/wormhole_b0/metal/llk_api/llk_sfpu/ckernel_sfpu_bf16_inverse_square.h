// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_inverse_square_core.h"
}
namespace ckernel::sfpu::bf16 {
template <class Config, int Iterations = 32>
inline void calculate_inverse_square() {
    static_assert(Iterations == 32);
    static_assert(!Config::kDirectTail && Config::kFiniteReferencePrecedence);
    sfpu_reciprocal_init<false>();
    static_assert(Config::kWhRepair);
    for (int d = 0; d < Iterations; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[d];
        sfpi::vFloat result = sfpi::inverse_square_scalar<Config>(
            x,
            [](sfpi::vFloat value) { return sfpu_reciprocal_iter<1>(value); },
            [](sfpi::vFloat value) { return value; });
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        if constexpr (Config::kNanResultWord || Config::kPositiveInfinityWord || Config::kNegativeInfinityWord) {
            // Typed raw terminals of the non-finite inputs, after rounding as on Blackhole.
            v_if(sfpi::exexp(sfpi::setsgn(x, 0), sfpi::ExponentMode::Biased) == 255) {
                result = __builtin_bit_cast(float, uint32_t(Config::kNanResultWord) << 16);
            }
            v_endif;
            v_if(sfpi::as<sfpi::vInt>(x) == std::int32_t(0x7f800000)) {
                result = __builtin_bit_cast(float, uint32_t(Config::kPositiveInfinityWord) << 16);
            }
            v_endif;
            v_if(sfpi::as<sfpi::vInt>(x) == std::int32_t(0xff800000)) {
                result = __builtin_bit_cast(float, uint32_t(Config::kNegativeInfinityWord) << 16);
            }
            v_endif;
        }
        sfpi::dst_reg[d] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
