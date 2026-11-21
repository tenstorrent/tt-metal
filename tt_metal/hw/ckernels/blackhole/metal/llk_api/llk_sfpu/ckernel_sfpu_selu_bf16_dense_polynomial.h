// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly::selu {
using ::sfpi::as;
using ::sfpi::DataLayout;
using ::sfpi::vFloat;
using ::sfpi::vInt;
// The one-sided clamp compares in order on WH and takes the SFPI maximum on BH.
template <bool Ordered, uint32_t BoundBits>
__attribute__((always_inline)) inline vFloat dense_lower_clamp(vFloat input) {
    constexpr float bound = __builtin_bit_cast(float, BoundBits);
    if constexpr (Ordered) {
        v_if(input < bound) { input = bound; }
        v_endif;
        return input;
    } else {
        return sfpi::max(input, bound);
    }
}

template <typename Config>
struct dense_polynomial_transport {
    // A constant a Config does not park (kParked false: none) loads as an immediate.
    template <uint32_t IDX>
    __attribute__((always_inline)) static inline vFloat lut_fetch() {
        return vFloat(Config::lut(IDX));
    }

    template <uint32_t BASE>
    struct segment_coefficients {
        template <uint32_t INDEX>
        __attribute__((always_inline)) inline vFloat load() const {
            return lut_fetch<BASE + INDEX>();
        }
    };

    // A segment from the smallest positive subnormal owns x > 0. A float compare reads a subnormal
    // x as 0, but TT-NN delivers subnormals to DEST unflushed, so the raw bits decide: -0 reads negative.
    template <uint32_t SEG>
    static constexpr bool positive_segment() {
        return __builtin_bit_cast(uint32_t, Config::lut(SEG)) == 1u;
    }

    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void eval_seg_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
        constexpr uint32_t DEG = Config::degree(SEG);
        constexpr uint32_t BASE = Config::kCoefficientOffset + SEG * Config::kCoefficientsPerSegment;
        eval_polynomial_transport<DEG>(segment_coefficients<BASE>{}, x1, x2, r1, r2);
    }

    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void cascade_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
        if constexpr (SEG < Config::kSegments) {
            vFloat tmp1, tmp2;
            eval_seg_dual<SEG>(x1, x2, tmp1, tmp2);
            if constexpr (positive_segment<SEG>()) {
                vInt bits1 = as<vInt>(x1);
                vInt bits2 = as<vInt>(x2);
                v_if(bits1 >= 0 && bits1 != 0) { r1 = tmp1; }
                v_endif;
                v_if(bits2 >= 0 && bits2 != 0) { r2 = tmp2; }
                v_endif;
            } else {
                vFloat b = lut_fetch<SEG>();
                v_if(x1 >= b) { r1 = tmp1; }
                v_endif;
                v_if(x2 >= b) { r2 = tmp2; }
                v_endif;
            }
            cascade_dual<SEG + 1>(x1, x2, r1, r2);
        }
    }
};
}  // namespace ckernel::sfpu::ttpoly::selu
