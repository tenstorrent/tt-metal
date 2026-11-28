// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly::celu {
using ::sfpi::DataLayout;
using ::sfpi::vFloat;
template <bool Ordered, uint32_t BoundBits>
__attribute__((always_inline)) inline vFloat dense_lower_clamp(vFloat input) {
    constexpr float bound = __builtin_bit_cast(float, BoundBits);
    return sfpi::max(input, bound);
}

template <typename Config>
struct dense_polynomial_transport {
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
    // Config::kPositiveSegment is false when no segment starts there.
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
            {
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
}  // namespace ckernel::sfpu::ttpoly::celu
