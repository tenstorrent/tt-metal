// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "sfpi.h"

// Include polynomial_horner.h before this transport helper.
namespace sfpi {
// The selected one-sided terminal uses ordered comparison on WH and the
// original SFPI maximum on BH. Both canonical and compact paths call here.
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

template <bool Ordered, uint32_t BoundBits>
__attribute__((always_inline)) inline vFloat dense_lower_clamp(vFloat input, vUInt raw) {
    vUInt raw_sign = raw & vUInt(0x8000u);
    v_if(raw_sign == vUInt(0x8000u)) { input = setsgn(input, 1); }
    v_endif;
    return dense_lower_clamp<Ordered, BoundBits>(input);
}

template <uint32_t ConstantBits>
__attribute__((always_inline)) inline void dense_negative_nan_constant(vUInt raw, vFloat& result) {
    vUInt delta = (raw ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(delta == 0u) {
        vUInt mantissa = raw & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = __builtin_bit_cast(float, ConstantBits); }
        v_endif;
    }
    v_endif;
}

template <typename Config>
struct dense_polynomial_transport {
    template <uint32_t S>
    __attribute__((always_inline)) static inline void park_boundaries() {
        if constexpr (S < Config::kSegments) {
            constexpr int row = Config::park_row(S);
            if constexpr (row >= 0) {
                dst_reg[Config::kRowBase + row].template mode<DataLayout::F32>() = vFloat(Config::lut(S));
            }
            park_boundaries<S + 1>();
        }
    }

    template <uint32_t S, uint32_t K>
    __attribute__((always_inline)) static inline void park_seg_coeffs() {
        if constexpr (K <= Config::degree(S)) {
            constexpr uint32_t idx = Config::kCoefficientOffset + S * Config::kCoefficientsPerSegment + K;
            constexpr int row = Config::park_row(idx);
            if constexpr (row >= 0) {
                dst_reg[Config::kRowBase + row].template mode<DataLayout::F32>() = vFloat(Config::lut(idx));
            }
            park_seg_coeffs<S, K + 1>();
        }
    }

    template <uint32_t S>
    __attribute__((always_inline)) static inline void park_coeffs() {
        if constexpr (S < Config::kSegments) {
            park_seg_coeffs<S, 0>();
            park_coeffs<S + 1>();
        }
    }

    template <uint32_t IDX>
    __attribute__((always_inline)) static inline vFloat lut_fetch() {
        constexpr int row = Config::park_row(IDX);
        if constexpr (row >= 0) {
            return dst_reg[Config::kRowBase + row].template mode<DataLayout::F32>();
        } else {
            return vFloat(Config::lut(IDX));
        }
    }

    // --- dual Horner over one segment (identical op sequence to
    // --- eval_polynomial_dual: shared coefficient register, MAD(r, x, c) rungs).
    // A verifier-bound selected reconstruction may request the single-row twin
    // below when an auxiliary tail leaf would push the dual body's six persistent
    // lane values beyond Blackhole's eight SFPU LREGs.  Both schedules use these
    // exact parked fetchers, coefficient bytes, segment predicates, and Horner
    // order; only cross-lane interleaving changes.
    template <uint32_t BASE>
    struct segment_coefficients {
        template <uint32_t INDEX>
        __attribute__((always_inline)) inline vFloat load() const {
            return lut_fetch<BASE + INDEX>();
        }
    };

    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void eval_seg_single(vFloat x, vFloat& r) {
        constexpr uint32_t DEG = Config::degree(SEG);
        constexpr uint32_t BASE = Config::kCoefficientOffset + SEG * Config::kCoefficientsPerSegment;
        eval_polynomial_transport<DEG>(segment_coefficients<BASE>{}, x, r);
    }

    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void cascade_single(vFloat x, vFloat& r) {
        if constexpr (SEG < Config::kSegments) {
            vFloat tmp;
            eval_seg_single<SEG>(x, tmp);
            v_if(x >= lut_fetch<SEG>()) { r = tmp; }
            v_endif;
            cascade_single<SEG + 1>(x, r);
        }
    }

    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void eval_seg_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
        constexpr uint32_t DEG = Config::degree(SEG);
        constexpr uint32_t BASE = Config::kCoefficientOffset + SEG * Config::kCoefficientsPerSegment;
        eval_polynomial_transport<DEG>(segment_coefficients<BASE>{}, x1, x2, r1, r2);
    }

    // --- segment cascade (identical predicated-move structure to
    // --- unroll_segment_dual; boundary compare reads the parked row).
    template <uint32_t SEG>
    __attribute__((always_inline)) static inline void cascade_dual(vFloat x1, vFloat x2, vFloat& r1, vFloat& r2) {
        if constexpr (SEG < Config::kSegments) {
            vFloat tmp1, tmp2;
            eval_seg_dual<SEG>(x1, x2, tmp1, tmp2);
            vFloat b = lut_fetch<SEG>();
            v_if(x1 >= b) { r1 = tmp1; }
            v_endif;
            v_if(x2 >= b) { r2 = tmp2; }
            v_endif;
            cascade_dual<SEG + 1>(x1, x2, r1, r2);
        }
    }
};
}  // namespace sfpi
