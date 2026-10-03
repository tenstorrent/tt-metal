// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace sfpi {
template <bool SharedPool, unsigned Iterations = 2>
inline vFloat half_angle_root(vFloat x) {
    // Same shared one- or two-step recurrence; the caller owns the S55 constant pool.
    static_assert(Iterations == 1 || Iterations == 2);
    vFloat out = x;
    v_if(x != 0.0f) {
        vInt half_bits = as<vInt>(as<vUInt>(x) >> 1);
        vFloat approx = as<vFloat>(vConstIntPrgm0 - half_bits);
        vFloat negative_half_x;
        if constexpr (SharedPool) {
            negative_half_x = x * -0.5f;
        } else {
            negative_half_x = x * vConstFloatPrgm1;
        }
        approx = ((approx * approx) * negative_half_x + vConstFloatPrgm2) * approx;
        if constexpr (Iterations == 2) {
            approx = ((approx * approx) * negative_half_x + vConstFloatPrgm2) * approx;
        }
        out = approx * x;
    }
    v_endif;
    return out;
}
template <unsigned DEGREE, bool SharedPool, typename Coefficient>
inline vFloat half_angle_ratio(vFloat coordinate, Coefficient coefficient) {
    vFloat ratio;
    if constexpr (SharedPool) {
        ratio = vConstFloatPrgm1;
    } else {
        ratio = coefficient(DEGREE);
    }
#pragma GCC unroll 12
    for (int k = DEGREE - 1; k >= 0; --k) {
        ratio =
            __builtin_rvtt_sfpmad(ratio.get(), coordinate.get(), vFloat(coefficient(k)).get(), SFPMAD_MOD1_OFFSET_NONE);
    }
    return ratio;
}
template <unsigned Mask, unsigned Value, unsigned Excluded, unsigned Class>
inline void half_angle_negative_nan(vUInt raw_u16, vFloat& y) {
    vUInt exponent_and_sign = raw_u16 & vUInt(Mask);
    v_if(exponent_and_sign == vUInt(Value) && raw_u16 != vUInt(Excluded)) {
        static_assert(Class <= 4u, "half-angle raw terminal requires a nonfinite/zero class");
        if constexpr (Class == 0u) {
            y = std::numeric_limits<float>::quiet_NaN();
        } else if constexpr (Class == 1u) {
            y = std::numeric_limits<float>::infinity();
        } else if constexpr (Class == 2u) {
            y = -std::numeric_limits<float>::infinity();
        } else if constexpr (Class == 3u) {
            y = vFloat(0.0f);
        } else {
            y = setsgn(vFloat(0.0f), 1);
        }
    }
    v_endif;
}
template <typename Config, typename Root, typename Ratio>
inline vFloat half_angle_evaluate(vFloat x, Root root_fn, Ratio ratio_fn) {
    vFloat ax = setsgn(x, 0);
    // Evaluate the arithmetic body without an outer condition frame.  The
    // root recurrence has its own target guard, and nesting it under a false
    // valid-domain predicate is not a verifier-safe Blackhole CC schedule.
    // Invalid lanes are overwritten after all arithmetic temporaries retire.
    vFloat coordinate = (Config::kLimit - ax) * Config::kCoordinateScale;
    vFloat root = root_fn(coordinate);

    // Both zones consume the same verified ratio leaf.  Select its coordinate
    // and multiplicative root first, then issue one shared polynomial body;
    // duplicating that body in two SFPU predicates records both copies.
    v_if(ax < Config::kSplit) {
        coordinate = ax * ax;
        root = ax;
    }
    v_endif;

    vFloat ratio = ratio_fn(coordinate);
    vFloat unsigned_result = root * ratio;
    v_if(ax >= Config::kSplit) {
        if constexpr (Config::kReconstructFromValue) {
            unsigned_result = __builtin_rvtt_sfpmad(
                unsigned_result.get(),
                vFloat(Config::kResultScale).get(),
                vFloat(Config::kResultBias).get(),
                SFPMAD_MOD1_OFFSET_NONE);
        } else {
            vFloat scaled_ratio = Config::kResultScale * ratio;
            unsigned_result = __builtin_rvtt_sfpmad(
                root.get(), scaled_ratio.get(), vFloat(Config::kResultBias).get(), SFPMAD_MOD1_OFFSET_NONE);
        }
    }
    v_endif;
    vFloat signed_result = copysgn(unsigned_result, x);
    if constexpr (Config::kPostcompose) {
        signed_result = __builtin_rvtt_sfpmad(
            signed_result.get(),
            vFloat(Config::kOutputScale).get(),
            vFloat(Config::kOutputBias).get(),
            SFPMAD_MOD1_OFFSET_NONE);
    }
    v_if(ax > Config::kLimit) {
        static_assert(Config::kInvalidClass <= 4u, "half-angle invalid exterior requires a class result");
        if constexpr (Config::kInvalidClass == 0u) {
            signed_result = std::numeric_limits<float>::quiet_NaN();
        } else if constexpr (Config::kInvalidClass == 1u) {
            signed_result = std::numeric_limits<float>::infinity();
        } else if constexpr (Config::kInvalidClass == 2u) {
            signed_result = -std::numeric_limits<float>::infinity();
        } else if constexpr (Config::kInvalidClass == 3u) {
            signed_result = vFloat(0.0f);
        } else {
            signed_result = setsgn(vFloat(0.0f), 1);
        }
    }
    v_endif;
    return signed_result;
}
}  // namespace sfpi
