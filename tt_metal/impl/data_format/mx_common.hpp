// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::mx {

enum class InfNanRepresentation : uint8_t {
    NotRepresentable = 0,
    ExpAllOnesManZero,
    ExpAllOnesManNonZero,
    ExpAllOnesManAllOnes,
};

struct FormatParams {
    uint32_t block_size = 32;
    int scale_bias = 0x7F;
    uint32_t elem_exp_bits = 0;
    uint32_t elem_man_bits = 0;
    int elem_exp_bias = 0;
    int elem_exp_max_unbiased = 0;
    int elem_exp_min_unbiased = 0;
    int elem_exp_subnorm_encoding = 0;
    uint32_t elem_man_max = 0;
    uint32_t elem_width_bits = 0;
    uint32_t elem_width_storage_bits = 0;
    bool sat_supported = false;
    uint32_t elem_sat_pos_bits = 0;
    uint32_t elem_sat_neg_bits = 0;
    InfNanRepresentation inf_rep = InfNanRepresentation::NotRepresentable;
    InfNanRepresentation nan_rep = InfNanRepresentation::NotRepresentable;
    // MX integer-format (MxInt2/4/8) parameters. Unused by the floating-point MX
    // formats (which leave is_integer=false). When is_integer is true the element
    // is a signed two's-complement integer of width elem_width_bits with an
    // implicit scale of 1/elem_int_scale; the shared block scale (E8M0) is applied
    // on top. elem_int_max is the symmetric clamp magnitude.
    bool is_integer = false;
    uint32_t elem_int_scale = 0;  // round(scaled * elem_int_scale): 64 (MxInt8), 4 (MxInt4), 1 (MxInt2)
    int elem_int_max = 0;         // symmetric clamp: 127 (MxInt8), 7 (MxInt4), 1 (MxInt2)
};

struct RoundResult {
    uint32_t mantissa = 0;
    uint32_t overflow = 0;
};

struct BlockScaleResult {
    uint8_t shared_exp_biased = 0;
    // Unbiased shared exponent. Effective scale is 2^shared_exp_adj — kept as
    // an int so callers can ldexp directly and avoid building/dividing by a
    // float power of two.
    int shared_exp_adj = 0;
};

struct TileWordCounts {
    uint32_t exp_count = 0;
    uint32_t exp_bytes = 0;
    uint32_t exp_words = 0;
    uint32_t elem_words = 0;
};

// 2^k for integer k, avoiding the libm overhead of std::ldexp on the hot
// per-element MX pack/unpack paths. Fast path bit-constructs a normal float
// for k in [-126, 127]; the rare edges (subnormal scale, E8M0 NaN scale)
// defer to std::ldexp so behavior matches at boundaries.
inline float pow2_f32(int k) {
    if (k >= -126 && k <= 127) {
        return std::bit_cast<float>(static_cast<uint32_t>(127 + k) << 23);
    }
    return std::ldexp(1.0f, k);
}

// Inlined so that call sites passing a constexpr FormatParams (e.g.
// kMxFp6RParams) can constant-propagate field accesses, mask widths, and
// inf/nan/sat branches. convert_to_mxfp_elem_bits inlines this on the per-
// element hot path.
constexpr RoundResult round_ties_even(uint32_t input_mantissa, int output_width, int input_width = 23) {
    if (output_width < 0) {
        return {0, 0};
    }
    auto mask_of = [](int w) -> uint32_t { return (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1u); };
    // No rounding needed: output is at least as wide as input.
    if (output_width >= input_width) {
        return {input_mantissa & mask_of(output_width), 0};
    }
    const int shift_out = input_width - output_width;  // 1..32
    const uint32_t shifted = (shift_out >= 32) ? 0u : (input_mantissa >> shift_out);
    const uint32_t round_bit = (input_mantissa >> (shift_out - 1)) & 1u;
    const uint32_t sticky = input_mantissa & mask_of(shift_out - 1);  // 0 when shift_out == 1
    const uint32_t lsb = shifted & 1u;                                // 0 when output_width == 0
    // Round up iff round bit is set AND (any sticky bit OR kept LSB is 1).
    const uint32_t round_inc = round_bit & ((sticky != 0 ? 1u : 0u) | lsb);
    const uint32_t new_mantissa = shifted + round_inc;
    if (output_width == 0) {
        // Whole rounded value is overflow; mantissa field is empty.
        return {0, new_mantissa};
    }
    return {new_mantissa & mask_of(output_width), new_mantissa >> output_width};
}

// Derives the BlockScaleResult from already-computed per-block stats so the
// shared exponent can be obtained from a single pass over the values (see
// update_block_stats in mx_tile_pack.hpp).
inline BlockScaleResult finalize_block_scale(
    uint32_t max_abs_bits,
    bool all_nan,
    bool all_inf_or_zero,
    bool any_inf,
    const FormatParams& params,
    bool exp_rnd_en = false) {
    int max_abs_exp = 0;
    if (max_abs_bits != 0) {
        uint32_t exp_field = max_abs_bits >> 23;
        if (exp_field != 0) {
            max_abs_exp = static_cast<int>(exp_field) - 127;
        } else {
            float max_abs = std::bit_cast<float>(max_abs_bits);
            max_abs_exp = static_cast<int>(std::floor(std::log2(max_abs)));
        }
    }
    int shared_exp = max_abs_exp;
    int shared_exp_adj_for_elem_max = -127;

    if (shared_exp - params.elem_exp_max_unbiased >= -127) {
        if (exp_rnd_en && shared_exp < 127) {
            shared_exp += 1;
        }
        shared_exp_adj_for_elem_max = shared_exp - params.elem_exp_max_unbiased;
    } else if (exp_rnd_en && shared_exp < 127) {
        shared_exp_adj_for_elem_max = -127;
    }

    int shared_exp_biased = shared_exp_adj_for_elem_max + params.scale_bias;
    if (all_nan) {
        shared_exp_biased = 0xFF;
    } else if (all_inf_or_zero && any_inf) {
        shared_exp_biased = 0xFE;
    }

    return {static_cast<uint8_t>(shared_exp_biased), shared_exp_adj_for_elem_max};
}

TileWordCounts compute_tile_word_counts(uint32_t elem_count, const FormatParams& params);

inline uint32_t convert_to_mxfp_elem_bits(float datum, const FormatParams& params) {
    uint32_t ui32 = std::bit_cast<uint32_t>(datum);
    uint8_t sign = static_cast<uint8_t>((ui32 >> 31) & 0x1u);
    uint32_t fp32_exp_biased = (ui32 >> 23) & 0xFFu;
    uint32_t fp32_mant = ui32 & 0x7FFFFFu;

    bool fp32_is_zero = (fp32_exp_biased == 0) && (fp32_mant == 0);
    if (fp32_is_zero) {
        uint32_t elem_sign_bit = static_cast<uint32_t>(sign) << (params.elem_man_bits + params.elem_exp_bits);
        return elem_sign_bit;
    }

    // Use the already-extracted exponent/mantissa fields rather than std::isinf
    // and std::isnan, which on most libc implementations involve an extra
    // function call (and may go through fpclassify).
    bool fp32_is_inf = (fp32_exp_biased == 0xFFu) && (fp32_mant == 0u);
    bool fp32_is_nan = (fp32_exp_biased == 0xFFu) && (fp32_mant != 0u);

    if (fp32_is_inf) {
        if (params.inf_rep != InfNanRepresentation::NotRepresentable) {
            uint32_t exp_all_ones = (1u << params.elem_exp_bits) - 1u;
            uint32_t man = 0;
            uint32_t sign_bit = static_cast<uint32_t>(sign) << (params.elem_man_bits + params.elem_exp_bits);
            return sign_bit | (exp_all_ones << params.elem_man_bits) | man;
        }
        if (params.sat_supported) {
            return sign ? params.elem_sat_neg_bits : params.elem_sat_pos_bits;
        }
        return 0;
    }

    if (fp32_is_nan) {
        if (params.nan_rep == InfNanRepresentation::ExpAllOnesManNonZero) {
            uint32_t exp_all_ones = (1u << params.elem_exp_bits) - 1u;
            uint32_t man = 1u;
            return (exp_all_ones << params.elem_man_bits) | man;
        }
        if (params.nan_rep == InfNanRepresentation::ExpAllOnesManAllOnes) {
            uint32_t exp_all_ones = (1u << params.elem_exp_bits) - 1u;
            uint32_t man = (1u << params.elem_man_bits) - 1u;
            return (exp_all_ones << params.elem_man_bits) | man;
        }
        return 0;
    }

    uint32_t mant_exp_adjusted = 0;
    uint32_t elem_exp_biased = 0;

    if (fp32_exp_biased != 0) {
        int elem_exp_unbiased = static_cast<int>(fp32_exp_biased) - 127;
        TT_ASSERT(elem_exp_unbiased <= params.elem_exp_max_unbiased, "MX element exponent out of range after scaling");

        int mant_width = static_cast<int>(params.elem_man_bits);
        auto [mant_round, exp_inc] = round_ties_even(fp32_mant, mant_width, 23);
        uint32_t elem_mant_shifted = mant_round;
        int elem_exp_unbiased_adj = elem_exp_unbiased + static_cast<int>(exp_inc);
        mant_exp_adjusted = elem_mant_shifted;

        if (elem_exp_unbiased_adj < params.elem_exp_min_unbiased) {
            elem_exp_unbiased_adj -= static_cast<int>(exp_inc);
            uint32_t mant_with_hb = fp32_mant | (1u << 23);
            int shift = std::abs(params.elem_exp_min_unbiased - elem_exp_unbiased_adj);
            // Right-shifting a uint32_t by >= 32 is undefined behaviour.
            // For very small inputs (|v| << block max) shift can exceed the 24
            // significant bits of mant_with_hb, in which case the result is
            // unconditionally zero.
            uint32_t mant_exp_adjusted_pre = (shift >= 24) ? 0u : (mant_with_hb >> shift);
            auto [mant_round_sub, exp_inc_sub] = round_ties_even(mant_exp_adjusted_pre, mant_width, 24);
            mant_exp_adjusted = mant_round_sub;
            elem_exp_unbiased_adj =
                -params.elem_exp_bias + params.elem_exp_subnorm_encoding + static_cast<int>(exp_inc_sub);
        }

        if ((elem_exp_unbiased_adj > params.elem_exp_max_unbiased) ||
            (elem_exp_unbiased_adj == params.elem_exp_max_unbiased && elem_mant_shifted > params.elem_man_max)) {
            elem_exp_unbiased_adj = params.elem_exp_max_unbiased;
            mant_exp_adjusted = params.elem_man_max;
        }

        elem_exp_biased = static_cast<uint32_t>(elem_exp_unbiased_adj + params.elem_exp_bias);
    } else {
        mant_exp_adjusted = 0;
        elem_exp_biased = 0;
    }

    uint32_t elem_bits = (elem_exp_biased << params.elem_man_bits) | (mant_exp_adjusted & params.elem_man_max);
    if (sign) {
        elem_bits |= static_cast<uint32_t>(1u << (params.elem_man_bits + params.elem_exp_bits));
    }
    return elem_bits;
}

inline float convert_from_mxfp_elem_bits(uint32_t elem_bits, uint8_t scale_exp_biased, const FormatParams& params) {
    if (scale_exp_biased == 0xFF) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    uint32_t elem_man_mask = (1u << params.elem_man_bits) - 1u;
    uint32_t elem_man = elem_bits & elem_man_mask;
    uint32_t sign_bit = elem_bits >> (params.elem_man_bits + params.elem_exp_bits);
    uint32_t elem_exp_biased = (elem_bits >> params.elem_man_bits) & ((1u << params.elem_exp_bits) - 1u);

    uint32_t exp_all_ones = (params.elem_exp_bits == 0) ? 0 : ((1u << params.elem_exp_bits) - 1u);
    uint32_t man_all_ones = elem_man_mask;

    if (params.inf_rep == InfNanRepresentation::ExpAllOnesManZero) {
        if (elem_exp_biased == exp_all_ones && elem_man == 0) {
            return sign_bit ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
    }

    if (params.nan_rep == InfNanRepresentation::ExpAllOnesManNonZero) {
        if (elem_exp_biased == exp_all_ones && elem_man != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else if (params.nan_rep == InfNanRepresentation::ExpAllOnesManAllOnes) {
        if (elem_exp_biased == exp_all_ones && elem_man == man_all_ones) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    uint32_t elem_man_one = 1u << params.elem_man_bits;
    bool include_hidden_bit = (elem_exp_biased == static_cast<uint32_t>(params.elem_exp_subnorm_encoding));
    uint32_t elem_man_with_int_bit = include_hidden_bit ? elem_man : (elem_man | elem_man_one);

    int elem_exp_unbiased = 0;
    if (!(elem_exp_biased == 0 && elem_man_with_int_bit == 0)) {
        elem_exp_unbiased = static_cast<int>(elem_exp_biased) - params.elem_exp_bias;
    }

    float sign_f = (sign_bit != 0) ? -1.0f : 1.0f;
    float exp_f = (elem_exp_biased != static_cast<uint32_t>(params.elem_exp_subnorm_encoding))
                      ? pow2_f32(elem_exp_unbiased)
                      : pow2_f32(params.elem_exp_min_unbiased);
    float man_f = static_cast<float>(elem_man_with_int_bit) / static_cast<float>(elem_man_one);

    return sign_f * exp_f * man_f;
}

// Round to nearest, ties to even, computed purely from the value so the result
// is deterministic regardless of the ambient FP rounding mode (the MX packer is
// the reference the device output is compared against). Used by the integer MX
// element quantizer below.
inline float rint_ties_even(float x) {
    const float floor_x = std::floor(x);
    const float frac = x - floor_x;
    if (frac < 0.5f) {
        return floor_x;
    }
    if (frac > 0.5f) {
        return floor_x + 1.0f;
    }
    // Exact tie: round to the even neighbour. floor_x is integral; its low bit
    // is its parity (true for both signs under two's complement). The int cast
    // is safe because callers only pass finite, block-scaled magnitudes
    // (|x| <= ~128: scaled in [-2, 2) times elem_int_scale <= 64; NaN/Inf are
    // handled before this helper), so floor_x fits in int.
    const int floor_i = static_cast<int>(floor_x);
    return (floor_i & 1) == 0 ? floor_x : floor_x + 1.0f;
}

// Quantize a single (already block-scaled) value into an MxInt element: a signed
// two's-complement integer of width `params.elem_width_bits`, returned masked
// into the low bits. Mirrors the validated tt-llk golden
// (_mxint_block_scale_and_quantize):
//   - NaN element -> 0 (MxInt has no NaN representation)
//   - +/-Inf element -> saturate to +/-elem_int_max
//   - otherwise int = clamp(rint_ties_even(scaled * elem_int_scale), +/-elem_int_max)
// `scaled` is the input value already divided by the block scale (2^shared_exp).
inline uint32_t convert_to_mxint_elem_bits(float scaled, const FormatParams& params) {
    const uint32_t elem_mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);
    const int elem_max = params.elem_int_max;

    uint32_t bits = std::bit_cast<uint32_t>(scaled);
    uint32_t abs_bits = bits & 0x7FFFFFFFu;
    uint32_t exp_field = abs_bits >> 23;
    uint32_t mant_field = abs_bits & 0x7FFFFFu;
    bool is_nan = (exp_field == 0xFFu) && (mant_field != 0);
    bool is_inf = (exp_field == 0xFFu) && (mant_field == 0);

    int int_val;
    if (is_nan) {
        int_val = 0;
    } else if (is_inf) {
        int_val = (bits >> 31) ? -elem_max : elem_max;
    } else {
        float q = rint_ties_even(scaled * static_cast<float>(params.elem_int_scale));
        if (q > static_cast<float>(elem_max)) {
            int_val = elem_max;
        } else if (q < static_cast<float>(-elem_max)) {
            int_val = -elem_max;
        } else {
            int_val = static_cast<int>(q);
        }
    }

    return static_cast<uint32_t>(int_val) & elem_mask;
}

// Decode a single MxInt element field (width `params.elem_width_bits`, two's
// complement) into the value before the block scale is applied:
// int_val / elem_int_scale. The caller multiplies by the block scale
// 2^(scale-bias). Unlike the floating-point decoder this takes no scale byte:
// the NaN block scale (0xFF) is handled by the caller (zeros the block), so a
// 0 * 2^(0xFF-bias) = 0 * inf = NaN trap never reaches here.
inline float convert_from_mxint_elem_bits(uint32_t elem_bits, const FormatParams& params) {
    const uint32_t width = params.elem_width_bits;
    const uint32_t sign_bit = 1u << (width - 1);
    int int_val = static_cast<int>(elem_bits & ((1u << width) - 1u));
    if (elem_bits & sign_bit) {
        int_val -= static_cast<int>(1u << width);  // sign-extend two's complement
    }
    return static_cast<float>(int_val) / static_cast<float>(params.elem_int_scale);
}

}  // namespace tt::tt_metal::mx
