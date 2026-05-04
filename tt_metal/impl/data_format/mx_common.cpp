// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mx_common.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include <tt_stl/assert.hpp>

#include "math.hpp"

namespace tt::tt_metal::mx {

RoundResult round_ties_even(uint32_t input_mantissa, int output_width, int input_width) {
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

BlockScaleResult compute_block_scale(
    tt::stl::Span<const float> values, size_t block_offset, const FormatParams& params, bool exp_rnd_en) {
    TT_ASSERT(block_offset + params.block_size <= values.size());

    // Track the maximum finite |v| as raw IEEE-754 bits so we can extract its
    // exponent directly without going through std::log2/std::floor. For two
    // finite non-negative floats, the one with the larger raw bit pattern is
    // also the larger value, so a uint32_t max suffices.
    uint32_t max_abs_bits = 0;
    bool all_nan = true;
    bool all_inf_or_zero = true;
    bool any_inf = false;

    for (uint32_t i = 0; i < params.block_size; ++i) {
        uint32_t bits = std::bit_cast<uint32_t>(values[block_offset + i]);
        uint32_t abs_bits = bits & 0x7FFFFFFFu;
        uint32_t exp_field = abs_bits >> 23;
        uint32_t mant_field = abs_bits & 0x7FFFFFu;
        bool is_nan = (exp_field == 0xFFu) && (mant_field != 0);
        bool is_inf = (exp_field == 0xFFu) && (mant_field == 0);
        bool is_zero = (abs_bits == 0u);

        all_nan = all_nan && is_nan;
        any_inf = any_inf || is_inf;
        all_inf_or_zero = all_inf_or_zero && (is_inf || is_nan || is_zero);

        if (!is_nan && !is_inf && abs_bits > max_abs_bits) {
            max_abs_bits = abs_bits;
        }
    }

    // floor(log2(max_abs)) computed from the IEEE-754 exponent field directly.
    // Subnormals (exp_field == 0) are rare in MX inputs; fall back to std::log2
    // there to preserve the original semantics without complicating the hot path.
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

uint32_t compute_exp_count(uint32_t elem_count, const FormatParams& params) {
    TT_ASSERT(params.block_size > 0, "MX block size must be > 0");
    TT_ASSERT(elem_count % params.block_size == 0, "MX element count must be divisible by block size");
    return elem_count / params.block_size;
}

uint32_t compute_exp_bytes(uint32_t exp_count, uint32_t l1_alignment) { return tt::round_up(exp_count, l1_alignment); }

uint32_t compute_elem_words(uint32_t elem_count, const FormatParams& params) {
    TT_ASSERT(params.elem_width_storage_bits > 0, "MX element storage width must be > 0");
    uint32_t bits = elem_count * static_cast<uint32_t>(params.elem_width_storage_bits);
    return (bits + 31) / 32;
}

TileWordCounts compute_tile_word_counts(uint32_t elem_count, uint32_t l1_alignment, const FormatParams& params) {
    TileWordCounts counts;
    counts.exp_count = compute_exp_count(elem_count, params);
    counts.exp_bytes = compute_exp_bytes(counts.exp_count, l1_alignment);
    counts.exp_words = counts.exp_bytes / 4;
    counts.elem_words = compute_elem_words(elem_count, params);
    return counts;
}

uint32_t convert_to_mx_elem_bits(float datum, const FormatParams& params) {
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

        int mant_width = params.elem_man_bits;
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

float convert_from_mx_elem_bits(uint32_t elem_bits, uint8_t scale_exp_biased, const FormatParams& params) {
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
                      ? std::ldexp(1.0f, elem_exp_unbiased)
                      : std::ldexp(1.0f, params.elem_exp_min_unbiased);
    float man_f = static_cast<float>(elem_man_with_int_bit) / static_cast<float>(elem_man_one);

    return sign_f * exp_f * man_f;
}

void pack_exp_words(const std::vector<uint8_t>& exps, uint32_t exp_words, std::vector<uint32_t>& out) {
    out.reserve(out.size() + exp_words);
    const size_t exps_length = exps.size();
    for (uint32_t w = 0; w < exp_words; ++w) {
        uint32_t word = 0;
        const size_t base = size_t(w) * 4;
        const size_t take = (exps_length > base) ? std::min<size_t>(4, exps_length - base) : 0;
        for (size_t b = 0; b < take; ++b) {
            word |= uint32_t(exps[base + b]) << (8 * b);
        }
        out.push_back(word);
    }
}

void pack_elem_words(
    const std::vector<uint8_t>& elems, uint32_t elem_words, const FormatParams& params, std::vector<uint32_t>& out) {
    TT_ASSERT(params.elem_width_storage_bits >= params.elem_width_bits);
    uint32_t elem_index = 0;
    uint32_t mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);

    if (params.elem_width_storage_bits == 8) {
        uint32_t shift = static_cast<uint32_t>(params.elem_width_storage_bits - params.elem_width_bits);
        uint32_t word = 0;
        for (uint32_t i = 0; i < elem_words * 4; ++i) {
            uint32_t elem_val = elem_index < elems.size() ? (elems[elem_index++] & mask) : 0;
            word |= elem_val << (shift + (i % 4) * 8);
            if ((i % 4) == 3) {
                out.push_back(word);
                word = 0;
            }
        }
        return;
    }

    if (params.elem_width_storage_bits == 4) {
        uint32_t word = 0;
        for (uint32_t i = 0; i < elem_words * 8; ++i) {
            uint32_t elem_val = elem_index < elems.size() ? (elems[elem_index++] & mask) : 0;
            word |= elem_val << ((i % 8) * 4);
            if ((i % 8) == 7) {
                out.push_back(word);
                word = 0;
            }
        }
        return;
    }

    TT_THROW("Unsupported MX element storage width {}", params.elem_width_storage_bits);
}

void unpack_exp_words(
    tt::stl::Span<const uint32_t> words,
    size_t& word_offset,
    uint32_t exp_words,
    uint32_t exp_count,
    std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(exp_count);
    for (uint32_t w = 0; w < exp_words; ++w) {
        const uint32_t word = words[word_offset++];
        const size_t base = size_t(w) * 4;
        const size_t take = (exp_count > base) ? std::min<size_t>(4, exp_count - base) : 0;
        for (size_t b = 0; b < take; ++b) {
            out.push_back(static_cast<uint8_t>((word >> (8 * b)) & 0xFFu));
        }
    }
}

void unpack_elem_words(
    tt::stl::Span<const uint32_t> words,
    size_t& word_offset,
    uint32_t elem_words,
    uint32_t elem_count,
    const FormatParams& params,
    std::vector<uint8_t>& out) {
    TT_ASSERT(params.elem_width_storage_bits >= params.elem_width_bits);
    out.clear();
    out.reserve(elem_count);

    uint32_t mask = (params.elem_width_bits >= 32) ? 0xFFFFFFFFu : ((1u << params.elem_width_bits) - 1u);

    if (params.elem_width_storage_bits == 8) {
        uint32_t shift = static_cast<uint32_t>(params.elem_width_storage_bits - params.elem_width_bits);
        for (uint32_t w = 0; w < elem_words; ++w) {
            uint32_t word = words[word_offset++];
            for (uint32_t b = 0; b < 4 && out.size() < elem_count; ++b) {
                uint32_t elem_val = (word >> (8 * b)) & 0xFFu;
                elem_val = (elem_val >> shift) & mask;
                out.push_back(static_cast<uint8_t>(elem_val));
            }
        }
        return;
    }

    if (params.elem_width_storage_bits == 4) {
        for (uint32_t w = 0; w < elem_words; ++w) {
            uint32_t word = words[word_offset++];
            for (uint32_t b = 0; b < 8 && out.size() < elem_count; ++b) {
                uint32_t elem_val = (word >> (4 * b)) & 0xFu;
                out.push_back(static_cast<uint8_t>(elem_val & mask));
            }
        }
        return;
    }

    TT_THROW("Unsupported MX element storage width {}", params.elem_width_storage_bits);
}

}  // namespace tt::tt_metal::mx
