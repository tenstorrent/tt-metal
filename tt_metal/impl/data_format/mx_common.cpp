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
    if (input_width == output_width) {
        return {input_mantissa, 0};
    }
    const int shift_out = input_width - output_width;
    if (shift_out <= 0) {
        uint32_t mask = (output_width >= 32) ? 0xFFFFFFFFu : ((1u << output_width) - 1u);
        return {input_mantissa & mask, 0};
    }
    uint32_t rounded_bits = input_mantissa & ((1u << shift_out) - 1u);
    uint32_t rounded_msb = (rounded_bits >> (shift_out - 1)) & 0x1u;
    uint32_t rounded_lsbs = rounded_bits & ((1u << (shift_out - 1)) - 1u);
    uint32_t mantissa_lsb = (input_mantissa >> shift_out) & 0x1u;

    uint32_t round_inc = 0;
    if (rounded_msb && rounded_lsbs != 0) {
        round_inc = 1;
    } else if (rounded_msb && rounded_lsbs == 0) {
        round_inc = mantissa_lsb;
    }

    if (output_width == 0) {
        round_inc = (rounded_msb && rounded_lsbs != 0) ? 1 : 0;
        output_width = 1;
    }

    uint32_t new_mantissa = (input_mantissa >> shift_out) + round_inc;
    uint32_t mask = (output_width >= 32) ? 0xFFFFFFFFu : ((1u << output_width) - 1u);
    return {new_mantissa & mask, new_mantissa >> output_width};
}

BlockScaleResult compute_block_scale(
    tt::stl::Span<const float> values, size_t block_offset, const FormatParams& params, bool exp_rnd_en) {
    TT_ASSERT(block_offset + params.block_size <= values.size());

    float max_abs = 0.0f;
    bool all_nan = true;
    bool all_inf_or_zero = true;
    bool any_inf = false;

    for (uint32_t i = 0; i < params.block_size; ++i) {
        float v = values[block_offset + i];
        if (!std::isnan(v)) {
            all_nan = false;
        }
        if (std::isinf(v)) {
            any_inf = true;
        }
        if (!(std::isinf(v) || std::isnan(v) || v == 0.0f)) {
            all_inf_or_zero = false;
        }
        float v_abs = (std::isnan(v) || std::isinf(v)) ? 0.0f : std::abs(v);
        max_abs = std::max(max_abs, v_abs);
    }

    int max_abs_exp = (max_abs == 0.0f) ? 0 : static_cast<int>(std::floor(std::log2(max_abs)));
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

    float scale = std::ldexp(1.0f, shared_exp_adj_for_elem_max);
    return {static_cast<uint8_t>(shared_exp_biased), scale};
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

    if (std::isinf(datum)) {
        if (params.inf_rep != InfNanRep::NotRepresentable) {
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

    if (std::isnan(datum)) {
        if (params.nan_rep == InfNanRep::ExpAllOnesManNonZero) {
            uint32_t exp_all_ones = (1u << params.elem_exp_bits) - 1u;
            uint32_t man = 1u;
            return (exp_all_ones << params.elem_man_bits) | man;
        }
        if (params.nan_rep == InfNanRep::ExpAllOnesManAllOnes) {
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
            uint32_t mant_exp_adjusted_pre = mant_with_hb >> shift;
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

    if (params.inf_rep == InfNanRep::ExpAllOnesManZero) {
        if (elem_exp_biased == exp_all_ones && elem_man == 0) {
            return sign_bit ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
    }

    if (params.nan_rep == InfNanRep::ExpAllOnesManNonZero) {
        if (elem_exp_biased == exp_all_ones && elem_man != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else if (params.nan_rep == InfNanRep::ExpAllOnesManAllOnes) {
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
    uint32_t exp_index = 0;
    for (uint32_t w = 0; w < exp_words; ++w) {
        uint32_t word = 0;
        for (uint32_t b = 0; b < 4; ++b) {
            uint32_t exp_val = exp_index < exps.size() ? exps[exp_index++] : 0;
            word |= (exp_val << (8 * b));
        }
        out.push_back(word);
    }
}

void pack_elem_words(
    const std::vector<uint8_t>& elems, uint32_t elem_words, const FormatParams& params, std::vector<uint32_t>& out) {
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

    if (params.elem_width_storage_bits == 2) {
        uint32_t word = 0;
        for (uint32_t i = 0; i < elem_words * 16; ++i) {
            uint32_t elem_val = elem_index < elems.size() ? (elems[elem_index++] & mask) : 0;
            word |= elem_val << ((i % 16) * 2);
            if ((i % 16) == 15) {
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
        uint32_t word = words[word_offset++];
        for (uint32_t b = 0; b < 4 && out.size() < exp_count; ++b) {
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

    if (params.elem_width_storage_bits == 2) {
        for (uint32_t w = 0; w < elem_words; ++w) {
            uint32_t word = words[word_offset++];
            for (uint32_t b = 0; b < 16 && out.size() < elem_count; ++b) {
                uint32_t elem_val = (word >> (2 * b)) & 0x3u;
                out.push_back(static_cast<uint8_t>(elem_val & mask));
            }
        }
        return;
    }

    TT_THROW("Unsupported MX element storage width {}", params.elem_width_storage_bits);
}

}  // namespace tt::tt_metal::mx
