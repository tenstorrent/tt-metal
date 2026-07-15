// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _PATTERNS_H
#define _PATTERNS_H

#include <stdint.h>

enum {
    DRAM_PATTERN_CHECKERBOARD,
    DRAM_PATTERN_RANDOM,
    DRAM_PATTERN_MARCHING_ONES,
    DRAM_PATTERN_MARCHING_ZEROES,
    DRAM_PATTERN_REVERSIBLE_RANDOM,
    DRAM_PATTERN_RANDOM_XOSHIRO128PP,
    DRAM_PATTERN_TOGGLE_BITS,
    DRAM_PATTERN_SATURATION,
    DRAM_PATTERN_MARCHING_ONE_BITS,
    DRAM_PATTERN_MARCHING_ZERO_BITS,
    DRAM_PATTERN_COUNTER,
    DRAM_PATTERN_ADDRESS,
    DRAM_PATTERN_BYTEWISE_SSN,
};

[[maybe_unused]]
static uint32_t num_passes_for_pattern(uint32_t pattern_id) {
    switch (pattern_id) {
        case DRAM_PATTERN_CHECKERBOARD: return 2u;
        case DRAM_PATTERN_MARCHING_ONES: return 32u;
        case DRAM_PATTERN_MARCHING_ZEROES: return 32u;
        case DRAM_PATTERN_TOGGLE_BITS: return 2u;
        case DRAM_PATTERN_SATURATION: return 2u;
        case DRAM_PATTERN_MARCHING_ONE_BITS: return 33u;
        case DRAM_PATTERN_MARCHING_ZERO_BITS: return 33u;
        default: return 1u;
    }
}

[[maybe_unused]]
static const char* pattern_name(uint32_t pattern_id) {
    switch (pattern_id) {
        case DRAM_PATTERN_COUNTER: return "Counter";
        case DRAM_PATTERN_CHECKERBOARD: return "Checkerboard";
        case DRAM_PATTERN_ADDRESS: return "Address";
        case DRAM_PATTERN_MARCHING_ONES: return "MarchingOnes";
        case DRAM_PATTERN_MARCHING_ZEROES: return "MarchingZeroes";
        case DRAM_PATTERN_MARCHING_ONE_BITS: return "MarchingOneBits";
        case DRAM_PATTERN_MARCHING_ZERO_BITS: return "MarchingZeroBits";
        case DRAM_PATTERN_TOGGLE_BITS: return "ToggleBits";
        case DRAM_PATTERN_SATURATION: return "Saturation";
        case DRAM_PATTERN_REVERSIBLE_RANDOM: return "ReversibleRandom";
        case DRAM_PATTERN_RANDOM: return "Random";
        case DRAM_PATTERN_RANDOM_XOSHIRO128PP: return "RandomXoshiro128pp";
        case DRAM_PATTERN_BYTEWISE_SSN: return "ByteWiseSsn";
        default: return "Unknown";
    }
}

struct DramXoshiro128ppState {
    uint32_t s0;
    uint32_t s1;
    uint32_t s2;
    uint32_t s3;
};

static inline uint32_t dram_pattern_random_step(uint32_t value) {
    value ^= (value << 13);
    value ^= (value >> 17);
    value ^= (value << 5);
    return value;
}

static inline uint32_t dram_rotl32(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static inline DramXoshiro128ppState dram_pattern_random_xoshiro128pp_init(uint32_t seed) {
    DramXoshiro128ppState state;
    state.s0 = seed;
    state.s1 = 1u;
    state.s2 = 2u;
    state.s3 = 3u;
    return state;
}

static inline uint32_t dram_pattern_random_xoshiro128pp_next(DramXoshiro128ppState& state) {
    const uint32_t result = dram_rotl32(state.s0 + state.s3, 7) + state.s0;

    const uint32_t t = state.s1 << 9;

    state.s2 ^= state.s0;
    state.s3 ^= state.s1;
    state.s1 ^= state.s2;
    state.s0 ^= state.s3;

    state.s2 ^= t;
    state.s3 = dram_rotl32(state.s3, 11);

    return result;
}

static inline uint32_t dram_pattern_checkerboard(uint32_t pass, uint32_t word_index) {
    uint32_t even_odd_memory_row = word_index & 1u;
    return 0x55555555u << (even_odd_memory_row ^ pass);
}

static inline uint32_t dram_pattern_random(uint32_t seed, uint32_t pass, uint32_t word_index) {
    uint32_t rng_state = seed ^ pass;
    if (rng_state == 0u) {
        rng_state = 1u;
    }

    for (uint32_t i = 0; i <= word_index; ++i) {
        rng_state = dram_pattern_random_step(rng_state);
    }

    return rng_state;
}

static inline uint32_t dram_pattern_marching_ones(uint32_t pass, uint32_t word_index) {
    uint32_t shift = (pass + word_index) & 31u;
    return 1u << shift;
}

static inline uint32_t dram_pattern_marching_zeroes(uint32_t pass, uint32_t word_index) {
    uint32_t shift = (pass + word_index) & 31u;
    return ~(1u << shift);
}

static inline uint32_t dram_pattern_reversible_random(uint32_t word_index) {
    const uint32_t delta = 0x9E3779B9u;
    const uint32_t k0 = 0x796DA607u;
    const uint32_t k1 = 0x78093AF1u;
    const uint32_t k2 = 0xB5B19E1Au;
    const uint32_t k3 = 0x48213943u;
    uint32_t v = word_index;
    uint32_t v1 = 0u;
    uint32_t sum = 0u;

    for (uint32_t i = 0; i < 32u; ++i) {
        sum += delta;
        v += ((v1 << 4) + k0) ^ (v1 + sum) ^ ((v1 >> 5) + k1);
        v1 += ((v << 4) + k2) ^ (v + sum) ^ ((v >> 5) + k3);
    }
    return v;
}

static inline uint32_t dram_pattern_toggle_bits(uint32_t pass, uint32_t word_index) {
    uint32_t pattern = ((word_index & 2u) >> 1) ^ pass;
    return 0u - pattern;
}

static inline uint32_t dram_pattern_saturation(uint32_t pass, uint32_t word_index) {
    const uint32_t channel_width = 2u;
    const uint32_t padding_length = 15u;
    const uint32_t pattern_length_in_words = (padding_length + 1u) * channel_width;
    const uint32_t pattern_index = word_index / pattern_length_in_words;
    const uint32_t test_bit_index = pattern_index % (32u * channel_width);
    const uint32_t test_bit_word = test_bit_index / 32u;
    const uint32_t word_in_pattern = word_index % pattern_length_in_words;
    const uint32_t padding_word = (pass == 0u) ? 0u : 0xFFFFFFFFu;

    if (word_in_pattern < padding_length * channel_width) {
        return padding_word;
    } else {
        const uint32_t test_word_index = word_in_pattern % channel_width;
        if (test_word_index == test_bit_word) {
            return (1u << (test_bit_index % 32u)) ^ padding_word;
        } else {
            return padding_word;
        }
    }
}

static inline uint32_t dram_pattern_marching_one_bits(uint32_t pass) {
    return (pass == 32u) ? 0xFFFFFFFFu : ((1u << pass) - 1u);
}

static inline uint32_t dram_pattern_marching_zero_bits(uint32_t pass) {
    return ~((pass == 32u) ? 0xFFFFFFFFu : ((1u << pass) - 1u));
}

static inline uint32_t dram_pattern_counter(uint32_t seed, uint32_t word_index) { return seed + word_index; }

static inline uint32_t dram_pattern_address(uint32_t repeat_index, uint32_t word_index) {
    return word_index | (repeat_index << 29);
}

static inline uint32_t dram_pattern_bytewise_ssn(uint32_t repeat_index, uint32_t word_index) {
    uint32_t byte_val = (word_index / 2u) & 0xFFu;
    if (repeat_index & 1u) {
        byte_val = (~byte_val) & 0xFFu;
    }

    uint32_t value = 0u;
    for (uint32_t i = 0; i < 4u; ++i) {
        value = (value << 8) | byte_val;
    }

    return (word_index & 1u) ? ~value : value;
}

/* =========================
 * Fast-path buffer fillers
 * ========================= */

static inline void dram_pattern_checkerboard_fill_buffer(uint32_t* dst_words, uint32_t word_count, uint32_t pass) {
    const uint32_t first = (pass & 1u) ? 0xAAAAAAAAu : 0x55555555u;
    const uint32_t second = ~first;

    for (uint32_t i = 0; i < word_count; i += 2u) {
        dst_words[i] = first;
        if (i + 1u < word_count) {
            dst_words[i + 1u] = second;
        }
    }
}

static inline void dram_pattern_counter_fill_buffer(
    uint32_t* dst_words, uint32_t word_count, uint32_t seed, uint32_t base_word_index) {
    uint32_t value = seed + base_word_index;
    for (uint32_t i = 0; i < word_count; ++i) {
        dst_words[i] = value++;
    }
}

static inline void dram_pattern_address_fill_buffer(
    uint32_t* dst_words, uint32_t word_count, uint32_t repeat_index, uint32_t base_word_index) {
    uint32_t value = base_word_index | (repeat_index << 29);
    for (uint32_t i = 0; i < word_count; ++i) {
        dst_words[i] = value++;
    }
}

static inline void dram_pattern_constant_fill_buffer(uint32_t* dst_words, uint32_t word_count, uint32_t value) {
    for (uint32_t i = 0; i < word_count; ++i) {
        dst_words[i] = value;
    }
}

static inline uint32_t dram_pattern_generate(
    uint32_t pattern_id, uint32_t seed, uint32_t pass, uint32_t word_index, uint32_t repeat_index) {
    switch (pattern_id) {
        case DRAM_PATTERN_CHECKERBOARD: return dram_pattern_checkerboard(pass, word_index);
        case DRAM_PATTERN_RANDOM: return dram_pattern_random(seed, pass, word_index);
        case DRAM_PATTERN_MARCHING_ONES: return dram_pattern_marching_ones(pass, word_index);
        case DRAM_PATTERN_MARCHING_ZEROES: return dram_pattern_marching_zeroes(pass, word_index);
        case DRAM_PATTERN_REVERSIBLE_RANDOM: return dram_pattern_reversible_random(word_index);

        case DRAM_PATTERN_RANDOM_XOSHIRO128PP: {
            DramXoshiro128ppState state = dram_pattern_random_xoshiro128pp_init(seed ^ pass);
            for (uint32_t i = 0; i < word_index; ++i) {
                (void)dram_pattern_random_xoshiro128pp_next(state);
            }
            return dram_pattern_random_xoshiro128pp_next(state);
        }

        case DRAM_PATTERN_TOGGLE_BITS: return dram_pattern_toggle_bits(pass, word_index);
        case DRAM_PATTERN_SATURATION: return dram_pattern_saturation(pass, word_index);
        case DRAM_PATTERN_MARCHING_ONE_BITS: return dram_pattern_marching_one_bits(pass);
        case DRAM_PATTERN_MARCHING_ZERO_BITS: return dram_pattern_marching_zero_bits(pass);
        case DRAM_PATTERN_COUNTER: return dram_pattern_counter(seed, word_index);
        case DRAM_PATTERN_ADDRESS: return dram_pattern_address(repeat_index, word_index);
        case DRAM_PATTERN_BYTEWISE_SSN: return dram_pattern_bytewise_ssn(repeat_index, word_index);
        default: return 0u;
    }
}

#endif /* _PATTERNS_H */
