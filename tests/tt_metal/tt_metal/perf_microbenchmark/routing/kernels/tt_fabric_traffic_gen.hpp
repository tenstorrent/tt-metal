// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define is_power_of_2(x) (((x) > 0) && (((x) & ((x) - 1)) == 0))

inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Derive a 16B-aligned payload size in [16, max_payload_size_bytes] from a PRNG state.
// Sender and receiver must call this with the same seed value after the same prng_next step.
inline uint32_t derive_aligned_payload_size_bytes(uint32_t seed, uint32_t max_payload_size_bytes) {
    const uint32_t max_units = max_payload_size_bytes / 16;
    return ((seed % max_units) + 1) * 16;
}

inline void fill_packet_data(tt_l1_ptr uint32_t* start_addr, uint32_t num_words, uint32_t start_val) {
    constexpr uint32_t packet_word_stride_words = tt::tt_fabric::PACKET_WORD_SIZE_BYTES / sizeof(uint32_t);
    tt_l1_ptr uint32_t* addr = start_addr + (packet_word_stride_words - 1);
    for (uint32_t i = 0; i < num_words; i++) {
        *addr = start_val++;
        addr += packet_word_stride_words;
    }
}

inline bool check_packet_data(
    tt_l1_ptr uint32_t* start_addr,
    uint32_t num_words,
    uint32_t start_val,
    uint32_t& mismatch_addr,
    uint32_t& mismatch_val,
    uint32_t& expected_val) {
    constexpr uint32_t packet_word_stride_words = tt::tt_fabric::PACKET_WORD_SIZE_BYTES / sizeof(uint32_t);
    tt_l1_ptr uint32_t* addr = start_addr + (packet_word_stride_words - 1);
    invalidate_l1_cache();
    for (uint32_t i = 0; i < num_words; i++) {
        if (*addr != start_val) {
            mismatch_addr = reinterpret_cast<uint32_t>(addr);
            mismatch_val = *addr;
            expected_val = start_val;
            return false;
        }
        start_val++;
        addr += packet_word_stride_words;
    }
    return true;
}
