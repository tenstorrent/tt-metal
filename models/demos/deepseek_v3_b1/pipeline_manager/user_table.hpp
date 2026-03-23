// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// Per-user register file. MAX_USERS entries, parallel arrays indexed by user_id.
struct UserTable {
    std::array<std::atomic<UserState>, MAX_USERS> state;
    std::array<int32_t, MAX_USERS> current_position;
    std::array<int32_t, MAX_USERS> max_new_tokens;
    std::array<int32_t, MAX_USERS> tokens_generated;
    std::array<std::atomic<int32_t>, MAX_USERS> in_flight_count;
    std::array<int32_t, MAX_USERS> prefill_pos;
    std::array<int32_t, MAX_USERS> prefill_chunk_remaining;
    std::array<bool, MAX_USERS> spec_decode_enabled;
    std::array<float, MAX_USERS> temperature;
    std::array<float, MAX_USERS> top_p;
    std::array<int32_t, MAX_USERS> top_k;

    UserTable() { reset_all(); }

    void reset_all() {
        for (int i = 0; i < MAX_USERS; i++) {
            reset(i);
        }
    }

    void reset(int uid) {
        state[uid].store(UserState::INACTIVE, std::memory_order_relaxed);
        current_position[uid] = 0;
        max_new_tokens[uid] = 0;
        tokens_generated[uid] = 0;
        in_flight_count[uid].store(0, std::memory_order_relaxed);
        prefill_pos[uid] = 0;
        prefill_chunk_remaining[uid] = 0;
        spec_decode_enabled[uid] = false;
        temperature[uid] = 1.0f;
        top_p[uid] = 1.0f;
        top_k[uid] = -1;
    }
};

// Fixed-size 2D array storing prompt tokens for users in PREFILL state.
// No concurrent access to the same row: API writes when user is not in prefill queue,
// Writer reads during prefill.
struct PromptTable {
    std::array<std::array<int32_t, MAX_SEQ_LEN>, MAX_USERS> tokens;
    std::array<int32_t, MAX_USERS> lengths;

    PromptTable() { lengths.fill(0); }

    void store(int uid, const int32_t* prompt, int len, int offset = 0) {
        int clamped = std::min(len, MAX_SEQ_LEN - offset);
        for (int i = 0; i < clamped; i++) {
            tokens[uid][offset + i] = prompt[i];
        }
        lengths[uid] = offset + clamped;
    }

    int32_t get_token(int uid, int pos) const { return tokens[uid][pos]; }

    int get_length(int uid) const { return lengths[uid]; }

    void clear(int uid) { lengths[uid] = 0; }
};

// Pending cancellation tracking. Single atomic uint64_t bitmap.
struct CancelBitmap {
    std::atomic<uint64_t> bits{0};

    void mark(int uid) { bits.fetch_or(uint64_t(1) << uid, std::memory_order_release); }

    void clear(int uid) { bits.fetch_and(~(uint64_t(1) << uid), std::memory_order_release); }

    // Atomically clear the bit and return true if it was set.
    // Used as a claim guard so only one thread performs cleanup.
    bool try_clear(int uid) {
        uint64_t mask = uint64_t(1) << uid;
        uint64_t old = bits.fetch_and(~mask, std::memory_order_acq_rel);
        return (old & mask) != 0;
    }

    bool is_set(int uid) const { return (bits.load(std::memory_order_acquire) >> uid) & 1; }

    uint64_t snapshot() const { return bits.load(std::memory_order_acquire); }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
