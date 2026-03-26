// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>

#include "models/demos/deepseek_v3_b1/pipeline_manager/bounded_queue.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

struct DecodeStagingEntry {
    int32_t user_id = -1;
    int32_t token_id = EMPTY_TOKEN;
    int32_t position = 0;
    uint32_t generation = 0;
};

// Decode token handoff from Reader thread to Writer thread.
// Each entry carries a generation counter. The Writer skips entries
// whose generation doesn't match the current user generation, handling
// stale entries from cancelled sessions without timing dependencies.
struct DecodeStaging {
    static constexpr int FIFO_CAPACITY = MAX_USERS * 4;
    BoundedQueue<DecodeStagingEntry, FIFO_CAPACITY> fifo;
    std::array<std::atomic<uint32_t>, MAX_USERS> generation;

    DecodeStaging() {
        for (int i = 0; i < MAX_USERS; i++) {
            generation[i].store(0, std::memory_order_relaxed);
        }
    }

    void advance_generation(int uid) { generation[uid].fetch_add(1, std::memory_order_release); }

    void stage(int uid, int32_t tok, int32_t pos) {
        uint32_t gen = generation[uid].load(std::memory_order_relaxed);
        DecodeStagingEntry entry{.user_id = uid, .token_id = tok, .position = pos, .generation = gen};
        if (!fifo.try_push(entry)) {
            std::cerr << "FATAL: decode_staging FIFO full — loopback token dropped for uid " << uid << std::endl;
        }
    }

    // Pops the next entry, silently skipping stale entries from old generations.
    bool try_pop(int& uid, int32_t& tok, int32_t& pos) {
        DecodeStagingEntry entry;
        while (fifo.try_pop(entry)) {
            uint32_t current_gen = generation[entry.user_id].load(std::memory_order_acquire);
            if (entry.generation == current_gen) {
                uid = entry.user_id;
                tok = entry.token_id;
                pos = entry.position;
                return true;
            }
        }
        return false;
    }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
