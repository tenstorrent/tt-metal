// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "models/demos/deepseek_v3_b1/pipeline_manager/bounded_queue.hpp"
#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

struct DecodeStagingEntry {
    int32_t user_id = -1;
    int32_t token_id = EMPTY_TOKEN;
    int32_t position = 0;
};

// Decode token handoff from Reader thread to Writer thread.
// Reader stages completed decode tokens; Writer pops them for re-injection.
// Stale entries from cancelled users are handled by the Writer checking
// cancel_pending after popping — no per-entry invalidation needed.
struct DecodeStaging {
    BoundedQueue<DecodeStagingEntry, MAX_USERS> fifo;

    void stage(int uid, int32_t tok, int32_t pos) { fifo.try_push({.user_id = uid, .token_id = tok, .position = pos}); }

    bool try_pop(int& uid, int32_t& tok, int32_t& pos) {
        DecodeStagingEntry entry;
        if (!fifo.try_pop(entry)) {
            return false;
        }
        uid = entry.user_id;
        tok = entry.token_id;
        pos = entry.position;
        return true;
    }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
