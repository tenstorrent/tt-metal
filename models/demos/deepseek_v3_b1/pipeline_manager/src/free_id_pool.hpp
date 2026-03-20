// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

#include "pipeline_manager/pipeline_manager_types.hpp"

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// Bitmap-based O(1)-ish user ID allocator.
// For MAX_USERS <= 64, a single atomic uint64_t where bit N set = slot N free.
// Allocate via CTZ scan + CAS, free via atomic OR.
struct FreeIdPool {
    static_assert(MAX_USERS <= 64, "FreeIdPool requires MAX_USERS <= 64 for single-word bitmap");

    std::atomic<uint64_t> bitmap;

    FreeIdPool() {
        if constexpr (MAX_USERS == 64) {
            bitmap.store(~uint64_t(0), std::memory_order_relaxed);
        } else {
            bitmap.store((uint64_t(1) << MAX_USERS) - 1, std::memory_order_relaxed);
        }
    }

    int allocate() {
        uint64_t current = bitmap.load(std::memory_order_relaxed);
        while (current != 0) {
            int bit = __builtin_ctzll(current);
            uint64_t mask = uint64_t(1) << bit;
            if (bitmap.compare_exchange_weak(
                    current, current & ~mask, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return bit;
            }
        }
        return -1;
    }

    void free(int uid) { bitmap.fetch_or(uint64_t(1) << uid, std::memory_order_release); }

    int count_free() const { return __builtin_popcountll(bitmap.load(std::memory_order_relaxed)); }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
