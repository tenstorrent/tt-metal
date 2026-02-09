// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_cache.hpp"

namespace tt::tt_metal {

void JitBuildCache::build_once(size_t hash, const std::function<void()>& build_fn) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (true) {
        auto it = entries_.find(hash);
        if (it != entries_.end()) {
            if (it->second == State::Built) {
                // Already built, nothing to do.
                return;
            }
            // Another thread is building this hash. Wait and re-check.
            cv_.wait(lock);
            continue;
        }

        // Hash not present -- we are the builder.
        entries_.emplace(hash, State::Building);
        break;
    }

    lock.unlock();

    try {
        build_fn();
    } catch (...) {
        // Build failed -- remove the entry so subsequent callers can retry.
        std::lock_guard<std::mutex> guard(mutex_);
        entries_.erase(hash);
        cv_.notify_all();
        throw;
    }

    {
        std::lock_guard<std::mutex> guard(mutex_);
        // Re-insert as Built. If clear() was called while we were building,
        // the entry was erased. We still mark it Built so that concurrent
        // waiters (who will re-check after clear's notify) see the completed
        // build rather than re-triggering it for the same in-flight work.
        entries_[hash] = State::Built;
    }
    cv_.notify_all();
}

void JitBuildCache::clear() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        entries_.clear();
    }
    // Wake up any threads waiting in build_once(); they will re-check
    // and either find the entry gone (become a new builder) or find
    // it re-inserted as Built by the completing builder thread.
    cv_.notify_all();
}

}  // namespace tt::tt_metal
