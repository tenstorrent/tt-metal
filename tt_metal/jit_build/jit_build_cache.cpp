// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        entries_[hash] = State::Built;
    }
    cv_.notify_all();
}

void JitBuildCache::clear() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // Only erase Built entries. In-flight builds (Building) are preserved so
        // that waiters continue to wait for the current builder rather than
        // starting a duplicate build for the same hash.
        std::erase_if(entries_, [](const auto& p) { return p.second == State::Built; });
    }
    cv_.notify_all();
}

}  // namespace tt::tt_metal
