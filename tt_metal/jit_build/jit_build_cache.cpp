// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build_cache.hpp"

namespace tt::tt_metal {

bool JitBuildCache::build_once(size_t hash, const std::function<void()>& build_fn) {
    while (true) {
        switch (build_once_no_wait(hash, build_fn)) {
            case BuildOnceStatus::BuiltByCaller: return true;
            case BuildOnceStatus::AlreadyBuilt: return false;
            case BuildOnceStatus::InProgress: {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] {
                    auto it = entries_.find(hash);
                    return it == entries_.end() || it->second == State::Built;
                });
                break;
            }
        }
    }
}

JitBuildCache::BuildOnceStatus JitBuildCache::build_once_no_wait(size_t hash, const std::function<void()>& build_fn) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = entries_.find(hash); it != entries_.end()) {
            if (it->second == State::Built) {
                return BuildOnceStatus::AlreadyBuilt;
            }
            return BuildOnceStatus::InProgress;
        }

        entries_.emplace(hash, State::Building);
    }

    try {
        build_fn();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_.erase(hash);
        }
        cv_.notify_all();
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[hash] = State::Built;
    }
    cv_.notify_all();
    return BuildOnceStatus::BuiltByCaller;
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
