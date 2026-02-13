// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace tt::tt_metal {

// Thread-safe build-once cache for JIT compilation.
//
// Ensures that for a given hash (representing a build target), the build function
// is executed exactly once. Concurrent callers with the same hash block until the
// build completes. Callers arriving after the build is done return immediately.
//
// Used to deduplicate both kernel and firmware JIT builds across threads.
class JitBuildCache {
public:
    static JitBuildCache& inst() {
        static JitBuildCache instance;
        return instance;
    }

    // Execute build_fn exactly once for a given hash.
    // Concurrent callers with the same hash block until the build completes.
    // Returns immediately if hash was already built.
    // If build_fn throws, the entry is removed so subsequent callers can retry.
    void build_once(size_t hash, const std::function<void()>& build_fn);

    // Clear completed entries. After clear(), the next build_once() for any hash
    // will re-execute the build function.
    void clear();

private:
    JitBuildCache() = default;

    enum class State { Building, Built };

    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<size_t, State> entries_;
};

}  // namespace tt::tt_metal
