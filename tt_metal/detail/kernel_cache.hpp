// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <stdint.h>
#include <unordered_set>
#include <condition_variable>

namespace tt::tt_metal::detail {
struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        return hashes_.find(khash) != hashes_.end();
    }
    bool add(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool ret = false;
        if (hashes_.find(khash) == hashes_.end()) {
            hashes_.insert(khash);
            ret = true;
        }
        return ret;
    }

    void wait_for_bin_generated(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        bin_ready_cv_.wait(lock, [this, khash]() { return generated_bins_.find(khash) != generated_bins_.end(); });
    }

    void add_generated_bin(size_t khash) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            generated_bins_.insert(khash);
        }
        // Notify all waiting threads that a new binary is generated
        bin_ready_cv_.notify_all();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        hashes_.clear();
        generated_bins_.clear();
        // Notify waiting threads that cache has been cleared
        bin_ready_cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable bin_ready_cv_;
    std::unordered_set<size_t> hashes_;
    std::unordered_set<size_t> generated_bins_;
};

/**
 * Clear the current kernel compilation cache.
 *
 * Return value: void
 */
inline void ClearKernelCache() { detail::HashLookup::inst().clear(); }
}  // namespace tt::tt_metal::detail
