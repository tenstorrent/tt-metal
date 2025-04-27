// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace tt::tt_metal::detail {
struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        return map_key_to_generated_bin_.find(khash) != map_key_to_generated_bin_.end();
    }

    bool add(size_t khash, size_t key) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = map_key_to_generated_bin_.find(key);
        if (it == map_key_to_generated_bin_.end()) {
            map_key_to_generated_bin_[key] = khash;
            return true;
        }
        return false;
    }

    std::optional<size_t> get(size_t key) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = map_key_to_generated_bin_.find(key);
        if (it != map_key_to_generated_bin_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    bool add_generated_bin(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = generated_bin_.find(khash);
        if (it == generated_bin_.end()) {
            generated_bin_.insert(khash);
            return true;
        }
        return false;
    }

    bool is_bin_generated(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = generated_bin_.find(khash);
        return it != generated_bin_.end();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        map_key_to_generated_bin_.clear();
        generated_bin_.clear();
    }

private:
    std::mutex mutex_;
    std::unordered_map<size_t, size_t> map_key_to_generated_bin_;
    std::unordered_set<size_t> generated_bin_;
};

/**
 * Clear the current kernel compilation cache.
 *
 * Return value: void
 */
inline void ClearKernelCache() { detail::HashLookup::inst().clear(); }
}  // namespace tt::tt_metal::detail
