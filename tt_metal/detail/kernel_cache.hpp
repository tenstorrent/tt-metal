// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <stdint.h>
#include <unordered_set>
#include <unordered_map>
#include <optional>

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

    bool is_bin_generated(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        return generated_bins_.find(khash) != generated_bins_.end();
    }

    void add_generated_bin(size_t khash) {
        std::unique_lock<std::mutex> lock(mutex_);
        generated_bins_.insert(khash);
    }

    bool add_key_to_generated_bin(size_t khash, size_t key) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = map_key_to_generated_bin_.find(key);
        if (it == map_key_to_generated_bin_.end()) {
            map_key_to_generated_bin_[key] = khash;
            return true;
        }
        return false;
    }

    std::optional<size_t> get_generated_bin(size_t key) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = map_key_to_generated_bin_.find(key);
        if (it != map_key_to_generated_bin_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        hashes_.clear();
        generated_bins_.clear();
        map_key_to_generated_bin_.clear();
    }

private:
    std::mutex mutex_;
    std::unordered_set<size_t> hashes_;
    std::unordered_set<size_t> generated_bins_;
    std::unordered_map<size_t, size_t> map_key_to_generated_bin_;
};

/**
 * Clear the current kernel compilation cache.
 *
 * Return value: void
 */
inline void ClearKernelCache() { detail::HashLookup::inst().clear(); }
}  // namespace tt::tt_metal::detail
