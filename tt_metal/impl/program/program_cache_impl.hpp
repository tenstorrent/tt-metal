// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <tt-metalium/program_cache.hpp>

namespace tt::tt_metal::program_cache::detail {

struct ProgramCacheKeyHasher {
    std::size_t operator()(const ProgramCacheKey& key) const { return static_cast<std::size_t>(key.hash); }
};

struct ProgramCacheImpl {
    bool contains(const ProgramCacheKey& program_key) const { return cache_.contains(program_key); }

    CachedProgramFactory& get(const ProgramCacheKey& program_key) { return cache_.at(program_key); }

    void insert(const ProgramCacheKey& program_key, CachedProgramFactory&& program) {
        cache_.insert({program_key, std::move(program)});
    }

    void enable() { is_enabled_ = true; }
    void disable() { is_enabled_ = false; }
    bool is_enabled() const { return is_enabled_; }

    void set_cache_misses_allowed(bool allowed) { allow_cache_misses_ = allowed; }
    bool cache_misses_allowed() const { return allow_cache_misses_; }

    void clear() { cache_.clear(); }
    std::size_t num_entries() const { return cache_.size(); }

    bool is_enabled_ = true;
    bool allow_cache_misses_ = true;
    std::unordered_map<ProgramCacheKey, CachedProgramFactory, ProgramCacheKeyHasher> cache_;
};

}  // namespace tt::tt_metal::program_cache::detail
