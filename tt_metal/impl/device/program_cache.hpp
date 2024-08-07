// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tt_stl/unique_any.hpp"

namespace tt::tt_metal {

namespace program_cache {

namespace detail {
// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from
// TT-Metal and TT-Eager using tt::stl::concepts::unique_any.
struct ProgramCache {
    inline bool contains(uint64_t program_hash) { return is_enabled() ? this->cache_.count(program_hash) > 0 : false; }

    template <typename T>
    inline T& get(uint64_t program_hash) {
        return this->cache_.at(program_hash).get<T>();
    }

    template <typename T>
    inline void insert(uint64_t program_hash, T&& program) {
        if ( !is_enabled()) {
            return;
        }
        using cache_t = decltype(this->cache_);
        this->cache_.try_emplace(program_hash, program);
    }

    void enable() {
        is_enabled_ = true;
    }

    void disable() {
        is_enabled_ = false;
    }

    bool is_enabled() const {
        return is_enabled_;
    }

    void clear() {
        this->cache_.clear();
    }

    inline std::size_t num_entries() const { return this->cache_.size(); }

   private:
    inline static bool is_enabled_ = false;

    static constexpr auto MAX_CACHED_PROGRAM_SIZE = 1024;
    static constexpr auto ALIGNMENT = 32;
    std::unordered_map<uint64_t, tt::stl::unique_any<MAX_CACHED_PROGRAM_SIZE, ALIGNMENT>> cache_{};
};

}
}
}
