// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <memory>
#include "tt_metal/common/logger.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt::tt_metal {

namespace program_cache {

namespace detail {
// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from TT-Metal
// and TT-Eager using std::shared_ptr<void>.
struct ProgramCache {
    inline std::tuple<std::shared_ptr<void>, bool> find(uint64_t program_hash) {
        auto cache_hit = this->cache_.count(program_hash) > 0;
        if (cache_hit) {
            return {this->cache_.at(program_hash), cache_hit};
        }
        return {std::shared_ptr<int>(), cache_hit};
    }
    inline void insert(uint64_t program_hash, std::shared_ptr<void> program_ptr) {
        this->cache_[program_hash] = program_ptr;
    }

    void enable() {
        is_enabled_ = true;
    }

    void disable() {
        is_enabled_ = false;
    }

    bool is_enabled()  {
        return is_enabled_;
    }

    void clear() {
        this->cache_.clear();
    }

    inline std::size_t num_entries() const { return this->cache_.size(); }

   private:
    inline static bool is_enabled_ = false;
    std::unordered_map<uint64_t, std::shared_ptr<void>> cache_{};
};

}
}
}
