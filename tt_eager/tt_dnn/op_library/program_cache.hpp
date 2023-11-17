// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_eager/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
namespace tt::tt_metal {

namespace program_cache {

namespace detail {

struct ProgramCache {
    inline std::tuple<operation::ProgramWithCallbacks&, bool> get_or_create(
        const operation::DeviceOperation& op,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors,
        Device* device) {
        auto program_hash = op.compute_program_hash(input_tensors, optional_input_tensors);
        auto cache_hit = this->cache_.count(program_hash) > 0;
        if (cache_hit) {
            tt::log_debug(tt::LogOp, "Program Cache: HIT - Getting program from the cache with hash \"{}\"", program_hash);
            auto& program = this->cache_.at(program_hash);
            return {program, cache_hit};
        } else {
            tt::log_debug(tt::LogOp, "Program Cache: MISS - Compiling new program with hash \"{}\"", program_hash);
            this->cache_[program_hash] = op.create_program(input_tensors, optional_input_tensors, output_tensors);
            auto& program = this->cache_[program_hash].program;
            return {this->cache_[program_hash], cache_hit};
        }
    }

    void enable() {
        this->is_enabled_ = true;
    }

    void disable() {
        this->is_enabled_ = false;
    }

    bool is_enabled() const {
        return this->is_enabled_;
    }

    void clear() {
        this->cache_.clear();
    }

    inline std::size_t num_entries() const { return this->cache_.size(); }

   private:
    bool is_enabled_ = false;
    std::unordered_map<operation::Hash, operation::ProgramWithCallbacks> cache_{};
};

inline ProgramCache PROGRAM_CACHE{};

}

template <typename... Args>
inline std::tuple<operation::ProgramWithCallbacks&, bool> get_or_create(Args&&... args) {
    ZoneScoped;
    return detail::PROGRAM_CACHE.get_or_create(std::forward<Args>(args)...);
}

inline bool is_enabled() { return detail::PROGRAM_CACHE.is_enabled(); }

inline void enable() {
    tt::log_info(tt::LogOp, "Program Cache: enabled.");
    detail::PROGRAM_CACHE.enable();
}

inline void disable_and_clear() {
    tt::log_info(tt::LogOp, "Program Cache: disabled and cleared.");
    detail::PROGRAM_CACHE.disable();
    detail::PROGRAM_CACHE.clear();
}

inline std::size_t num_entries() { return detail::PROGRAM_CACHE.num_entries(); }
}

}
