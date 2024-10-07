// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program_handle.hpp"
#include "tt_metal/tt_stl/unique_any.hpp"

namespace tt::tt_metal::program_cache::detail {

template <typename shared_variables_t>
struct CachedProgram {
    tt::tt_metal::ProgramHandle program;
    // Cached program needs to share shared_variables between create and override_runtime_arguments functions
    shared_variables_t shared_variables;

    CachedProgram(tt::tt_metal::ProgramHandle program, shared_variables_t&& shared_variables) :
        program{program}, shared_variables{std::forward<shared_variables_t>(shared_variables)} {}
};

struct CachedProgramFactory {
    static constexpr auto MAX_SIZE = 4096;
    static constexpr auto ALIGNMENT = 32;

    tt::stl::unique_any<MAX_SIZE, ALIGNMENT> cached_program;
    bool should_release = true;

    // program_factory_index is used to map a runtime value to a program factory type that is being used
    std::size_t program_factory_index;

    template <typename shared_variables_t>
    CachedProgramFactory(CachedProgram<shared_variables_t>&& program, std::size_t program_factory_index) :
        cached_program{std::move(program)}, program_factory_index{program_factory_index} {
            static_assert(offsetof(CachedProgram<shared_variables_t>, program) == 0, "program must be the first member");
    }

    CachedProgramFactory(const CachedProgramFactory&) = delete;
    CachedProgramFactory& operator=(const CachedProgramFactory&) = delete;

    CachedProgramFactory(CachedProgramFactory&&) = default;
    CachedProgramFactory& operator=(CachedProgramFactory&&) = default;

    ~CachedProgramFactory() {
        if (this->should_release) {
            auto& handle = this->cached_program.template get<ProgramHandle>();
            CloseProgram(handle);
        }
    }
};

// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from
// TT-Metal and TT-Eager using tt::stl::concepts::unique_any.
struct ProgramCache {
    bool contains(uint64_t program_hash) { return this->cache_.count(program_hash) > 0; }

    CachedProgramFactory& get(uint64_t program_hash) { return this->cache_.at(program_hash); }

    void insert(uint64_t program_hash, CachedProgramFactory&& program) {
        this->cache_.try_emplace(program_hash, std::move(program));
        program.should_release = false;
    }

    void enable() { is_enabled_ = true; }

    void disable() { is_enabled_ = false; }

    bool is_enabled() { return is_enabled_; }

    void clear() { this->cache_.clear(); }

    std::size_t num_entries() const { return this->cache_.size(); }

   private:
    inline static bool is_enabled_ = false;
    std::unordered_map<uint64_t, CachedProgramFactory> cache_{};
};

}  // namespace tt::tt_metal::program_cache::detail
