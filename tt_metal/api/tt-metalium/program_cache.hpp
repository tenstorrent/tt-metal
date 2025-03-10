// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "program_impl.hpp"
#include "unique_any.hpp"
#include "mesh_workload.hpp"

namespace tt::tt_metal::program_cache::detail {
template <typename shared_variables_t>
struct CachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    // Shared variables between create and override_runtime_arguments functions
    shared_variables_t shared_variables;

    CachedMeshWorkload(tt::tt_metal::distributed::MeshWorkload&& workload, shared_variables_t&& shared_variables) :
        workload{std::move(workload)}, shared_variables{std::forward<shared_variables_t>(shared_variables)} {}
};

template <typename shared_variables_t>
struct CachedProgram {
    tt::tt_metal::Program program;
    // Cached program needs to share shared_variables between create and override_runtime_arguments functions
    shared_variables_t shared_variables;

    CachedProgram(tt::tt_metal::Program&& program, shared_variables_t&& shared_variables) :
        program{std::move(program)}, shared_variables{std::forward<shared_variables_t>(shared_variables)} {}
};

// Adapter that provides a unified interface for both CachedProgram and CachedMeshWorkload
template <typename shared_variables_t>
class ProgramAdapter {
private:
    using CachedObject = std::variant<CachedMeshWorkload<shared_variables_t>, CachedProgram<shared_variables_t>>;
    CachedObject cached_object_;
    // Helper to retrieve the first program from a mesh workload
    static tt::tt_metal::Program& get_first_program(CachedMeshWorkload<shared_variables_t>& cached_mesh_workload) {
        // Get the programs map from the workload
        auto& programs = cached_mesh_workload.workload.get_programs();

        // There must be at least one program in the workload
        TT_FATAL(!programs.empty(), "Mesh workload must have at least one program");

        // Return the first program in the workload
        auto& first_program_pair = *programs.begin();
        return first_program_pair.second;
    }

public:
    // These are references to the original objects
    tt::tt_metal::Program& program;
    shared_variables_t& shared_variables;

    // Constructor for CachedProgram
    ProgramAdapter(CachedProgram<shared_variables_t>&& cached_program) :
        cached_object_(std::move(cached_program)),
        program(std::get<CachedProgram<shared_variables_t>>(cached_object_).program),
        shared_variables(std::get<CachedProgram<shared_variables_t>>(cached_object_).shared_variables) {}

    // Constructor for Program and shared variables
    ProgramAdapter(tt::tt_metal::Program&& program, shared_variables_t&& shared_vars) :
        ProgramAdapter(CachedProgram<shared_variables_t>{std::move(program), std::move(shared_vars)}) {}

    // Constructor for CachedMeshWorkload
    ProgramAdapter(CachedMeshWorkload<shared_variables_t>&& cached_mesh_workload) :
        cached_object_(std::move(cached_mesh_workload)),
        program(get_first_program(std::get<CachedMeshWorkload<shared_variables_t>>(cached_object_))),
        shared_variables(std::get<CachedMeshWorkload<shared_variables_t>>(cached_object_).shared_variables) {}

    ProgramAdapter(ProgramAdapter&& other) noexcept :
        cached_object_{std::move(other.cached_object_)},
        program{
            (cached_object_.index() == 0)
                ? get_first_program(std::get<CachedMeshWorkload<shared_variables_t>>(cached_object_))
                : std::get<CachedProgram<shared_variables_t>>(cached_object_).program},
        shared_variables{
            (cached_object_.index() == 0)
                ? std::get<CachedMeshWorkload<shared_variables_t>>(cached_object_).shared_variables
                : std::get<CachedProgram<shared_variables_t>>(cached_object_).shared_variables} {}

    // Get the CachedMeshWorkload (throws if not a mesh workload)
    CachedMeshWorkload<shared_variables_t>& get_cached_mesh_workload() {
        return std::get<CachedMeshWorkload<shared_variables_t>>(cached_object_);
    }

    // Get the CachedProgram (throws if not a program)
    CachedProgram<shared_variables_t>& get_cached_program() {
        return std::get<CachedProgram<shared_variables_t>>(cached_object_);
    }
};

struct CachedProgramFactory {
    static constexpr auto MAX_SIZE = 4096;
    static constexpr auto ALIGNMENT = 32;

    tt::stl::unique_any<MAX_SIZE, ALIGNMENT> cached_program;
    // program_factory_index is used to map a runtime value to a program factory type that is being used
    std::size_t program_factory_index;

    template <typename shared_variables_t>
    CachedProgramFactory(ProgramAdapter<shared_variables_t>&& cached_program, std::size_t program_factory_index) :
        cached_program{std::move(cached_program)}, program_factory_index{program_factory_index} {}
};

// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from
// TT-Metal and TT-Eager using tt::stl::concepts::unique_any.
struct ProgramCache {
    bool contains(uint64_t program_hash) { return this->cache_.count(program_hash) > 0; }

    CachedProgramFactory& get(uint64_t program_hash) { return this->cache_.at(program_hash); }

    void insert(uint64_t program_hash, CachedProgramFactory&& program) {
        this->cache_.insert({program_hash, std::move(program)});
    }

    void enable() { is_enabled_ = true; }

    void disable() { is_enabled_ = false; }

    bool is_enabled() { return is_enabled_; }

    void clear() { this->cache_.clear(); }

    std::size_t num_entries() const { return this->cache_.size(); }

private:
    bool is_enabled_ = false;
    std::unordered_map<uint64_t, CachedProgramFactory> cache_{};
};

}  // namespace tt::tt_metal::program_cache::detail
