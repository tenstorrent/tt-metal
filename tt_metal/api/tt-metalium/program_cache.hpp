// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include <tt-metalium/program.hpp>
#include <tt_stl/unique_any.hpp>
#include <tt_stl/overloaded.hpp>

#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::program_cache::detail {

template <typename shared_variables_t>
struct CachedProgram {
private:
    std::optional<tt::tt_metal::Program> owned_program;
    std::optional<shared_variables_t> owned_shared_variables;

    // Hidden to avoid misuse of `CachedProgram` as a proxy.
    CachedProgram(tt::tt_metal::Program& program, shared_variables_t& shared_variables) :
        program{program}, shared_variables{shared_variables} {}

public:
    tt::tt_metal::Program& program;

    // Cached program needs to share shared_variables between create and override_runtime_arguments functions
    shared_variables_t& shared_variables;

    CachedProgram(tt::tt_metal::Program&& program, shared_variables_t&& shared_variables) :
        owned_program{std::move(program)},
        owned_shared_variables{std::move(shared_variables)},
        program{*owned_program},
        shared_variables{*owned_shared_variables} {}

    // Move constructor is needed to make `CachedProgram` work with `unique_any`.
    // `CachedProgram` with owned `program` and `shared_variables` are moved; unowned proxies have their references
    // effectively copied.
    CachedProgram(CachedProgram&& other) noexcept :
        owned_program(std::move(other.owned_program)),
        owned_shared_variables(std::move(other.owned_shared_variables)),
        program(owned_program ? *owned_program : other.program),
        shared_variables(owned_shared_variables ? *owned_shared_variables : other.shared_variables) {}

    // Creates a "proxy" `CachedProgram` that references `program` and `shared_variables`.
    // Used for adapting `CachedMeshWorkload` to `CachedProgram`, and interfacing with TTNN Ops in
    // `override_runtime_arguments` methods.
    static CachedProgram proxy(tt::tt_metal::Program& program, shared_variables_t& shared_variables) {
        return CachedProgram{program, shared_variables};
    }

    CachedProgram(const CachedProgram&) = delete;
    CachedProgram& operator=(const CachedProgram&) = delete;
    CachedProgram& operator=(CachedProgram&&) = delete;
};

template <typename shared_variables_t>
struct CachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    shared_variables_t shared_variables;

    CachedMeshWorkload(tt::tt_metal::distributed::MeshWorkload&& workload, shared_variables_t&& shared_variables) :
        workload{std::move(workload)}, shared_variables{std::move(shared_variables)} {}
};

// Adapted cached mesh workload is used to interpop TT-distributed infra that dispatches MeshWorkloads and program
// factories written for a single device: programs and shared variables are created by single-device program factories,
// then stamped out to the entire mesh.
template <typename shared_variables_t>
struct AdaptedCachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<distributed::MeshCoordinateRange, shared_variables_t> shared_variables;

    AdaptedCachedMeshWorkload(
        tt::tt_metal::distributed::MeshWorkload&& workload,
        std::unordered_map<distributed::MeshCoordinateRange, shared_variables_t>&& shared_variables) :
        workload{std::move(workload)}, shared_variables{std::move(shared_variables)} {}
};

struct CachedProgramFactory {
    static constexpr auto MAX_SIZE = 4096;
    static constexpr auto ALIGNMENT = 32;

    tt::stl::unique_any<MAX_SIZE, ALIGNMENT> cached_program;

    // Used to map a runtime value to a program factory type that is being used
    std::size_t program_factory_index = 0;

    template <typename shared_variables_t>
    CachedProgramFactory(CachedProgram<shared_variables_t>&& cached_program, std::size_t program_factory_index) :
        cached_program{std::move(cached_program)}, program_factory_index{program_factory_index} {}

    template <typename shared_variables_t>
    CachedProgramFactory(CachedMeshWorkload<shared_variables_t>&& cached_workload, std::size_t program_factory_index) :
        cached_program{std::move(cached_workload)}, program_factory_index{program_factory_index} {}

    template <typename shared_variables_t>
    CachedProgramFactory(
        AdaptedCachedMeshWorkload<shared_variables_t>&& cached_workload, std::size_t program_factory_index) :
        cached_program{std::move(cached_workload)}, program_factory_index{program_factory_index} {}
};

// Generic Program Cache: This data structure is tied to a device handle and can store generic program types from
// TT-Metal and TT-Eager using tt::stl::concepts::unique_any.
struct ProgramCache {
    bool contains(uint64_t program_hash) const { return this->cache_.count(program_hash) > 0; }

    CachedProgramFactory& get(uint64_t program_hash) { return this->cache_.at(program_hash); }

    void insert(uint64_t program_hash, CachedProgramFactory&& program) {
        this->cache_.insert({program_hash, std::move(program)});
    }

    void enable() { is_enabled_ = true; }

    void disable() { is_enabled_ = false; }

    bool is_enabled() const { return is_enabled_; }

    void set_cache_misses_allowed(bool allowed) { allow_cache_misses_ = allowed; }
    bool cache_misses_allowed() const { return allow_cache_misses_; }

    void clear() { this->cache_.clear(); }

    std::size_t num_entries() const { return this->cache_.size(); }

private:
    bool is_enabled_ = true;
    bool allow_cache_misses_ = true;
    std::unordered_map<uint64_t, CachedProgramFactory> cache_{};
};

}  // namespace tt::tt_metal::program_cache::detail
