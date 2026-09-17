// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include "impl/context/metal_context.hpp"
#include "jit_build/jit_build_settings.hpp"

// ============================================================================
// Semaphore mechanism solver
// ============================================================================
//
// Decides how each semaphore is accessed. The answer is a SemScope, which codegen bakes
// into every binding's token, so the kernel receives it as a compile-time property.
//
// Two steps:
//   1. CollectSemaphoreBinders() takes the census: which kernel instances bind each semaphore,
//      how many hart instances that adds up to, and which nodes they cover.
//   2. ResolveSemaphoreScopes() reads the census and picks, per semaphore, the fastest access
//      path that keeps its operations atomic.
// Callers need the name -> scope map, plus the binder hart count that the cached-pool seed
// protocol carries in each binding handle.
//
// The mechanism depends on the target device, not just the ProgramSpec: emule has no cached
// pool, and Gen1 only ever gets LOCAL_NONATOMIC. So the same spec can resolve differently on
// different devices.

namespace tt::tt_metal::experimental {

NodeRangeSet to_node_range_set(const Nodes& nodes);

namespace sem_solver {

using SemaphoreNameToScopeMap = std::unordered_map<SemaphoreSpecName, SemScope>;

// Data structure used to keep track of the kernel instances that bind to a given
// semaphore, and their placement (derived from kernel bindings).
struct SemaphoreBinderInfo {
    struct BinderRecord {
        const KernelSpec* kernel = nullptr;
        const SemaphoreBinding* binding = nullptr;
    };
    std::vector<BinderRecord> binders;
    NodeRangeSet binder_node_set;
    uint32_t binder_instance_count = 0;
};

// Every bound semaphore in the program. A declared but unbound semaphore has no entry.
using SemaphoreBinderCensus = std::unordered_map<SemaphoreSpecName, SemaphoreBinderInfo>;

// Look up a semaphore's binder info; an unbound semaphore returns an empty record.
inline const SemaphoreBinderInfo& SemaphoreBinders(const SemaphoreBinderCensus& census, const SemaphoreSpecName& name) {
    static const SemaphoreBinderInfo kEmpty{};
    const auto it = census.find(name);
    return it != census.end() ? it->second : kEmpty;
}

// Returns true if every binder is a data-movement kernel, false otherwise.
inline bool all_binders_are_dm(const SemaphoreBinderInfo& binders) {
    for (const auto& rec : binders.binders) {
        if (!rec.kernel->is_data_movement_kernel()) {
            return false;
        }
    }
    return true;
}

// A semaphore can use the local cached pool only if it lives on one node and every binder is a
// DM kernel on that same node, return true if this is the case, false otherwise.
inline bool cached_geometry_ok(const SemaphoreSpec& sem, const SemaphoreBinderInfo& binders) {
    const NodeRangeSet sem_nodes = to_node_range_set(sem.target_nodes);
    return sem_nodes.num_cores() == 1 &&
           sem_nodes.merge(binders.binder_node_set).num_cores() == sem_nodes.num_cores() && all_binders_are_dm(binders);
}

// Check if the cached tier is available on this target device.
inline bool is_gen2_target() { return tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR; }

inline bool cached_tier_available() {
    return MetalContext::instance().rtoptions().get_target_device() != tt::TargetDevice::Emule;
}

// Picks the fastest access path that keeps this semaphore's operations atomic. Every
// binder is treated as a possible reader and writer.
inline SemScope ResolveSemaphoreScope(const SemaphoreSpec& sem, const SemaphoreBinderInfo& binders) {
    // Gen1 (Wormhole/Blackhole)
    if (!is_gen2_target()) {
        return SemScope::LOCAL_NONATOMIC;
    }

    // Gen2, <=1 binder instance on 1 node
    const NodeRangeSet sem_nodes = to_node_range_set(sem.target_nodes);
    const bool single_node = sem_nodes.num_cores() == 1;
    if (binders.binder_instance_count <= 1 && single_node) {
        return SemScope::LOCAL_NONATOMIC;
    }

    // Gen2, all binders are DMs on the same 1 node as the semaphore
    if (cached_tier_available() && cached_geometry_ok(sem, binders)) {
        return SemScope::DM_LOCAL_CACHED;
    }

    // Gen2, anything else (binders on multiple nodes)
    return SemScope::EXTERNAL;
}

// Census the kernel instances that bind each semaphore. kernel_node_set is each kernel's
// effective placement (derived by CollectSpecData); it is what turns a list of binding kernels
// into an instance count and a node set, so this runs after those node sets exist.
inline SemaphoreBinderCensus CollectSemaphoreBinders(
    const ProgramSpec& spec, const std::unordered_map<KernelSpecName, NodeRangeSet>& kernel_node_set) {
    SemaphoreBinderCensus census;

    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.semaphore_bindings) {
            SemaphoreBinderInfo& sem_info = census[binding.semaphore_spec_name];
            // A kernel may bind a given semaphore only once. A second binding would be the same
            // harts reaching the same L1 word under a second name: the derivation below would
            // count those harts twice (over-sizing the cached pool's seed), and codegen would
            // emit two pool entry blocks for the single row they share.
            for (const auto& rec : sem_info.binders) {
                TT_FATAL(
                    rec.kernel != &kernel,
                    "Kernel '{}' binds semaphore '{}' more than once (accessor names '{}' and '{}'). A "
                    "kernel may bind a given semaphore at most once; to refer to it by another name in "
                    "kernel code, alias the handle (constexpr auto x = sem::y) instead.",
                    kernel.unique_id,
                    binding.semaphore_spec_name,
                    rec.binding->accessor_name,
                    binding.accessor_name);
            }
            sem_info.binders.push_back({&kernel, &binding});
        }
    }

    // Derive each semaphore's binder instance count and binder node set: union of binding-kernels' node sets.
    for (auto& [sem_name, sem_info] : census) {
        for (const auto& rec : sem_info.binders) {
            const NodeRangeSet& binder_nodes = kernel_node_set.at(rec.kernel->unique_id);
            sem_info.binder_instance_count += binder_nodes.num_cores() * rec.kernel->num_threads;
            sem_info.binder_node_set = sem_info.binder_node_set.merge(binder_nodes);
        }
    }

    return census;
}

// Resolve every semaphore the program declares. Unbound ones resolve too.
inline SemaphoreNameToScopeMap ResolveSemaphoreScopes(const ProgramSpec& spec, const SemaphoreBinderCensus& census) {
    SemaphoreNameToScopeMap scopes;
    scopes.reserve(spec.semaphores.size());
    for (const auto& sem : spec.semaphores) {
        scopes[sem.unique_id] = ResolveSemaphoreScope(sem, SemaphoreBinders(census, sem.unique_id));
    }
    return scopes;
}

// Hart instances that bind this semaphore. The cached pool is seeded by having every binder hart
// check in, so the count has to reach the kernel in its binding handle.
inline uint32_t BinderHartCount(const SemaphoreBinderCensus& census, const SemaphoreSpecName& name) {
    return SemaphoreBinders(census, name).binder_instance_count;
}

}  // namespace sem_solver

}  // namespace tt::tt_metal::experimental
