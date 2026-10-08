// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

namespace {

using SemaphoreBinderInfo = CollectedSpecData::SemaphoreBinderInfo;
using SemaphoreBinderCensus = CollectedSpecData::SemaphoreBinderCensus;

// Look up a semaphore's binder info; an unbound semaphore returns an empty record.
const SemaphoreBinderInfo& SemaphoreBinders(const SemaphoreBinderCensus& census, const SemaphoreSpecName& name) {
    static const SemaphoreBinderInfo kEmpty{};
    const auto it = census.find(name);
    return it != census.end() ? it->second : kEmpty;
}

// Returns true if every binder is a data-movement kernel, false otherwise.
bool all_binders_are_dm(const SemaphoreBinderInfo& binders) {
    for (const auto& rec : binders.binders) {
        if (!rec.kernel->is_data_movement_kernel()) {
            return false;
        }
    }
    return true;
}

// Returns true if there is at least one binder and every one of them is a compute kernel.
bool all_binders_are_compute(const SemaphoreBinderInfo& binders) {
    if (binders.binders.empty()) {
        return false;
    }
    for (const auto& rec : binders.binders) {
        if (!rec.kernel->is_compute_kernel()) {
            return false;
        }
    }
    return true;
}

// A semaphore can use the local cached pool only if it lives on one node and every binder is a
// DM kernel on that same node, return true if this is the case, false otherwise.
bool cached_geometry_ok(const SemaphoreSpec& sem, const SemaphoreBinderInfo& binders) {
    const NodeRangeSet sem_nodes = to_node_range_set(sem.target_nodes);
    return sem_nodes.num_cores() == 1 &&
           sem_nodes.merge(binders.binder_node_set).num_cores() == sem_nodes.num_cores() && all_binders_are_dm(binders);
}

// Check if the cached tier is available on this target device. Arch and the target device both
// come from the program's own env (BuildProgramFromSpec builds against the mesh device's env),
// not the default context.
bool is_gen2_target(const Hal& hal) { return hal.get_arch() == tt::ARCH::QUASAR; }

bool cached_tier_available(MetalEnvImpl& env) {
    return env.get_rtoptions().get_target_device() != tt::TargetDevice::Emule;
}

// Picks the fastest access path that keeps this semaphore's operations atomic. Every
// binder is treated as a possible reader and writer.
SemScope ResolveSemaphoreScope(const SemaphoreSpec& sem, const SemaphoreBinderInfo& binders, MetalEnvImpl& env) {
    const Hal& hal = env.get_hal();
    // Gen1 (Wormhole/Blackhole)
    if (!is_gen2_target(hal)) {
        // COMPUTE_ATOMIC is a Blackhole UNPACK <-> PACK mechanism (the Tensix hardware semaphore)
        // and nothing else, so it applies only when EVERY binder is a compute kernel. A mixed
        // compute/DM binding is rejected by ValidateComputeSemaphores (validation/resource/semaphore.cpp),
        // since a DM core cannot reach that semaphore. One compute binding compiles into three TRISC
        // binaries with two writers (UNPACK and PACK), so a compute-bound word must never take the
        // non-atomic path. Wormhole has no compute implementation; its compute bindings are rejected
        // on the host.
        if (all_binders_are_compute(binders) && hal.get_arch() == tt::ARCH::BLACKHOLE) {
            return SemScope::COMPUTE_ATOMIC;
        }
        return SemScope::LOCAL_NONATOMIC;
    }

    // Gen2, <=1 binder instance on 1 node
    const NodeRangeSet sem_nodes = to_node_range_set(sem.target_nodes);
    const bool single_node = sem_nodes.num_cores() == 1;
    if (binders.binder_instance_count <= 1 && single_node) {
        return SemScope::LOCAL_NONATOMIC;
    }

    // Gen2, all binders are DMs on the same 1 node as the semaphore
    if (cached_tier_available(env) && cached_geometry_ok(sem, binders)) {
        return SemScope::DM_LOCAL_CACHED;
    }

    // Gen2, anything else (binders on multiple nodes)
    return SemScope::EXTERNAL;
}

// Hart instances that bind this semaphore. The cached pool is seeded by having every binder hart
// check in, so the count has to reach the kernel in its binding handle.
uint32_t BinderHartCount(const SemaphoreBinderCensus& census, const SemaphoreSpecName& name) {
    return SemaphoreBinders(census, name).binder_instance_count;
}

}  // namespace

SemaphoreNameToScopeMap ResolveSemaphoreScopes(
    const ProgramSpec& spec, const SemaphoreBinderCensus& census, MetalEnvImpl& env) {
    SemaphoreNameToScopeMap scopes;
    scopes.reserve(spec.semaphores.size());
    for (const auto& sem : spec.semaphores) {
        scopes[sem.unique_id] = ResolveSemaphoreScope(sem, SemaphoreBinders(census, sem.unique_id), env);
    }
    return scopes;
}

// Create map of accessor name -> semaphore handle: the logical id, the resolved scope,
// and the binder hart count for local cached semaphores.
tt::tt_metal::SemaphoreBindingHandleMap MakeSemaphoreBindingHandles(
    const KernelSpec& kernel_spec,
    const SemaphoreBinderCensus& semaphore_binders,
    const SemaphoreNameToIdMap& semaphore_name_to_id,
    const SemaphoreNameToScopeMap& semaphore_name_to_scope) {
    tt::tt_metal::SemaphoreBindingHandleMap out;
    out.reserve(kernel_spec.semaphore_bindings.size());
    for (const auto& semaphore_binding : kernel_spec.semaphore_bindings) {
        const uint32_t id = semaphore_name_to_id.at(semaphore_binding.semaphore_spec_name);
        TT_FATAL(
            id <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' semaphore '{}' id {} does not fit uint16_t",
            kernel_spec.unique_id,
            semaphore_binding.semaphore_spec_name,
            id);
        const SemScope scope = semaphore_name_to_scope.at(semaphore_binding.semaphore_spec_name);
        const uint32_t total_binder_harts =
            scope == SemScope::DM_LOCAL_CACHED
                ? BinderHartCount(semaphore_binders, semaphore_binding.semaphore_spec_name)
                : 0u;
        TT_FATAL(
            total_binder_harts <= 0x7FFFu,
            "Semaphore '{}' has {} binder harts; the cached seed protocol supports at most 32767",
            semaphore_binding.semaphore_spec_name,
            total_binder_harts);
        out.emplace(
            semaphore_binding.accessor_name,
            tt::tt_metal::SemaphoreBindingHandle{static_cast<uint16_t>(id), scope, total_binder_harts});
    }
    return out;
}

SemaphoreHandles RegisterSemaphores(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalEnvImpl& metal_env,
    detail::ProgramImpl& program_impl) {
    // Create Semaphores and build the name -> ID map.
    // NOTE: Iterate over spec.semaphores to preserve user-provided deterministic ordering.
    SemaphoreNameToIdMap semaphore_name_to_id;
    for (const auto& semaphore_spec : spec.semaphores) {
        const SemaphoreSpecName& semaphore_name = semaphore_spec.unique_id;
        const uint32_t init_value = semaphore_spec.advanced_options.initial_value;
        uint32_t sem_id =
            program_impl.create_semaphore(to_node_range_set(semaphore_spec.target_nodes), init_value, CoreType::WORKER);
        program_impl.register_semaphore_spec_name(semaphore_name.get(), sem_id);
        semaphore_name_to_id[semaphore_name] = sem_id;
    }

    // Pick each semaphore's access mechanism. Resolve against this program's env (the mesh
    // device's), not the default context, so a non-default-context device resolves its own arch
    // and target device.
    SemaphoreNameToScopeMap semaphore_name_to_scope =
        ResolveSemaphoreScopes(spec, collected.semaphore_binders, metal_env);
    return SemaphoreHandles{.id = std::move(semaphore_name_to_id), .scope = std::move(semaphore_name_to_scope)};
}

}  // namespace tt::tt_metal::experimental
