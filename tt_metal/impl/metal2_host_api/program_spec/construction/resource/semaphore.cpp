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

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

// Create map of accessor name -> semaphore handle: the logical id, the resolved scope,
// and the binder hart count for local cached semaphores.
tt::tt_metal::SemaphoreBindingHandleMap MakeSemaphoreBindingHandles(
    const KernelSpec& kernel_spec,
    const sem_solver::SemaphoreBinderCensus& semaphore_binders,
    const SemaphoreNameToIdMap& semaphore_name_to_id,
    const sem_solver::SemaphoreNameToScopeMap& semaphore_name_to_scope) {
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
                ? sem_solver::BinderHartCount(semaphore_binders, semaphore_binding.semaphore_spec_name)
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
    sem_solver::SemaphoreNameToScopeMap semaphore_name_to_scope =
        sem_solver::ResolveSemaphoreScopes(spec, collected.semaphore_binders, metal_env);
    return SemaphoreHandles{.id = std::move(semaphore_name_to_id), .scope = std::move(semaphore_name_to_scope)};
}

}  // namespace tt::tt_metal::experimental
