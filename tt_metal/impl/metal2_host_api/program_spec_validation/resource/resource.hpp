// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {

// ----------------------------------------------------------------------------
// Resource validation: declarations, bindings, and cross-kernel usage of every
// resource a ProgramSpec declares (DFBs, prefetcher pipes, scratchpads,
// semaphores, tensor parameters).
// ----------------------------------------------------------------------------
//
// Callers outside this folder use only the three phase drivers below
// (resource.cpp fixes the order of checks within each phase).

// One kernel's bindings to every resource kind.
void ValidateResourceBindings(const KernelSpec& kernel, const ValidationContext& ctx);

// Each resource declaration on its own.
void ValidateResourceSpecs(const ValidationContext& ctx);

// Rules that need every kernel's bindings: shared use, endpoints, aliasing, unused declarations.
void ValidateResourceUsage(const ValidationContext& ctx);

// ----------------------------------------------------------------------------
// These are the individual validator
// ----------------------------------------------------------------------------

// dfb/dfb.cpp
void ValidateDFBSpec(const DataflowBufferSpec& dfb, const ValidationContext& ctx);
void ValidateDFBBindings(const KernelSpec& kernel, const ValidationContext& ctx);

// dfb/endpoints.cpp
void ValidateDFBEndpoints(const ValidationContext& ctx);

// dfb/aliasing.cpp
void ValidateDFBAliasing(const ValidationContext& ctx);

// prefetcher_pipe/prefetcher_pipe.cpp
void ValidatePrefetcherPipeParameter(const PrefetcherPipeParameter& pipe, const ValidationContext& ctx);
void ValidatePrefetcherPipeBindings(const KernelSpec& kernel);
void ValidatePrefetcherPipesUsed(const ValidationContext& ctx);

// prefetcher_pipe/roles.cpp
struct PrefetcherPipeRoles {
    std::unordered_map<PrefetcherPipeParamName, NodeRangeSet> pipe_receiver_set;
    std::unordered_map<PrefetcherPipeParamName, const KernelSpec*> receiver_kernel_of;
};
PrefetcherPipeRoles ValidatePrefetcherPipeRoles(const ValidationContext& ctx);

// prefetcher_pipe/lanes_and_relays.cpp
void ValidatePrefetcherPipeLanesAndRelays(const ValidationContext& ctx, const PrefetcherPipeRoles& roles);

// scratchpad.cpp
void ValidateScratchpadSpec(const ScratchpadSpec& scratchpad, const ValidationContext& ctx);
void ValidateScratchpadBindings(const KernelSpec& kernel);
void ValidateScratchpadsBound(const ValidationContext& ctx);

// semaphore.cpp
void ValidateSemaphoreSpec(const SemaphoreSpec& sem, const ValidationContext& ctx);
void ValidateSemaphoreBindings(const KernelSpec& kernel, const ValidationContext& ctx);
void ValidateComputeSemaphores(const ValidationContext& ctx);

// tensor.cpp
void ValidateTensorBindings(const KernelSpec& kernel);
void ValidateTensorParametersUsed(const ValidationContext& ctx);

// ----------------------------------------------------------------------------
// Shared by every resource's binding checks
// ----------------------------------------------------------------------------

template <typename KernelId>
inline void ValidateAccessorNameLength(const KernelId& kernel_id, std::string_view kind, std::string_view name) {
    TT_FATAL(
        name.size() <= MAX_ACCESSOR_NAME_LENGTH,
        "Kernel '{}' {} accessor_name '{}' is {} characters; an accessor_name must be at most {} characters",
        kernel_id,
        kind,
        name,
        name.size(),
        MAX_ACCESSOR_NAME_LENGTH);
}

}  // namespace tt::tt_metal::experimental
