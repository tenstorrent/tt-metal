// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"

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
void ValidateResourceBindings(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch);

// Each resource declaration on its own.
void ValidateResourceSpecs(
    const ValidationContext& ctx,
    tt::ARCH arch,
    uint32_t l1_alignment,
    const NumBanksFromBufferType& num_banks_from_buffer_type);

// Rules that need every kernel's bindings: shared use, endpoints, aliasing, unused declarations.
void ValidateResourceUsage(const ValidationContext& ctx, tt::ARCH arch);

// ----------------------------------------------------------------------------
// These are the individual validator
// ----------------------------------------------------------------------------

// dfb.cpp
void ValidateDFBSpec(
    const DataflowBufferSpec& dfb,
    const CollectedSpecData& collected,
    uint32_t l1_alignment,
    tt::ARCH arch,
    const NumBanksFromBufferType& num_banks_from_buffer_type);
void ValidateDFBBindings(const KernelSpec& kernel, const ValidationContext& ctx);
void ValidateDFBEndpoints(const ValidationContext& ctx, tt::ARCH arch);
void ValidateDFBAliasing(const ValidationContext& ctx);

// prefetcher_pipe.cpp
void ValidatePrefetcherPipeParameter(const PrefetcherPipeParameter& pipe, uint32_t l1_alignment);
void ValidatePrefetcherPipeBindings(const KernelSpec& kernel);
void ValidatePrefetcherPipesUsed(const ValidationContext& ctx);
struct PrefetcherPipeRoles {
    std::unordered_map<PrefetcherPipeParamName, NodeRangeSet> pipe_receiver_set;
    std::unordered_map<PrefetcherPipeParamName, const KernelSpec*> receiver_kernel_of;
};
PrefetcherPipeRoles ValidatePrefetcherPipeRoles(const ValidationContext& ctx);
void ValidatePrefetcherPipeLanesAndRelays(
    const ValidationContext& ctx, const PrefetcherPipeRoles& roles, tt::ARCH arch);

// scratchpad.cpp
void ValidateScratchpadSpec(const ScratchpadSpec& scratchpad, tt::ARCH arch);
void ValidateScratchpadBindings(const KernelSpec& kernel);
void ValidateScratchpadsBound(const ValidationContext& ctx);

// semaphore.cpp
void ValidateSemaphoreSpec(const SemaphoreSpec& sem, tt::ARCH arch);
void ValidateSemaphoreBindings(const KernelSpec& kernel, tt::ARCH arch);
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
