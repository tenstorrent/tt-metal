// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateResourceBindings(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch) {
    ValidateDFBBindings(kernel, ctx);
    ValidateSemaphoreBindings(kernel, arch);
    ValidateScratchpadBindings(kernel);
    ValidateTensorBindings(kernel);
    ValidatePrefetcherPipeBindings(kernel);
}

void ValidateResourceSpecs(
    const ValidationContext& ctx,
    tt::ARCH arch,
    uint32_t l1_alignment,
    const NumBanksFromBufferType& num_banks_from_buffer_type) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;
    for (const auto& dfb : spec.dataflow_buffers) {
        ValidateDFBSpec(dfb, collected, l1_alignment, arch, num_banks_from_buffer_type);
    }
    for (const auto& scratchpad : spec.scratchpads) {
        ValidateScratchpadSpec(scratchpad, arch);
    }
    for (const auto& sem : spec.semaphores) {
        ValidateSemaphoreSpec(sem, arch);
    }
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        ValidatePrefetcherPipeParameter(pipe, l1_alignment);
    }
}

void ValidateResourceUsage(const ValidationContext& ctx, tt::ARCH arch) {
    // Pipe credit lanes rely on the relay DFBs' PRODUCER kernels being uniform (ValidateDFBEndpoints).
    ValidateComputeSemaphores(ctx);
    ValidateDFBEndpoints(ctx, arch);
    const PrefetcherPipeRoles pipe_roles = ValidatePrefetcherPipeRoles(ctx);
    ValidatePrefetcherPipeLanesAndRelays(ctx, pipe_roles, arch);
    ValidateDFBAliasing(ctx);

    ValidateScratchpadsBound(ctx);
    ValidateTensorParametersUsed(ctx);
    ValidatePrefetcherPipesUsed(ctx);
}

}  // namespace tt::tt_metal::experimental
