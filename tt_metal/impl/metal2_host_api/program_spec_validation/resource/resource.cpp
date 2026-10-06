// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateResourceBindings(const KernelSpec& kernel, const ValidationContext& ctx) {
    ValidateDFBBindings(kernel, ctx);
    ValidateSemaphoreBindings(kernel, ctx);
    ValidateScratchpadBindings(kernel);
    ValidateTensorBindings(kernel);
    ValidatePrefetcherPipeBindings(kernel);
}

void ValidateResourceSpecs(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    for (const auto& dfb : spec.dataflow_buffers) {
        ValidateDFBSpec(dfb, ctx);
    }
    for (const auto& scratchpad : spec.scratchpads) {
        ValidateScratchpadSpec(scratchpad, ctx);
    }
    for (const auto& sem : spec.semaphores) {
        ValidateSemaphoreSpec(sem, ctx);
    }
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        ValidatePrefetcherPipeParameter(pipe, ctx);
    }
}

void ValidateResourceUsage(const ValidationContext& ctx) {
    // Pipe credit lanes rely on the relay DFBs' PRODUCER kernels being uniform (ValidateDFBEndpoints).
    ValidateComputeSemaphores(ctx);
    ValidateDFBEndpoints(ctx);
    const PrefetcherPipeRoles pipe_roles = ValidatePrefetcherPipeRoles(ctx);
    ValidatePrefetcherPipeLanesAndRelays(ctx, pipe_roles);
    ValidateDFBAliasing(ctx);

    ValidateScratchpadsBound(ctx);
    ValidateTensorParametersUsed(ctx);
    ValidatePrefetcherPipesUsed(ctx);
}

}  // namespace tt::tt_metal::experimental
