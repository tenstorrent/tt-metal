// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

#include "jit_build/data_format.hpp"

namespace tt::tt_metal::experimental {

void ValidateResourceTileMetadata(
    std::string_view resource_kind, std::string_view resource_name, DataFormat format, const Tile& tile) {
    constexpr uint32_t kMxBlockSize = 32;
    TT_FATAL(
        !tt::is_mx_format(format) || tile.get_tile_hw() % kMxBlockSize == 0,
        "{} '{}' has format '{}' and tile shape ({}x{}) with {} elements; "
        "MX tiles must contain a multiple of 32 elements",
        resource_kind,
        resource_name,
        format,
        tile.get_height(),
        tile.get_width(),
        tile.get_tile_hw());
}

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
    for (const auto& parameter : spec.advanced_options.sram_arena_parameters) {
        ValidateSRAMArenaParameter(parameter);
    }
    // Borrowed DFB validation can calculate tensor allocation sizes.
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        ValidateTensorParameterSpec(tensor_parameter);
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        ValidateDFBSpec(dfb, collected, l1_alignment, arch, num_banks_from_buffer_type);
    }
    for (const auto& scratchpad : spec.scratchpads) {
        ValidateScratchpadSpec(scratchpad, collected, l1_alignment, arch);
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

    ValidateScratchpadsUsed(ctx);
    ValidateTensorParametersUsed(ctx);
    ValidatePerNodeTensorPlacement(ctx);
    ValidatePrefetcherPipesUsed(ctx);
    ValidateSRAMArenaParametersUsed(ctx);
}

}  // namespace tt::tt_metal::experimental
