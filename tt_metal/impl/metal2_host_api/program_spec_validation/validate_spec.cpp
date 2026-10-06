// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateProgramSpec(
    const ProgramSpec& spec, const CollectedSpecData& collected, MetalContext& metal_ctx, const Allocator& allocator) {
    const Hal& hal = metal_ctx.hal();
    // Sanity check for supported architecture.
    TT_FATAL(is_gen1_arch(hal) || is_gen2_arch(hal), "Unsupported architecture.");

    const ValidationContext ctx{
        .spec = spec, .collected = collected, .metal_ctx = metal_ctx, .hal = hal, .allocator = allocator};

    // Order matters: later checks rely on earlier ones. Every local "> 0" / name-resolution check
    // passes before the structural checks that divide by or look up those values.
    ValidateProgramMisc(ctx);
    ValidateWorkUnitFields(ctx);

    for (const auto& kernel : spec.kernels) {
        ValidateKernelSpec(kernel, ctx);
    }
    ValidateResourceSpecs(ctx);
    ValidateResourceUsage(ctx);

    for (const auto& work_unit : spec.work_units) {
        ValidateWorkUnitSpec(work_unit, ctx);
        ValidateDFBSlotsPerNode(work_unit, ctx);
    }
    // Checked over every node at once rather than per WorkUnitSpec: WorkUnitSpecs with the same name
    // are exempt from the overlap check, so a node's kernels can come from more than one WorkUnitSpec.
    ValidateGen1DMPlacement(ctx);
    ValidateScratchpadBindersPerNode(ctx);

    // NOTE:
    // Placement consistency between kernels, DFBs, and WorkUnitSpecs is now structural,
    // not validated:
    //  - Kernels' effective node sets ARE the union of their containing WorkUnitSpecs' target_nodes
    //  - DFBs' allocation node sets are the union of their binding kernels' node sets
}

}  // namespace tt::tt_metal::experimental
