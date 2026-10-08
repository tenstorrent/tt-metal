// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec_validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

void ValidateNodeCapacity(const ValidationContext& ctx, tt::ARCH arch, uint32_t max_slots_per_core) {
    for (const auto& work_unit : ctx.spec.work_units) {
        ValidateWorkUnitSpec(work_unit, ctx, arch);
        ValidateDFBSlotsPerNode(work_unit, ctx, max_slots_per_core, arch);
    }
    // Checked over every node at once rather than per WorkUnitSpec: WorkUnitSpecs with the same name
    // are exempt from the overlap check, so a node's kernels can come from more than one WorkUnitSpec.
    ValidateGen1DMPlacement(ctx, arch);
    ValidateScratchpadBindersPerNode(ctx);
}

}  // namespace tt::tt_metal::experimental
