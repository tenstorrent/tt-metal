// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {

// ----------------------------------------------------------------------------
// Placement validation: where kernels and resources land (WorkUnitSpec target
// nodes) and what each node can hold.
// ----------------------------------------------------------------------------
//
// Callers outside this folder use only the two phase drivers below
// (placement.cpp fixes the order of checks within each phase).

// WorkUnitSpec target nodes: within the compute grid, at least one WorkUnitSpec, no overlap.
void ValidateWorkUnitFields(const ValidationContext& ctx, const CoreCoord& compute_grid_size);

// Per-node limits: work-unit capacity, DFB slots, Gen1 DM processors, scratchpad binders.
void ValidateNodeCapacity(const ValidationContext& ctx, tt::ARCH arch, uint32_t max_slots_per_core);

// ----------------------------------------------------------------------------
// These are the individual validator
// ----------------------------------------------------------------------------

// work_unit.cpp
void ValidateWorkUnitSpec(const WorkUnitSpec& work_unit, const ValidationContext& ctx, tt::ARCH arch);

// dfb.cpp
void ValidateDFBSlotsPerNode(
    const WorkUnitSpec& work_unit, const ValidationContext& ctx, uint32_t max_slots_per_core, tt::ARCH arch);

// kernel.cpp
void ValidateGen1DMPlacement(const ValidationContext& ctx, tt::ARCH arch);

// scratchpad.cpp
void ValidateScratchpadBindersPerNode(const ValidationContext& ctx);

}  // namespace tt::tt_metal::experimental
