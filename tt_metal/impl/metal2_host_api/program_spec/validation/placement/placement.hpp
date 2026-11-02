// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"

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

// Validate work-unit capacity, DFB slots, Gen1 DM processors, and scratchpad placement and sharing.
void ValidateNodeCapacity(const ValidationContext& ctx, tt::ARCH arch, uint32_t max_slots_per_core);

// ----------------------------------------------------------------------------
// These are the individual validator
// ----------------------------------------------------------------------------

// work_unit.cpp
void ValidateWorkUnitSpec(const WorkUnitSpec& work_unit, const ValidationContext& ctx, tt::ARCH arch);

// dfb.cpp
// Check the DFB slot limit across all resource nodes.
void ValidateDFBSlotsPerNode(const ValidationContext& ctx, uint32_t max_slots_per_core, tt::ARCH arch);

// kernel.cpp
void ValidateGen1DMPlacement(const ValidationContext& ctx, tt::ARCH arch);

// scratchpad.cpp
// Validate scratchpad placement overrides and sharing on the selected nodes.
// precondition: Every scratchpad is bound to at least one placed kernel.
void ValidateScratchpadPlacement(const ValidationContext& ctx, tt::ARCH arch);

}  // namespace tt::tt_metal::experimental
