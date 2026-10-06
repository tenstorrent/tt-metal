// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"

namespace tt::tt_metal::experimental {

// ----------------------------------------------------------------------------
// ValidateProgramSpec: Semantic validation
// ----------------------------------------------------------------------------
//
// Checks SEMANTIC rules (that don't affect the CollectedSpecData structure):
//   - Architecture requirements
//   - Resource limits
//   - Feature support
//   - Target node constraints (work_unit overlap, node coverage, node validity)
//
// Assumes CollectedSpecData is already built (see CollectSpecData).
void ValidateProgramSpec(
    const ProgramSpec& spec, const CollectedSpecData& collected, MetalContext& metal_ctx, const Allocator& allocator);

// TODO:
// These are moved out of CollectSpecData.
// I should moe this into ValidateProgramSpec.
void PostCollectionValidate(const ProgramSpec& spec, const CollectedSpecData& collected);

}  // namespace tt::tt_metal::experimental
