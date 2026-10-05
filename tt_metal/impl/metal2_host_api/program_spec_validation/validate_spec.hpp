// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"

namespace tt::tt_metal::experimental {

// ============================================================================
// Constants
// ============================================================================

// TODO: These constants should be queriable from the public API (currently HAL, for consistency)
//       They are currently also hardcoded in the temporary Quasar host_api.hpp. Need to clean this up.
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_RESERVED_DM_CORES_PER_NODE = 2;  // DM0 and DM1 reserved for internal use
static constexpr uint32_t QUASAR_USER_DM_CORES_PER_NODE = QUASAR_DM_CORES_PER_NODE - QUASAR_RESERVED_DM_CORES_PER_NODE;
static constexpr uint32_t QUASAR_TENSIX_ENGINES_PER_NODE = 4;

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

}  // namespace tt::tt_metal::experimental
