// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"
#include "llrt/hal.hpp"

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

// ----------------------------------------------------------------------------
// Per-domain validators, called by ValidateProgramSpec
// ----------------------------------------------------------------------------
//
// resource/ declares its own validators in resource/resource.hpp.

struct ValidationContext {
    const ProgramSpec& spec;
    const CollectedSpecData& collected;
    MetalContext& metal_ctx;
    const Hal& hal;
    const Allocator& allocator;
};

// placement/work_unit.cpp
void ValidateWorkUnitFields(const ValidationContext& ctx);
void ValidateWorkUnitSpec(const WorkUnitSpec& work_unit, const ValidationContext& ctx);

// placement/dfb.cpp
void ValidateDFBSlotsPerNode(const WorkUnitSpec& work_unit, const ValidationContext& ctx);

// placement/kernel.cpp
void ValidateGen1DMPlacement(const ValidationContext& ctx);

// placement/scratchpad.cpp
void ValidateScratchpadBindersPerNode(const ValidationContext& ctx);

// kernel_spec.cpp
void ValidateKernelSpec(const KernelSpec& kernel, const ValidationContext& ctx);

// hardware_config.cpp
void ValidateKernelHardwareConfig(const KernelSpec& kernel, const ValidationContext& ctx);

// program_spec.cpp
void ValidateProgramMisc(const ValidationContext& ctx);

}  // namespace tt::tt_metal::experimental
