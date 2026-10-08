// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"

namespace tt::tt_metal::experimental {

// Represent hardware infomation/ context needed for validation.
// This is intended to allow black-box validation of programs.
//
// Program validation should be able to run without a live device.
// Which should be addressed in a device query api set.
struct ValidationHardwareMetadata {
    tt::ARCH arch = tt::ARCH::Invalid;

    // ---- placement/ ----

    // Used by work-unit validation
    CoreCoord compute_grid;

    // Used by dfb validation
    uint32_t max_dfb_slots_per_node = 0;

    // ---- resource/ ----

    // Used by prefetcher_pipe and dfb validation
    uint32_t l1_alignment = 0;

    // Used by dfb validation
    uint32_t num_l1_banks = 0;
    uint32_t num_l1_small_banks = 0;
};

using NumBanksFromBufferType = std::function<uint32_t(BufferType)>;
inline auto make_num_banks_from_buffer_type_fun(const Allocator& allocator) {
    return [&allocator](BufferType buffer_type) { return allocator.get_num_banks(buffer_type); };
}

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
// resource/ and placement/ declare their own validators in resource/resource.hpp and
// placement/placement.hpp.

struct ValidationContext {
    const ProgramSpec& spec;
    const CollectedSpecData& collected;
};

// kernel_spec.cpp
void ValidateKernelSpec(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch);

// hardware_config.cpp
void ValidateKernelHardwareConfig(const KernelSpec& kernel, const ValidationContext& ctx, tt::ARCH arch);

// program_spec.cpp
void ValidateProgramMisc(const ValidationContext& ctx);

}  // namespace tt::tt_metal::experimental
