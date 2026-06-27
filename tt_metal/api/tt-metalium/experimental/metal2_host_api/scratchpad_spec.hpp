// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  ScratchpadSpec API
// ============================================================================
//
// A ScratchpadSpec is a descriptor for a kernel scratchpad,
// which is allocated on the SRAM ("L1") with the lifetime of a Program.
//
// INSTANCING: Like KernelSpec, a ScratchpadSpec is a *per-node template*.
//   One independent scratchpad instance is allocated per node where its bound
//   kernel runs, in that node's local SRAM. Scratchpad is available for both data movement and compute kernels.
//
// PLACEMENT: Derived from its bound kernel's WorkUnitSpec membership.
//
// BINDING SCOPE: At most one kernel can bind to a given ScratchpadSpec. Unlike
//   DFBs and semaphores, a scratchpad is private to its single bound kernel.
//
// ============================================================================

// A name identifying a ScratchpadSpec within a ProgramSpec.
using ScratchpadSpecName = ttsl::StrongType<std::string, struct ScratchpadSpecTag>;

// Scratchpad is a program-scope resource allocated for a program.
struct ScratchpadSpec {
    // Scratchpad identifier: used to reference this Scratchpad within the ProgramSpec
    ScratchpadSpecName unique_id;

    // Size of SRAM reserved for each node
    uint32_t size_per_node;
};

}  // namespace tt::tt_metal::experimental
