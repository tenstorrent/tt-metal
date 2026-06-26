// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  Scratchpad API
// ============================================================================
//
// A ScratchpadSpec is a descriptor for a kernel scratchpad,
// which is allocated on the SRAM ("L1") with the lifetime of a Program.
//
// INSTANCING: Like KernelSpec, a ScratchpadSpec is a *per-node template*.
//   One independent scratchpad instance is allocated per node where its endpoint
//   kernels run, in that node's local SRAM.
//
// PLACEMENT: Derived — the scratchpad's effective node set is the union of its bound
//   kernels' WorkUnitSpec target_nodes.
//
// BINDING SCOPE: At most one kernel can bind to a given ScratchpadSpec.
//
// ============================================================================

// A name identifying a ScratchPadSpec within a ProgramSpec.
using ScratchpadSpecName = ttsl::StrongType<std::string, struct ScratchPadSpecTag>;

// Scratch pad is a program-scope resource allocated for a program.
struct ScratchpadSpec {
    // Scratchpad identifier: used to reference this Scratchpad within the ProgramSpec
    ScratchpadSpecName unique_id;

    // Size of SRAM reserved for each node
    uint32_t size_per_node;
};

}  // namespace tt::tt_metal::experimental
