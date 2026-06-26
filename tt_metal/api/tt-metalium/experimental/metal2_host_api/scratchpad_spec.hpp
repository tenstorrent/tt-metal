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
// A ScratchpadSpec is a descriptor for a kernel scratchpad: a private,
// uninitialized region of node-local SRAM ("L1") for a kernel to use as
// working memory. It is the sanctioned primitive for "I just need some private
// L1" — replacing the practice of abusing a dataflow buffer (DFB) as scratch.
//
// A scratchpad has a single property: its size (per node, in bytes). Its
// contents are uninitialized; the bound kernel must write before it reads. The
// kernel accesses it via the device-side accessor:
//   auto s = Scratchpad(scratch::<accessor_name>);
//
// LIFETIME: Program-scope. The L1 backing is allocated for the Program's
//   execution lifetime, alongside DFBs (and from the same end of L1).
//
// BINDING SCOPE: A scratchpad is private to a single kernel — it is bound by
//   exactly one kernel, via one ScratchpadBinding (see KernelSpec). (This is
//   deliberately stricter than a DFB, which permits multiple same-role endpoint
//   bindings; a scratchpad has no cross-kernel sharing to express.)
//
// INSTANCING: Like KernelSpec, a ScratchpadSpec is a *per-node template*. One
//   independent scratchpad instance is allocated per node where its bound kernel
//   runs, in that node's local L1.
//
// PLACEMENT: Derived — the scratchpad's node set is the bound kernel's
//   WorkUnitSpec target_nodes. (Like a DFB, a scratchpad infers its nodes from
//   its binding; unlike a semaphore, which declares them.)
//
// ============================================================================

// A name identifying a ScratchpadSpec within a ProgramSpec.
using ScratchpadSpecName = ttsl::StrongType<std::string, struct ScratchpadSpecNameTag>;

struct ScratchpadSpec {
    // Scratchpad identifier: used to reference this scratchpad within the ProgramSpec
    ScratchpadSpecName unique_id;

    // Size of the L1 region reserved on each node, in bytes.
    uint32_t size_per_node = 0;
};

}  // namespace tt::tt_metal::experimental
