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
// working memory.
//
// A scratchpad has a single property: its size (per node, in bytes). Its
// contents are uninitialized; the bound kernel must write before it reads. The
// kernel accesses it via the device-side accessor:
//   auto s = Scratchpad(scratch::<accessor_name>);
//
// LIFETIME: Program-scope. The backing memory is allocated for the Program's
//   execution lifetime, alongside DFBs (and from the same region of L1).
//
// INSTANCING: Like KernelSpec, a ScratchpadSpec is a *per-node template*. One
//   independent scratchpad instance is allocated per node where its bound kernel
//   runs, in that node's local SRAM ("L1").
//
// BINDING: By default, a scratchpad instance is private to a single kernel
//   instance. It is legal for more than one KernelSpec to bind to the same
//   ScratchpadSpec only if they occupy disjoint node sets.
//
// PLACEMENT: Derived. Scratchpad is a *node-local resource* so the scratchpad's
//   node set is inferred from the bound kernels' WorkUnitSpec target_nodes.
//   (This is the same convention as DFB, also a node-local resource. Semaphore
//   and cross-node DFB are non-node-local resources; they assign target nodes
//   explicitly in their specs.)
//
// ============================================================================

// A name identifying a ScratchpadSpec within a ProgramSpec.
using ScratchpadSpecName = ttsl::StrongType<std::string, struct ScratchpadSpecNameTag>;

struct ScratchpadSpec {
    // Scratchpad identifier: used to reference this scratchpad within the ProgramSpec
    ScratchpadSpecName unique_id;

    // Size of the SRAM ("L1") region reserved on each node, in bytes.
    uint32_t size_per_node = 0;
};

}  // namespace tt::tt_metal::experimental
