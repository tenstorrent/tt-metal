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
// A scratchpad is private to its bound kernel. Both DM and compute kernels may use
// scratchpads. The kernel accesses it via the device-side accessor:
//   auto s = Scratchpad(scratch::<accessor_name>);
//
// KERNEL BINDING: By default, a scratchpad instance is private to a single kernel
//   instance. It is legal for more than one KernelSpec to bind to the same
//   ScratchpadSpec only if they occupy disjoint node sets.
//
// LIFETIME: Program-scope. The backing memory is allocated for the Program's
//   execution lifetime, alongside DFBs (and from the same region of L1).
//
// INSTANCING: Like KernelSpec, a ScratchpadSpec is a *per-node template*. One
//   independent scratchpad instance is allocated per node where its bound kernel
//   runs, in that node's local SRAM ("L1").
//
// PLACEMENT: Derived from its bound kernel(s)'s WorkUnitSpec membership, like DFB.
//   Node-local resources (scratchpad, DFB) derive their node set from their
//   kernel bindings; cross-node resources (semaphore, cross-node DFB) must specify
//   their node set explicitly.
//
// CAUTION: Scratchpad is a raw memory with no synchronization semantics.
//   Be cautious when using it in compute kernels, as the Unpack/Math/Pack pipeline
//   stages run on different physical RISC-V cores. Likewise, be cautious when using
//   it in multi-threaded kernels, as each thread runs on different RISC-V core(s).
//
// ============================================================================

// A name identifying a ScratchpadSpec within a ProgramSpec.
using ScratchpadSpecName = ttsl::StrongType<std::string, struct ScratchpadSpecNameTag>;

struct ScratchpadSpec {
    // Scratchpad identifier: used to reference this scratchpad within the ProgramSpec
    ScratchpadSpecName unique_id;

    // Size of the SRAM ("L1") region reserved on each node, in bytes.
    // (Only occupies space on nodes where the scratchpad's bound kernel instances run.)
    uint32_t size_per_node = 0;
};

}  // namespace tt::tt_metal::experimental
