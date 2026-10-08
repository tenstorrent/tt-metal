// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

// Scratchpad placement census (multi-binding rule).
//
// A scratchpad is private, node-local L1. More than one KernelSpec may bind the same
// ScratchpadSpec, but only on disjoint nodes: each node hosting the scratchpad must run exactly
// one binding kernel instance, so the per-node region stays private to that one kernel.
// (Allocation and CRTA delivery are per-binding-kernel — allocate_scratchpads stacks each
// kernel's scratchpad onto its own cores' allocators — so disjoint bindings never interact.) Two
// binding kernels on the same node would be true sharing, which is deferred behind a future
// AdvancedOption and rejected here. Mirrors the local-DFB per-node census. (A kernel
// binding the same scratchpad twice is already rejected during collection, so every binder here
// is a distinct kernel.)
void ValidateScratchpadBindersPerNode(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    for (const auto& scratchpad : spec.scratchpads) {
        auto binders_it = collected.scratchpad_binders.find(scratchpad.unique_id);
        if (binders_it == collected.scratchpad_binders.end() || binders_it->second.size() < 2) {
            continue;  // unbound (caught earlier) or single binder — always legal.
        }
        // Tally binding-kernel instances per node. A std::map keeps iteration (and error messages)
        // deterministic.
        std::map<NodeCoord, std::vector<const KernelSpec*>> binders_on_node;
        for (const KernelSpec* kernel : binders_it->second) {
            for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(kernel->unique_id))) {
                binders_on_node[node].push_back(kernel);
            }
        }
        for (const auto& [node, kernels] : binders_on_node) {
            if (kernels.size() <= 1) {
                continue;
            }
            std::string names;
            for (const KernelSpec* kernel : kernels) {
                names += (names.empty() ? "'" : ", '") + kernel->unique_id.get() + "'";
            }
            TT_FATAL(
                false,
                "ScratchpadSpec '{}' is bound by {} kernel instances on node {} ({}). A scratchpad is "
                "private node-local L1; multiple kernels may bind the same scratchpad only on disjoint "
                "nodes, so each node's instance stays private to one kernel. Sharing one node's "
                "scratchpad across kernels is not yet supported — give each binding kernel disjoint nodes.",
                scratchpad.unique_id,
                kernels.size(),
                node.str(),
                names);
        }
    }
}

}  // namespace tt::tt_metal::experimental
