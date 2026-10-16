// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

// Scratchpad placement census (multi-binding rule).
//
// By default, each node's scratchpad may be bound by only one kernel instance.
// The advanced multi-binding option permits shared backing memory on overlapping nodes;
// allocation groups those bindings and kernel code is responsible for synchronization.
// Duplicate bindings within one kernel are rejected earlier during collection.
void ValidateScratchpadBindersPerNode(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    for (const auto& scratchpad : spec.scratchpads) {
        if (scratchpad.advanced_options.allow_multi_kernel_binding) {
            continue;
        }
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
                "scratchpad across kernels requires advanced_options.allow_multi_kernel_binding = true "
                "and explicit synchronization in kernel code.",
                scratchpad.unique_id,
                kernels.size(),
                node.str(),
                names);
        }
    }
}

}  // namespace tt::tt_metal::experimental
