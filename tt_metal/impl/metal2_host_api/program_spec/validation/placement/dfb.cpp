// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

// Device slots are allocated per core (two DFBs may share a slot iff their node sets are
// disjoint), so the arch limit is on how many DFBs land on any single node — not on
// ProgramSpec::dataflow_buffers.size(). Gen1 lowers each slot to a circular buffer; Gen2
// indexes the packed config by device slot up to dfb::NUM_DFBS. Tile-counter exhaustion on
// Gen2 is still checked later at enqueue.
void ValidateDFBSlotsPerNode(
    const WorkUnitSpec& work_unit, const ValidationContext& ctx, uint32_t max_slots_per_core, tt::ARCH arch) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    const NodeRangeSet nodes = to_node_range_set(work_unit.target_nodes);
    std::unordered_map<NodeCoord, uint32_t> dfbs_per_node;
    for (const auto& dfb : spec.dataflow_buffers) {
        for (const NodeCoord& node : corerange_to_cores(collected.dfb_node_set.at(dfb.unique_id).intersection(nodes))) {
            dfbs_per_node[node]++;
        }
    }

    for (const auto& [node, count] : dfbs_per_node) {
        if (count <= max_slots_per_core) {
            continue;
        }
        if (is_gen1_arch(arch)) {
            TT_THROW(
                "ProgramSpec '{}' places {} DataflowBufferSpecs on node ({}, {}), but Gen1 "
                "supports at most {} device slots per core (disjoint cores may reuse slots).",
                spec.name,
                count,
                node.x,
                node.y,
                max_slots_per_core);
        } else if (is_gen2_arch(arch)) {
            TT_THROW(
                "ProgramSpec '{}' places {} DataflowBufferSpecs on node ({}, {}), but the "
                "target architecture supports at most {} device slots per core. The true "
                "limit is also configuration-dependent (tile counters) and is checked at "
                "enqueue.",
                spec.name,
                count,
                node.x,
                node.y,
                max_slots_per_core);
        } else {
            TT_FATAL(false, "Unknown architecture");
        }
    }
}

}  // namespace tt::tt_metal::experimental
