// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <set>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

// Validate DFB alias groups.
// Rules:
//  1. Transitivity: every DFB in an alias group must list every other member in its
//     alias_with field. This strict requirement is redundant by design. This is a
//     "dangerous" feature that a kernel author should use deliberately.
//  2. Same total size: entry_size * num_entries must match within a group.
//  3. Same node coverage: each DFB in the group must cover the same set of nodes
//  4. Consistent borrowed_from: either no member borrows, or all members borrow from
//     the same TensorParameter. (Aliased borrows from the same memory object is a
//     weird-but-valid scenario.)
// (That every alias_with name resolves, and is not the DFB itself, is checked in
// ValidateDFBSpec.)
void ValidateDFBAliasing(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    // The "extended group" of a DFB is its alias_with plus the DFB itself. Two DFBs
    // are in the same alias group iff their extended groups are equal.
    auto extended_group = [](const DataflowBufferSpec& d) {
        std::set<DFBSpecName> s(dfb_alias_with(d).begin(), dfb_alias_with(d).end());
        s.insert(d.unique_id);
        return s;
    };

    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb_alias_with(dfb).empty()) {
            continue;
        }
        const size_t total_size_a = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
        const auto group_a = extended_group(dfb);
        const auto& nodes_a = collected.dfb_node_set.at(dfb.unique_id);

        for (const auto& alias_name : dfb_alias_with(dfb)) {
            const DataflowBufferSpec* alias_spec = collected.dfb_by_name.at(alias_name);

            // Rule 1: full clique declaration.
            const auto group_b = extended_group(*alias_spec);
            if (group_a != group_b) {
                TT_THROW(
                    "DFBs '{}' and '{}' do not declare the same alias group. Every DFB in an "
                    "alias group must list every other member in its alias_with field.",
                    dfb.unique_id,
                    alias_name);
            }

            // Rule 2: same total size.
            const size_t total_size_b =
                static_cast<size_t>(alias_spec->entry_size) * static_cast<size_t>(alias_spec->num_entries);
            TT_FATAL(
                total_size_a == total_size_b,
                "Aliased DFBs '{}' and '{}' have different total sizes ({} vs {} bytes). "
                "Aliased DFBs must have the same total size (entry_size * num_entries).",
                dfb.unique_id,
                alias_name,
                total_size_a,
                total_size_b);

            // Rule 3: same node coverage.
            const auto& nodes_b = collected.dfb_node_set.at(alias_name);
            TT_FATAL(
                nodes_a == nodes_b,
                "Aliased DFBs '{}' and '{}' cover different sets of nodes. Aliased DFBs must "
                "cover the same node coverage (their bound kernels' WorkUnitSpec membership "
                "must yield identical target_nodes unions) — the shared L1 region must be "
                "reserved at the same cores for all members.",
                dfb.unique_id,
                alias_name);

            // Rule 4: consistent borrowed_from.
            TT_FATAL(
                dfb.borrowed_from == alias_spec->borrowed_from,
                "Aliased DFBs '{}' and '{}' have inconsistent borrowed_from. Either no member "
                "of an alias group borrows, or all members borrow from the same TensorParameter.",
                dfb.unique_id,
                alias_name);
        }
    }
}

}  // namespace tt::tt_metal::experimental
