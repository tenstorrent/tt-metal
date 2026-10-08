// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/processor_assignment/processor_assignment.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental {

// Kernel -> ComputeEngineMask map (Gen2/Quasar only).
// DM masks flow through KernelCouplingGroup (equivalence class) rather than per-KernelSpec, so the DM
// counterpart of this map is defined in the dm_solver namespace and keyed by KernelCouplingGroup*.
using ComputeEngineMaskMap = std::unordered_map<const KernelSpec*, ComputeEngineMask>;

// Reserve the n lowest-indexed cores not already in use.
template <std::size_t NUM_CORES>
std::optional<std::bitset<NUM_CORES>> ReserveProcessors(uint8_t n, const std::bitset<NUM_CORES>& already_in_use) {
    if (NUM_CORES - already_in_use.count() < n) {
        return std::nullopt;
    }

    std::bitset<NUM_CORES> newly_reserved;
    for (std::size_t i = 0; i < NUM_CORES && n > 0; i++) {
        if (!already_in_use.test(i)) {
            newly_reserved.set(i);
            n--;
        }
    }
    return newly_reserved;
}

// Reserve DM processors for a kernel on a WorkUnitSpec.
// Returns {this_kernel_mask, updated_cumulative_mask}
// Throws TT_FATAL on conflict or allocation failure (see simplifying assumption notes)
std::pair<DMProcessorMask, DMProcessorMask> ReserveDMProcessors(
    const KernelSpec* kernel_spec,
    std::optional<DMProcessorMask> existing_mask,
    DMProcessorMask cumulative_mask,
    const std::string& work_unit_id) {
    // Was this kernel already assigned a mask from a previous WorkUnitSpec?
    if (existing_mask.has_value()) {
        DMProcessorMask existing = existing_mask.value();

        // Check for conflict with what's already allocated on the current WorkUnitSpec
        TT_FATAL(
            (existing & cumulative_mask).none(),
            "Kernel '{}' requires processors already in use on WorkUnitSpec '{}'. "
            "One of the following must be true: \n"
            " - The ProgramSpec is invalid, and the legality checks were bypassed. \n"
            " - A solution exists, but the greedy algorithm failed to find it. \n"
            " - The runtime's \"common DM cores\" assumption has been violated!",
            kernel_spec->unique_id,
            work_unit_id);

        // Return existing mask and updated cumulative
        return {existing, cumulative_mask | existing};
    }

    // First time seeing this kernel - reserve new processors
    std::optional<DMProcessorMask> reserved = ReserveProcessors(kernel_spec->num_threads, cumulative_mask);
    TT_FATAL(
        reserved.has_value(),
        "Failed to reserve processors for DM kernel '{}' on WorkUnitSpec '{}'. "
        "The \"common DM cores\" assumption has been violated!",
        kernel_spec->unique_id,
        work_unit_id);

    DMProcessorMask mask = reserved.value();
    return {mask, cumulative_mask | mask};
}

// Assign compute processor mask for a kernel.
ComputeEngineMask AssignComputeProcessors(const KernelSpec* kernel_spec, const KernelSpecName& kernel_name) {
    auto reserved = ReserveProcessors(kernel_spec->num_threads, ComputeEngineMask{});
    TT_FATAL(
        reserved.has_value(),
        "Compute kernel '{}' reservation failed. Condition should be unreachable after validation.",
        kernel_name);
    return reserved.value();
}

// ----------------------------------------------------------------------------
// DM Processor Assignment
// ----------------------------------------------------------------------------
//
// Solves kernel-to-core assignments for DM kernels. Guaranteed to find a valid
// assignment if one exists under the "simplifying assumption" (each DM kernel
// uses the same processor cores on all nodes it targets).
//
// Approach:
//   1. (Optional) Sort kernels by "most constrained first" (more nodes, more threads)
//   2. Use greedy assignment: pick first available cores for each kernel
//   3. If greedy fails, backtrack by trying different kernel orderings
//
// Sorting step is optional. Not yet sure whether it's useful or not.
// It may be that straight greedy is sufficient in a majority of cases
// ----------------------------------------------------------------------------

namespace dm_solver {

// Map from kernel name to its derived effective node set (union of containing
// WorkUnitSpec target_nodes). Used by the solver to read each kernel's placement.
using KernelNodeSetMap = std::unordered_map<KernelSpecName, NodeRangeSet>;

// Equivalence class of DM kernels coupled by shared DFB endpoint roles.
//
// All DM kernels bound to the same DFB on the same role (PRODUCER/CONSUMER) must end up
// with identical DM RISC masks — the DFB's hardware config carries a single mask per role.
// Membership is computed by union-find over DM kernels: two kernels are merged if they share
// any DFB endpoint role; the transitive closure yields KernelCouplingGroups.
//
// num_threads is uniform within a group (the per-DFB-side num_threads validator + transitive
// equality guarantees this for any chain of shared endpoints).
//
// The DM solver operates on KernelCouplingGroups instead of individual KernelSpecs: each group is
// assigned one DMProcessorMask, which then applies to every member. A non-multi-bound DM
// kernel ends up in a singleton group.
struct KernelCouplingGroup {
    std::vector<const KernelSpec*> members;  // ≥ 1; canonical member is members.front()
    NodeRangeSet merged_node_set;            // union of members' node sets
    uint8_t num_threads = 0;                 // shared across members
};

// State for tracking per-node processor usage
class NodeUsageTracker {
public:
    DMProcessorMask& get_used_mask(const NodeCoord& node) {
        if (!node_used_masks_.contains(node)) {
            node_used_masks_[node] = DMProcessorMask{0b11};  // Reserve DM0, DM1
        }
        return node_used_masks_[node];
    }

    // Compute union of used masks across all target nodes
    DMProcessorMask get_combined_used_mask(const NodeRangeSet& target_nodes) {
        DMProcessorMask combined_used;
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                combined_used = combined_used | get_used_mask(node);
            }
        }
        return combined_used;
    }

    // Mark cores as used on all target nodes
    void mark_used(const NodeRangeSet& target_nodes, DMProcessorMask mask) {
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                get_used_mask(node) |= mask;
            }
        }
    }

    // Unmark cores on all target nodes (for backtracking)
    void unmark_used(const NodeRangeSet& target_nodes, DMProcessorMask mask) {
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                get_used_mask(node) &= ~mask;
            }
        }
    }

    void reset() { node_used_masks_.clear(); }

private:
    std::map<NodeCoord, DMProcessorMask> node_used_masks_;
};

// Result map: one DMProcessorMask per KernelCouplingGroup (which expands to its member kernels).
using KernelCouplingGroupMaskMap = std::unordered_map<const KernelCouplingGroup*, DMProcessorMask>;

// Constraint score for sorting: higher = more constrained (assigned earlier)
int ConstraintScore(const KernelCouplingGroup* g) {
    int node_count = static_cast<int>(g->merged_node_set.num_cores());
    int thread_count = static_cast<int>(g->num_threads);
    return (node_count * 100) + thread_count;  // nodes dominate, threads break ties
}

// Deterministic tiebreaker: sort by the lexicographically-smallest member unique_id.
// Group::members is canonicalized at construction time so members.front() is the sort key.
const std::string& group_sort_key(const KernelCouplingGroup* g) { return g->members.front()->unique_id.get(); }

void SortByConstraint(std::vector<const KernelCouplingGroup*>& groups) {
    std::sort(groups.begin(), groups.end(), [](const KernelCouplingGroup* a, const KernelCouplingGroup* b) {
        int score_a = ConstraintScore(a);
        int score_b = ConstraintScore(b);
        if (score_a != score_b) {
            return score_a > score_b;  // Higher score first
        }
        return group_sort_key(a) < group_sort_key(b);
    });
}

// Try to assign all groups in the given order using greedy selection.
// Returns true if successful, populates result map.
bool TryGreedyAssignment(
    const std::vector<const KernelCouplingGroup*>& group_order,
    NodeUsageTracker& tracker,
    KernelCouplingGroupMaskMap& result) {
    for (const KernelCouplingGroup* group : group_order) {
        DMProcessorMask combined_used = tracker.get_combined_used_mask(group->merged_node_set);

        auto selected = ReserveProcessors(group->num_threads, combined_used);
        if (!selected.has_value()) {
            return false;  // Can't assign this group
        }

        result[group] = selected.value();
        tracker.mark_used(group->merged_node_set, selected.value());
    }
    return true;
}

// Backtracking solver over kernel coupling group orderings.
// Note: In the worst case, this is O(N!) in the number of groups.
//       In practice, I expect this will almost always solve in the first greedy attempt (if sorted).
//       The backtracking is just here for pathological cases.
//       Even then, it shouldn't be horrendous. We won't have a huge number of kernels in a ProgramSpec.
//       And in the common case (traced), Program creation isn't on the critical path.
//       We can revisit if this ever becomes a problem.
bool SolveWithOrderingBacktrack(
    std::vector<const KernelCouplingGroup*> groups,  // by value - we'll permute it
    NodeUsageTracker& tracker,
    KernelCouplingGroupMaskMap& result) {
    // Try current ordering
    if (TryGreedyAssignment(groups, tracker, result)) {
        return true;
    }

    // Backtrack: try all permutations.
    // (std::next_permutation requires sorted input.)
    auto by_name = [](const KernelCouplingGroup* a, const KernelCouplingGroup* b) {
        return group_sort_key(a) < group_sort_key(b);
    };
    std::sort(groups.begin(), groups.end(), by_name);
    do {
        tracker.reset();
        result.clear();
        if (TryGreedyAssignment(groups, tracker, result)) {
            return true;
        }
    } while (std::next_permutation(groups.begin(), groups.end(), by_name));

    return false;
}

// Build DM kernel groups via union-find over shared DFB endpoint roles.
//
// Two DM kernels are merged if they share a DFB endpoint role (both PRODUCER of the same DFB,
// or both CONSUMER of the same DFB). The transitive closure yields equivalence classes.
//
// Compute kernels are not eligible — they don't participate in the DM solver. (Compute kernels
// bound to the same DFB role share num_threads, which makes AssignComputeProcessors deterministic
// across them, so no equivalence-class machinery is needed for compute.)
std::vector<KernelCouplingGroup> BuildDMKernelCouplingGroups(
    const std::vector<const KernelSpec*>& dm_kernels,
    const CollectedSpecData& collected,
    const KernelNodeSetMap& kernel_node_set) {
    // Small N — flat union-find indexed by position in dm_kernels.
    std::unordered_map<const KernelSpec*, size_t> idx_of;
    idx_of.reserve(dm_kernels.size());
    for (size_t i = 0; i < dm_kernels.size(); ++i) {
        idx_of[dm_kernels[i]] = i;
    }

    std::vector<size_t> parent(dm_kernels.size());
    std::iota(parent.begin(), parent.end(), size_t{0});
    auto find = [&parent](size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path compression
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](size_t a, size_t b) {
        size_t ra = find(a), rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    };

    // For each DFB, union all DM kernels on each side (PRODUCER, CONSUMER) independently.
    auto union_same_side = [&](const auto& endpoints) {
        std::optional<size_t> anchor;
        for (const auto& rec : endpoints) {
            if (!rec.kernel->is_data_movement_kernel()) {
                continue;
            }
            const size_t k = idx_of.at(rec.kernel);
            if (!anchor.has_value()) {
                anchor = k;
            } else {
                unite(anchor.value(), k);
            }
        }
    };
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        union_same_side(endpoint_info.producers);
        union_same_side(endpoint_info.consumers);
    }

    // Collect classes: root index → group.
    // Iterate dm_kernels in given order to preserve a deterministic per-class member order.
    std::unordered_map<size_t, size_t> root_to_group_idx;
    std::vector<KernelCouplingGroup> groups;
    groups.reserve(dm_kernels.size());
    for (size_t i = 0; i < dm_kernels.size(); ++i) {
        const size_t r = find(i);
        auto [it, inserted] = root_to_group_idx.try_emplace(r, groups.size());
        if (inserted) {
            groups.emplace_back();
        }
        groups[it->second].members.push_back(dm_kernels[i]);
    }

    // Finalize each group: merged_node_set + num_threads + canonical member sort.
    for (auto& g : groups) {
        std::sort(g.members.begin(), g.members.end(), [](const KernelSpec* a, const KernelSpec* b) {
            return a->unique_id < b->unique_id;
        });
        for (const KernelSpec* k : g.members) {
            g.merged_node_set = g.merged_node_set.merge(kernel_node_set.at(k->unique_id));
        }
        g.num_threads = g.members.front()->num_threads;
    }
    return groups;
}

}  // namespace dm_solver

namespace {

// Gen2 (Quasar) processor assignment: runs the backtracking DM solver and returns
// a KernelRiscMaskMap using the Gen2 bit encoding (DM: bits 0-7, compute: bits 8-15).
//
// The DM solver operates on KernelCouplingGroups (equivalence classes of DM kernels coupled by shared
// DFB endpoint roles), not individual kernels. Each group is assigned one DMProcessorMask
// which then applies to every member kernel. This ensures multi-bound same-role kernels end
// up with identical masks, matching the per-side single-mask shape of DataflowBufferConfig.
KernelRiscMaskMap SolveGen2KernelRiscMasks(const ProgramSpec& spec, const CollectedSpecData& collected) {
    ComputeEngineMaskMap compute_assignments;

    // Collect DM kernels and compute kernels separately.
    // Compute kernels get a deterministic per-kernel mask (one compute kernel per node assumption;
    // num_threads uniformity is enforced upstream, so same-role compute kernels get identical masks
    // without any coupling-group machinery).
    std::vector<const KernelSpec*> dm_kernels;
    dm_kernels.reserve(spec.kernels.size());
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            dm_kernels.push_back(&kernel);
        } else {
            compute_assignments[&kernel] = AssignComputeProcessors(&kernel, kernel.unique_id);
        }
    }

    // Build DM kernel groups (equivalence classes via shared DFB endpoint roles).
    // Each group's merged_node_set is the union of its members' node sets — the solver
    // will pick a mask that's free on every node in that union.
    std::vector<dm_solver::KernelCouplingGroup> groups =
        dm_solver::BuildDMKernelCouplingGroups(dm_kernels, collected, collected.kernel_node_set);

    std::vector<const dm_solver::KernelCouplingGroup*> group_ptrs;
    group_ptrs.reserve(groups.size());
    for (const auto& g : groups) {
        group_ptrs.push_back(&g);
    }

    // Sort by constraint score (most constrained first)
    constexpr bool kSortByConstraint = true;  // Toggle to disable upfront sorting
    if constexpr (kSortByConstraint) {
        dm_solver::SortByConstraint(group_ptrs);
    }

    // Solve DM assignments at the group level
    dm_solver::NodeUsageTracker tracker;
    dm_solver::KernelCouplingGroupMaskMap group_assignments;
    bool success = dm_solver::SolveWithOrderingBacktrack(group_ptrs, tracker, group_assignments);

    TT_FATAL(
        success,
        "Failed to find valid processor assignments for DM kernels. "
        "Either the ProgramSpec is invalid, or that the \"same DM cores on every node\" "
        "simplifying assumption has been violated.");

    // Convert to KernelRiscMaskMap using Gen2 bit encoding: expand each group's mask to all members.
    KernelRiscMaskMap result;
    for (const auto& [group, mask] : group_assignments) {
        for (const KernelSpec* member : group->members) {
            result[member] = static_cast<uint16_t>(mask.to_ulong());  // DM processors in bits 0-7
        }
    }
    for (const auto& [kernel, mask] : compute_assignments) {
        result[kernel] = static_cast<uint16_t>(mask.to_ulong() << 8);  // Compute engines in bits 8-15
    }

    // For multi-binding DFBs, all KernelSpecs on the same role must end up with identical risc_masks:
    // the DFB has a single producer_risc_mask / consumer_risc_mask in its hardware config. The solver
    // gives every member of a DM coupling group the same DM mask, and compute masks are deterministic
    // from num_threads (uniform per role), so this holds by construction; checked defensively.
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.advanced_options.allow_instance_multi_binding) {
            continue;
        }
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        auto check_uniform_mask = [&](const auto& records, std::string_view role) {
            if (records.size() < 2) {
                return;
            }
            const uint16_t first_mask = result.at(records[0].kernel);
            const auto* first_kernel = records[0].kernel;
            for (size_t i = 1; i < records.size(); ++i) {
                const uint16_t mask = result.at(records[i].kernel);
                if (mask == first_mask) {
                    continue;
                }
                TT_THROW(
                    "Internal error: Gen2 solver produced disagreeing risc_masks for DFB '{}' "
                    "{} bindings ('{}' = 0x{:x} vs '{}' = 0x{:x}). The coupling-group solver "
                    "extension should guarantee per-role mask uniformity by construction.",
                    dfb.unique_id,
                    role,
                    first_kernel->unique_id,
                    first_mask,
                    records[i].kernel->unique_id,
                    mask);
            }
        };
        check_uniform_mask(endpoints.producers, "PRODUCER");
        check_uniform_mask(endpoints.consumers, "CONSUMER");
    }
    return result;
}

// Gen1 (WH/BH) processor assignment: read the kernel's gen1_config processor and return a
// KernelRiscMaskMap using the Gen1 bit encoding (RISCV_0: bit 0, RISCV_1: bit 1, compute: bit 2).
KernelRiscMaskMap BuildGen1KernelRiscMasks(const ProgramSpec& spec) {
    static constexpr uint8_t GEN1_COMPUTE_RISC_BIT = 2;

    KernelRiscMaskMap result;
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            TT_FATAL(
                dm_config.config_1xx.has_value(),
                "KernelSpec '{}' is a data-movement kernel on Gen1 but has no config_1xx processor/NOC. "
                "Those settings are required to build a Gen1 data-movement kernel. Supply a DataMovement1XXConfig "
                "(e.g. CreateReaderDataMovementConfig()/CreateWriterDataMovementConfig()).",
                kernel.unique_id);
            const auto& gen1 = *dm_config.config_1xx;
            result[&kernel] = static_cast<uint16_t>(1u << static_cast<uint8_t>(gen1.processor));
        } else {
            result[&kernel] = static_cast<uint16_t>(1u << GEN1_COMPUTE_RISC_BIT);
        }
    }
    return result;
}

}  // namespace

KernelRiscMaskMap SolveKernelRiscMasks(const ProgramSpec& spec, const CollectedSpecData& collected, const Hal& hal) {
    return is_gen2_arch(hal) ? SolveGen2KernelRiscMasks(spec, collected) : BuildGen1KernelRiscMasks(spec);
}

}  // namespace tt::tt_metal::experimental
