// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_construction/construct_program.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>  // HalMemType, for the borrowed-DFB per-bank sizing check
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

// ============================================================================
// Type Definitions
// ============================================================================

// Bitmask for tracking processor allocation on a node
template <uint8_t NUM_CORES>
struct ProcessorMask {
    static_assert(NUM_CORES > 0 && NUM_CORES <= 8, "ProcessorMask supports 1-8 processors");
    static constexpr uint8_t VALID_BITS_MASK = (NUM_CORES == 8) ? 0xFF : ((1 << NUM_CORES) - 1);

    uint8_t bits = 0x00;

    // Operators
    bool operator==(ProcessorMask other) const { return bits == other.bits; }
    bool operator!=(ProcessorMask other) const { return bits != other.bits; }
    ProcessorMask operator|(ProcessorMask other) const { return {uint8_t(bits | other.bits)}; }
    ProcessorMask operator&(ProcessorMask other) const { return {uint8_t(bits & other.bits)}; }
    ProcessorMask operator~() const { return {uint8_t(~bits & VALID_BITS_MASK)}; }
    ProcessorMask& operator|=(ProcessorMask other) {
        bits |= other.bits;
        return *this;
    }
    ProcessorMask& operator&=(ProcessorMask other) {
        bits &= other.bits;
        return *this;
    }

    // Queries
    uint8_t num_in_use() const { return std::popcount(bits); }
    uint8_t num_available() const { return NUM_CORES - num_in_use(); }
    bool is_idx_available(uint8_t idx) const { return (bits & (1 << idx)) == 0; }
    bool is_idx_in_use(uint8_t idx) const { return (bits & (1 << idx)) != 0; }
    bool conflicts_with(ProcessorMask other) const { return (bits & other.bits) != 0; }
};

using DMProcessorMask = ProcessorMask<QUASAR_DM_CORES_PER_NODE>;
using ComputeEngineMask = ProcessorMask<QUASAR_TENSIX_ENGINES_PER_NODE>;

// Kernel -> ProcessorMask map (Gen2/Quasar only).
// DM masks flow through KernelCouplingGroup (equivalence class) rather than per-KernelSpec, so the DM
// counterpart of this map is defined in the dm_solver namespace and keyed by KernelCouplingGroup*.
using ComputeEngineMaskMap = std::unordered_map<const KernelSpec*, ComputeEngineMask>;

// Kernel -> DFB risc mask (passed to MakeDataflowBufferConfig)
//   Gen1: bit 0 = RISCV_0 (BRISC), bit 1 = RISCV_1 (NCRISC), bit 2 = Tensix compute
//   Gen2: bits 0-7 = DM processors, bits 8-15 = Tensix compute engines
using KernelRiscMaskMap = std::unordered_map<const KernelSpec*, uint16_t>;

// DFB name -> program-wide DFB ID map (host-side identity: aliasing, borrowed bindings)
using DFBNameToIdMap = std::unordered_map<DFBSpecName, uint32_t>;
// DFB name -> device slot map. The slot is what a kernel sees (the dfb::<name> accessor value) and
// what indexes the per-core config table, so it is what device-facing lowering must use.
using DFBNameToSlotMap = std::unordered_map<DFBSpecName, uint32_t>;
using SemaphoreNameToIdMap = std::unordered_map<SemaphoreSpecName, uint32_t>;

// ============================================================================
// Step 2: Processor Assignment
// ============================================================================
// TODO: move this to it's own file.

// ProcessorMask factory functions
template <uint8_t NUM_CORES>
ProcessorMask<NUM_CORES> CreateMask(uint8_t mask) {
    TT_FATAL(
        mask <= ProcessorMask<NUM_CORES>::VALID_BITS_MASK,
        "Mask specifies too many cores for ProcessorMask<{}>: {}",
        NUM_CORES,
        mask);
    return {mask};
}

template <uint8_t NUM_CORES>
std::optional<ProcessorMask<NUM_CORES>> ReserveProcessors(uint8_t n, const ProcessorMask<NUM_CORES>& already_in_use) {
    if (already_in_use.num_available() < n) {
        return std::nullopt;
    }

    ProcessorMask<NUM_CORES> newly_reserved;
    for (uint8_t i = 0; i < NUM_CORES && n > 0; i++) {
        if (already_in_use.is_idx_available(i)) {
            newly_reserved.bits |= (1 << i);
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
            !existing.conflicts_with(cumulative_mask),
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
    auto reserved = ReserveProcessors(kernel_spec->num_threads, CreateMask<QUASAR_TENSIX_ENGINES_PER_NODE>(0x00));
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
            node_used_masks_[node] = CreateMask<QUASAR_DM_CORES_PER_NODE>(0x03);  // Reserve DM0, DM1
        }
        return node_used_masks_[node];
    }

    // Compute union of used masks across all target nodes
    DMProcessorMask get_combined_used_mask(const NodeRangeSet& target_nodes) {
        DMProcessorMask combined_used = CreateMask<QUASAR_DM_CORES_PER_NODE>(0x00);
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
            result[member] = mask.bits;  // DM processors in bits 0-7
        }
    }
    for (const auto& [kernel, mask] : compute_assignments) {
        result[kernel] = static_cast<uint16_t>(mask.bits) << 8;  // Compute engines in bits 8-15
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

// ============================================================================
// Step 3: Program Building Helpers
// ============================================================================

std::optional<LLKMetadata> LLKMetadataFromDfb(const DataflowBufferSpec& spec) {
    if (!spec.data_format_metadata.has_value()) {
        TT_FATAL(
            !spec.tile_format_metadata.has_value(),
            "DFB '{}' need to have a configured data_format_metadata for it's tile_format_metadata to be respected",
            spec.unique_id);
        return std::nullopt;
    }
    const Tile tile = spec.tile_format_metadata.value_or(Tile{});
    return LLKMetadata{.format = *spec.data_format_metadata, .tile = tile};
}

std::optional<LLKMetadata> LLKMetadataFromScratchpad(const ScratchpadSpec& spec) {
    if (!spec.data_format_metadata.has_value()) {
        TT_FATAL(
            !spec.tile_format_metadata.has_value(),
            "Scratchpad '{}' need to have a configured data_format_metadata for it's tile_format_metadata to be "
            "respected",
            spec.unique_id);
        return std::nullopt;
    }
    const Tile tile = spec.tile_format_metadata.value_or(Tile{});
    return LLKMetadata{.format = *spec.data_format_metadata, .tile = tile};
}

LLKMetadata LLKMetadataFromTensorSpec(const TensorSpec& spec) {
    return LLKMetadata{.format = datatype_to_dataformat_converter(spec.data_type()), .tile = spec.tile()};
}

// Per-TensorParameter resolved layout.
//
// cta_payload: positional CTA words appended to the kernel's compile-time args for
// each binding of this TensorParameter. Mirrors what TensorAccessorArgs::append_to
// would produce on the legacy path, but is built from the TensorSpec + MeshDevice
// since no Buffer exists at spec-build time.
//
// extra_crta_words: additional CRTA words (beyond the always-present base address
// slot) that this binding occupies, used by the device-side accessor to read
// runtime-resolved fields. Non-zero when the TensorParameter opts into a dynamic
// field that lives in CRTAs: either sharded + dynamic_tensor_shape (which puts
// `rank` shape words in CRTAs), or interleaved row-major + dynamic_tensor_shape (one
// page-size word). The two are mutually exclusive per binding -- see runtime_field_is_page_size.
struct ResolvedTensorParameter {
    std::vector<uint32_t> cta_payload;

    // How many CRTA words (beyond the base address) does this binding consume?
    // This is only used if TensorParameter relaxations have been requested.
    uint32_t extra_crta_words = 0;

    // What info the runtime field CRTA words actually contain depends on the relaxation.
    // Currently, there are only two mutually exclusive possibilities (though more may be added):
    //  1. The interleaved row-major page-size (one CRTA only)
    //  2. The sharded dynamic_tensor_shape shape (one CRTA per tensor dim)
    // For now, since there are only two mutually exclusive possibilities, it's sufficient to
    // distinguish them with a boolean.
    bool runtime_field_is_page_size = false;

    // Compile-time LLK metadata derived from the TensorParameter's spec, baked onto the binding token.
    // Always present: a tensor has a dtype and a tile.
    LLKMetadata llk_metadata;
};

// Resolve a TensorParameter's static layout into a CTA payload + an extra CRTA word
// count for any runtime-resolved fields.
//
// CTA layout produced:
//  - word 0 is the args_config raw byte
//  - word 1 is aligned_page_size
// For sharded tensors only:
//  - word 2 is rank
//  - word 3 is num_banks
//  - remaining words: per-dim tensor_shape_in_pages (omitted if dynamic_tensor_shape),
//    per-dim shard_shape_in_pages, and packed bank coordinates (two per uint32)
//
// The tensor base address always lives in CRTAs (filled in per-enqueue from the
// corresponding TensorArgument). When dynamic_tensor_shape is set on a sharded tensor,
// the runtime tensor's shape is also written into CRTAs at enqueue time, in
// `extra_crta_words` slots immediately after the address slot.
// (See also ResolveTensorBindingsForKernel below.)
ResolvedTensorParameter ResolveTensorParameterStaticCTAs(
    const TensorParameter& tensor_parameter, const distributed::MeshDevice& mesh_device) {
    const TensorSpec& spec = tensor_parameter.spec;
    const MemoryConfig& memory_config = spec.memory_config();
    const BufferType buffer_type = memory_config.buffer_type();
    const bool is_dram = (buffer_type == BufferType::DRAM);
    const bool is_sharded = memory_config.is_sharded();
    // The tensor SHAPE only rides the CTA/CRTA payload for a sharded tensor: an interleaved payload
    // never carried it in the first place, and the device-side accessor doesn't read it. So this
    // particular induction is sharded-only.
    //
    // That is a statement about the shape words, NOT about the flag. dynamic_tensor_shape is a
    // dynamic relaxation on every layout -- see dyn_page immediately below, which moves an
    // interleaved ROW-MAJOR page size out of the CTAs. (match_padded_shape_only is the flag that is
    // purely a host-side validation loosening with no CTA/CRTA effect; do not transplant its
    // description onto this one.)
    const bool dyn_shape = tensor_parameter.relaxations.dynamic_tensor_shape && is_sharded;
    // dynamic_tensor_shape lets the bound tensor's logical shape vary. For an interleaved ROW-MAJOR
    // tensor the page size (= last_dim_width * elem_size) is part of that varying shape, so it must
    // ride a runtime CRTA word too -- otherwise it goes stale on a program-cache hit and the
    // accessor strides by the wrong number of bytes. We induce that here rather than expose a flag
    // for it: a useful page-size change is ALWAYS a shape change on row-major (you can't vary the
    // width without varying the logical shape), so there is no "page size varies but shape doesn't"
    // case to give a flag to. Tiled page size is dtype-fixed and sharded page size is spec-fixed, so
    // neither triggers this; sharded dynamic_tensor_shape carries shape-in-pages words instead
    // (dyn_shape above). dyn_shape and dyn_page are mutually exclusive by layout.
    //
    // match_page_size opts out of the induction, for the CONVERSE case: shape varies, width does
    // not. That implication runs only one way, so the flag does not reopen the reasoning above --
    // it declares a narrower equivalence class in which the page size is pinned, and the match
    // enforces it (tensor_spec_relaxations.cpp), so the CTA below cannot go stale.
    const bool dyn_page = tensor_parameter.relaxations.dynamic_tensor_shape &&
                          !tensor_parameter.relaxations.match_page_size && !is_sharded &&
                          spec.layout() == Layout::ROW_MAJOR;

    tensor_accessor::ArgsConfig args_config;
    if (is_sharded) {
        args_config.set(tensor_accessor::ArgConfig::Sharded);
    }
    if (is_dram) {
        args_config.set(tensor_accessor::ArgConfig::IsDram);
    }
    if (dyn_shape) {
        args_config.set(tensor_accessor::ArgConfig::RuntimeTensorShape);
    }
    if (dyn_page) {
        args_config.set(tensor_accessor::ArgConfig::RuntimePageSize);
    }

    // aligned_page_size: align the unaligned page size up to the buffer-type alignment.
    const size_t unaligned_page_size = spec.compute_page_size_bytes();
    const uint32_t alignment = mesh_device.allocator()->get_alignment(buffer_type);
    const size_t aligned_page_size = align(unaligned_page_size, static_cast<size_t>(alignment));
    TT_FATAL(
        aligned_page_size <= std::numeric_limits<uint32_t>::max(),
        "TensorParameter '{}' aligned page size {} exceeds uint32_t max {}",
        tensor_parameter.unique_id,
        aligned_page_size,
        std::numeric_limits<uint32_t>::max());

    ResolvedTensorParameter result{.llk_metadata = LLKMetadataFromTensorSpec(spec)};
    std::vector<uint32_t>& cta_payload = result.cta_payload;

    // Common header (always emitted, sharded or not):
    cta_payload.push_back(args_config.raw());
    // If the page size is static, it rides as a CTA.
    // (If it's dynamic, it will live in a CRTA word instead.)
    if (!dyn_page) {
        cta_payload.push_back(static_cast<uint32_t>(aligned_page_size));
    } else {
        TT_FATAL(!is_sharded, "Internal error: dynamic page size should not occur on a sharded tensor parameter");

        // One runtime field: the page size, re-derived from the bound buffer each dispatch
        // and emitted immediately after the base-address word (see EmitBindingCrtaValues).
        result.extra_crta_words = 1;
        result.runtime_field_is_page_size = true;
    }

    // The rest of the logic in this function pertains to sharded tensors only.
    // Early return for a non-sharded tensor.
    if (!is_sharded) {
        return result;
    }

    //////////////////////////////////
    // Sharded tensor handling
    //////////////////////////////////

    // Sharded: emit rank, num_banks, tensor_shape_in_pages (CTA only when static),
    // shard_shape_in_pages, bank_coords.
    const BufferShardingArgs sharding_args = spec.compute_buffer_sharding_args();
    const std::optional<BufferDistributionSpec>& bds_opt = sharding_args.buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "TensorParameter '{}' is sharded but TensorSpec produced no BufferDistributionSpec",
        tensor_parameter.unique_id);
    const BufferDistributionSpec& bds = *bds_opt;

    const Shape& tensor_shape = bds.tensor_shape_in_pages();
    const Shape& shard_shape = bds.shard_shape_in_pages();
    const std::vector<CoreCoord>& bank_coords = bds.cores();
    const size_t rank = tensor_shape.rank();
    const size_t n_banks = bank_coords.size();

    cta_payload.push_back(static_cast<uint32_t>(rank));
    TT_FATAL(
        n_banks < tensor_accessor::ShardContiguousBit,
        "TensorParameter '{}' has too many banks ({}) to pack the shard-contiguous flag",
        tensor_parameter.unique_id,
        n_banks);
    cta_payload.push_back(tensor_accessor::pack_num_banks(
        static_cast<uint32_t>(n_banks), bds.shard_distribution_strategy() == ShardDistributionStrategy::CONTIGUOUS_1D));

    if (!dyn_shape) {
        for (size_t i = 0; i < rank; ++i) {
            cta_payload.push_back(static_cast<uint32_t>(tensor_shape[i]));
        }
    } else {
        // Shape lives in CRTAs (one word per dim, written at enqueue time from the bound MeshTensor).
        result.extra_crta_words = static_cast<uint32_t>(rank);
    }
    for (size_t i = 0; i < rank; ++i) {
        cta_payload.push_back(static_cast<uint32_t>(shard_shape[i]));
    }

    // Bank coords packed two-per-uint32.
    // Non-DRAM coords are virtualized; DRAM coords are kept logical (DRAM bank id == logical x).
    const CoreType core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;
    auto resolve_coord = [&](const CoreCoord& logical) -> CoreCoord {
        if (is_dram) {
            return logical;
        }
        return mesh_device.virtual_core_from_logical_core(logical, core_type);
    };
    for (size_t i = 0; i < n_banks; i += 2) {
        const CoreCoord c1 = resolve_coord(bank_coords[i]);
        if (i + 1 < n_banks) {
            const CoreCoord c2 = resolve_coord(bank_coords[i + 1]);
            cta_payload.push_back(((c2.x & 0xFF) << 24) | ((c2.y & 0xFF) << 16) | ((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        } else {
            cta_payload.push_back(((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        }
    }

    return result;
}

// Per-kernel resolved tensor binding data:
//  - All the kernel's TensorBindingHandle (type is defined in kernel.hpp)
//  - The positional CTAs to append to the kernel's (unnamed) CTAs
//  - The full CRTA buffer layout (named CRTAs + binding section + vararg-section start),
//    precomputed here so consumers (headergen, runtime) don't have to re-derive section
//    boundaries by walking handles. See KernelCrtaLayout in jit_build_settings.hpp.
struct TensorBindingsForKernel {
    std::vector<TensorBindingHandle> handles;
    // Binding-only CTA payload; appended after the user CTA-vararg positional prefix.
    std::vector<uint32_t> cta_words;
    KernelCrtaLayout crta_layout;
};

// Resolve the tensor bindings for a single kernel:
//  1. Walk the kernel's tensor_bindings in declaration order
//  2. Pack each binding's CTA payload into a contiguous positional buffer
//  3. Assign each binding a slot in the kernel's TensorBinding section of the CRTA buffer.
//     The section is structurally separate, immediately following the user-named CRTAs.
//     Each binding occupies (1 + extra_crta_words) words: the always-present base address
//     word, plus any runtime accessor field words (e.g. shape, when dynamic_tensor_shape
//     is set on a sharded TensorParameter).
//  4. Record the resulting CRTA buffer layout (the three section sizes) on the output, so
//     the headergen and runtime can consult it directly instead of re-summing the bindings.
//
// (SetProgramRunArgs will fill the address slots and runtime field slots at enqueue
// time, extracting info from the TensorArgs.)
TensorBindingsForKernel ResolveTensorBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<TensorParamName, ResolvedTensorParameter>& resolved_tensor_parameters,
    size_t base_named_crta_count,
    uint32_t base_cta_offset) {
    TensorBindingsForKernel out;
    out.handles.reserve(kernel.tensor_bindings.size());

    // Absolute word index into the unified positional CTA buffer:
    //   [ CTA varargs (base_cta_offset words) | TensorBinding payloads ... ]
    uint32_t cta_word_offset = base_cta_offset;
    size_t crta_word_index = base_named_crta_count;
    uint32_t binding_section_words = 0;
    for (const auto& binding : kernel.tensor_bindings) {
        const ResolvedTensorParameter& resolved = resolved_tensor_parameters.at(binding.tensor_parameter_name);
        const std::vector<uint32_t>& binding_ctas = resolved.cta_payload;

        TensorBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.tensor_parameter_name = binding.tensor_parameter_name.get();
        handle.cta_offset = cta_word_offset;
        handle.addr_crta_offset = static_cast<uint32_t>(crta_word_index * sizeof(uint32_t));
        handle.num_runtime_field_crta_words = resolved.extra_crta_words;
        handle.runtime_field_is_page_size = resolved.runtime_field_is_page_size;
        handle.llk_metadata = resolved.llk_metadata;

        out.cta_words.insert(out.cta_words.end(), binding_ctas.begin(), binding_ctas.end());
        cta_word_offset += static_cast<uint32_t>(binding_ctas.size());
        const uint32_t binding_words = 1u + resolved.extra_crta_words;
        crta_word_index += binding_words;
        binding_section_words += binding_words;

        out.handles.push_back(std::move(handle));
    }

    out.crta_layout.num_named_words = static_cast<uint32_t>(base_named_crta_count);
    out.crta_layout.binding_section_words = binding_section_words;
    out.crta_layout.vararg_section_offset = static_cast<uint32_t>(base_named_crta_count) + binding_section_words;

    return out;
}

// Per-kernel resolved scratchpad bindings:
//  - one CRTA word per binding (the scratchpad's allocated L1 base address), in declaration order
//  - the scratchpad section sits immediately after the TensorBinding section and before varargs, so
//    each binding's absolute CRTA word index (and thus addr_crta_word) is fixed at codegen time
//    (varargs are open-ended / runtime-counted, so a section placed after them would not be).
// The allocated_address is left 0 here; allocate_scratchpads fills it once L1 is allocated.
struct ScratchpadBindingsForKernel {
    std::vector<ScratchpadBindingHandle> handles;
    uint32_t section_words = 0;  // == number of scratchpad bindings
};

ScratchpadBindingsForKernel ResolveScratchpadBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*>& scratchpad_by_name,
    size_t scratchpad_base_crta_word) {
    ScratchpadBindingsForKernel out;
    out.handles.reserve(kernel.scratchpad_bindings.size());

    size_t crta_word_index = scratchpad_base_crta_word;
    for (const auto& binding : kernel.scratchpad_bindings) {
        const ScratchpadSpec* scratchpad_spec = scratchpad_by_name.at(binding.scratchpad_spec_name);

        ScratchpadBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.size_bytes = scratchpad_spec->size_per_node;
        handle.addr_crta_word = static_cast<uint32_t>(crta_word_index);
        handle.llk_metadata = LLKMetadataFromScratchpad(*scratchpad_spec);
        // handle.allocated_address stays 0 until allocate_scratchpads runs.
        out.handles.push_back(std::move(handle));
        crta_word_index += 1;  // one address word per scratchpad binding
    }

    out.section_words = static_cast<uint32_t>(kernel.scratchpad_bindings.size());
    return out;
}

// Create map of local accessor name -> DFB device slot. This is the value baked into the kernel's
// dfb::<name> accessor, so it must be the device slot rather than the program-wide id.
// `dfb_name_to_is_relay` marks CrossNode/PrefetcherPipe relay locals so codegen emits
// RelayDFBBindingToken instead of DFBBindingToken.
// `dfb_name_to_prefetcher_pipe_id` carries the PrefetcherPipe slot for PrefetcherPipe relays (0xFF
// otherwise) so TRISC construction can align to the durable checkpoint.
tt::tt_metal::DataflowBufferBindingHandleMap MakeDataflowBufferBindingHandles(
    const KernelSpec& kernel_spec,
    const DFBNameToSlotMap& dfb_name_to_slot,
    const std::unordered_map<DFBSpecName, bool>& dfb_name_to_is_relay,
    const std::unordered_map<DFBSpecName, uint8_t>& dfb_name_to_prefetcher_pipe_id,
    const std::unordered_map<DFBSpecName, const DataflowBufferSpec*>& dfb_by_name) {
    tt::tt_metal::DataflowBufferBindingHandleMap out;
    out.reserve(kernel_spec.dfb_bindings.size());
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        const uint32_t slot = dfb_name_to_slot.at(dfb_binding.dfb_spec_name);
        TT_FATAL(
            slot <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' DFB '{}' device slot {} does not fit uint16_t",
            kernel_spec.unique_id,
            dfb_binding.dfb_spec_name,
            slot);
        tt::tt_metal::DataflowBufferBindingHandle handle;
        handle.logical_dfb_id = static_cast<uint16_t>(slot);
        handle.is_relay = dfb_name_to_is_relay.at(dfb_binding.dfb_spec_name);
        handle.prefetcher_pipe_id = dfb_name_to_prefetcher_pipe_id.at(dfb_binding.dfb_spec_name);
        if (!handle.is_relay) {
            handle.llk_metadata = LLKMetadataFromDfb(*dfb_by_name.at(dfb_binding.dfb_spec_name));
        }
        out.emplace(dfb_binding.accessor_name, handle);
    }
    return out;
}

// Create map of accessor name -> semaphore handle: the logical id, the resolved scope,
// and the binder hart count for local cached semaphores.
tt::tt_metal::SemaphoreBindingHandleMap MakeSemaphoreBindingHandles(
    const KernelSpec& kernel_spec,
    const sem_solver::SemaphoreBinderCensus& semaphore_binders,
    const SemaphoreNameToIdMap& semaphore_name_to_id,
    const sem_solver::SemaphoreNameToScopeMap& semaphore_name_to_scope) {
    tt::tt_metal::SemaphoreBindingHandleMap out;
    out.reserve(kernel_spec.semaphore_bindings.size());
    for (const auto& semaphore_binding : kernel_spec.semaphore_bindings) {
        const uint32_t id = semaphore_name_to_id.at(semaphore_binding.semaphore_spec_name);
        TT_FATAL(
            id <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' semaphore '{}' id {} does not fit uint16_t",
            kernel_spec.unique_id,
            semaphore_binding.semaphore_spec_name,
            id);
        const SemScope scope = semaphore_name_to_scope.at(semaphore_binding.semaphore_spec_name);
        const uint32_t total_binder_harts =
            scope == SemScope::DM_LOCAL_CACHED
                ? sem_solver::BinderHartCount(semaphore_binders, semaphore_binding.semaphore_spec_name)
                : 0u;
        TT_FATAL(
            total_binder_harts <= 0x7FFFu,
            "Semaphore '{}' has {} binder harts; the cached seed protocol supports at most 32767",
            semaphore_binding.semaphore_spec_name,
            total_binder_harts);
        out.emplace(
            semaphore_binding.accessor_name,
            tt::tt_metal::SemaphoreBindingHandle{static_cast<uint16_t>(id), scope, total_binder_harts});
    }
    return out;
}

// Create a DataflowBufferConfig from a DataflowBufferSpec and endpoint info.
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const KernelRiscMaskMap& kernel_to_risc_mask) {
    // With multi-binding, all same-role KernelSpecs share kind (DM/compute), access_pattern,
    // num_threads, and risc_mask. The first three are enforced in ValidateProgramSpec; the
    // fourth is solver-guaranteed on Gen2 (the coupling-group equivalence-class constraint)
    // and user-validated on Gen1 (see Step 2b in MakeProgramFromSpec). So any
    // representative producer/consumer gives the correct DFB config — we take the first.
    const KernelSpec* producer = dfb_endpoint_info.producers.front().kernel;
    const KernelSpec* consumer = dfb_endpoint_info.consumers.front().kernel;
    const DFBBinding* producer_binding = dfb_endpoint_info.producers.front().binding;
    const DFBBinding* consumer_binding = dfb_endpoint_info.consumers.front().binding;

    uint16_t producer_risc_mask = kernel_to_risc_mask.at(producer);
    uint16_t consumer_risc_mask = kernel_to_risc_mask.at(consumer);

    // Convert user-facing access pattern enum to hardware interface access pattern enum
    // (TODO: We should merge these enums; it's silly to have separate ones.)
    auto to_hw_access_pattern = [](DFBAccessPattern pattern) -> experimental::dfb::AccessPattern {
        switch (pattern) {
            case DFBAccessPattern::STRIDED: return experimental::dfb::AccessPattern::STRIDED;
            case DFBAccessPattern::ALL: return experimental::dfb::AccessPattern::ALL;
            case DFBAccessPattern::BLOCKED: TT_FATAL(false, "BLOCKED access pattern is not yet supported");
        }
        TT_FATAL(false, "Unknown DFBAccessPattern");
    };
    auto producer_access_pattern = to_hw_access_pattern(producer_binding->access_pattern);
    auto consumer_access_pattern = to_hw_access_pattern(consumer_binding->access_pattern);

    // A compute kernel that self-loops a DFB (binds it as both producer and consumer) lowers to the
    // intra-Tensix packer->unpacker flow, so the lower-layer DFB API needs TensixScope::INTRA. The
    // Metal 2.0 surface does not expose a scope option — INTRA is the only supported topology, applied
    // automatically here. Self-loop is detected as any overlap between the producer and consumer kernel
    // sets — under the multi-binding regime the first-record pointers may differ even when the kernel
    // sets are identical (the overlap is what matters, not vector ordering). Upstream validation
    // guarantees producer set == consumer set whenever any overlap exists, so reading from the first
    // producer is safe and representative. A DM self-loop (Gen1-only) needs no tensix_scope.
    const bool is_self_loop = [&] {
        for (const auto& p : dfb_endpoint_info.producers) {
            for (const auto& c : dfb_endpoint_info.consumers) {
                if (p.kernel == c.kernel) {
                    return true;
                }
            }
        }
        return false;
    }();
    std::optional<experimental::dfb::TensixScope> tensix_scope;
    if (is_self_loop && producer->is_compute_kernel()) {
        tensix_scope = experimental::dfb::TensixScope::INTRA;
    }

    // Compute the per-side implicit-sync value by polling the bound DM kernels' Gen2 votes.
    // Sides with no DM endpoints get implicit_sync=false (no DM endpoint to enable it for).
    // Validator guarantees per-side agreement among DM kernels, so any DM kernel's vote works.
    auto side_implicit_sync_enabled =
        [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints) -> bool {
        bool any_dm = false;
        bool disabled = false;
        for (const auto& ep : endpoints) {
            if (!ep.kernel->is_data_movement_kernel()) {
                continue;
            }
            any_dm = true;
            const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
            // config_2xx is unused on Gen1; implicit sync stays at the current default.
            if (DmKernelDisablesImplicitSync(dm_config, dfb_spec->unique_id)) {
                disabled = true;
            }
        }
        return any_dm && !disabled;
    };
    return experimental::dfb::DataflowBufferConfig{
        .entry_size = dfb_spec->entry_size,
        .num_entries = dfb_spec->num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = static_cast<uint8_t>(producer->num_threads),
        .pap = producer_access_pattern,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = static_cast<uint8_t>(consumer->num_threads),
        .cap = consumer_access_pattern,
        .enable_producer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.producers),
        .enable_consumer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.consumers),
        .data_format = dfb_spec->data_format_metadata.value_or(tt::DataFormat::Invalid),
        .tile = dfb_spec->tile_format_metadata,
        .tensix_scope = tensix_scope,
        // DFB borrowed memory mode is declared at program creation time.
        // The actual backing memory L1 address is attached at runtime: from the borrowed
        // TensorParameter's MeshTensor, or (relay) from the PrefetcherPipe ring the relay aliases.
        .borrows_memory =
            dfb_spec->borrowed_from.has_value() || !dfb_spec->advanced_options.prefetcher_pipe_relays.empty(),
        // A PrefetcherPipe relay is lane-interleaved (producer h owns entries h, h+P, ...).
        .is_relay = !dfb_spec->advanced_options.prefetcher_pipe_relays.empty()};
}

// ----------------------------------------------------------------------------
// MakeKernelSource: Create a KernelSource from a KernelSpec
// ----------------------------------------------------------------------------

KernelSource MakeKernelSource(const KernelSpec& kernel_spec, ContextId context_id) {
    return std::visit(
        [&](const auto& src) -> KernelSource {
            using T = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<T, std::filesystem::path>) {
                TT_FATAL(!src.empty(), "KernelSpec '{}' has empty source file path", kernel_spec.unique_id);
                return KernelSource::from_path(context_id, src);
            } else if constexpr (std::is_same_v<T, KernelSpec::SourceCode>) {
                TT_FATAL(!src.code.empty(), "KernelSpec '{}' has empty inline source code", kernel_spec.unique_id);
                return KernelSource::from_source(src.code);
            } else {
                static_assert(!sizeof(T*), "Unhandled KernelSpec::source alternative");
            }
        },
        kernel_spec.source);
}

// ----------------------------------------------------------------------------
// MakeGen1DataMovementConfig: Create a DataMovementConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

// (Temporary) Shims
// ProgramSpec APIs use vector<pair> for conceptually map-like data structures.
// This is deliberate, done so ProgramSpec stays hashable for TTNN's program caching.
// For now, just convert to the map types that the core runtime expects.
// TODO: Fix this inefficiency eventually.
std::unordered_map<std::string, uint32_t> to_named_compile_args_map(const KernelSpec::CompileTimeArgs& bindings) {
    return std::unordered_map<std::string, uint32_t>(bindings.begin(), bindings.end());
}
std::map<std::string, std::string> to_defines_map(const KernelSpec::CompilerOptions::Defines& defines) {
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}

DataMovementConfig MakeGen1DataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");
    const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel_spec.hw_config);
    TT_FATAL(
        dm_config.config_1xx.has_value(),
        "KernelSpec '{}' is a data-movement kernel on Gen1 but has no config_1xx processor/NOC. "
        "Those settings are required to build a Gen1 data-movement kernel. Supply a DataMovement1XXConfig "
        "(e.g. CreateReaderDataMovementConfig()/CreateWriterDataMovementConfig()).",
        kernel_spec.unique_id);
    const auto& gen1 = *dm_config.config_1xx;

    return DataMovementConfig{
        .processor = gen1.processor,
        .noc = gen1.noc,
        .noc_mode = gen1.noc_mode,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// BuildUnpackToDestModeVector:
// Translate the Metal 2.0 user-facing DFB-name->mode map into the (gross)
// CB-indexed vector that the JIT data-format machinery expects.
//
// This DFB/CB translation layer is confusing. The gory details:
//   - The JIT consumer (get_unpack_dst_formats) will read unpack_to_dest_mode at
//     index cb_id, where cb_id is the slot used by set_dfb_data_fmt_and_tile
//     in buf_dataformat_arr (aka, dfb->id).
//   - The unpack_mode for a DFB "d" needs to be at unpack_modes[d->id]
//   - The vector must be at least max_dfbs long, or the consumer gets angry
//     (it iterates buf_formats up to max_dfbs).
//   - This is true on WH, BH, and Quasar. (Yes, Quasar too.)
//
// What is the max DFB slot count?
//   - WH/BH: Hardcoded as max_dfbs. Different number on WH vs. BH.
//   - Quasar has a variable cap, based on tile-counter registers.
//     In actual practice, we'll run out LONG before we get the HAL-reported
//     limit of 64.
// ----------------------------------------------------------------------------

std::vector<UnpackToDestMode> BuildUnpackToDestModeVector(
    const ComputeHardwareConfig::ComputeUnpackModes& user_modes,
    const DFBNameToSlotMap& dfb_name_to_slot,
    const Hal& hal) {
    const uint32_t max_dfbs = hal.get_num_dataflow_buffers();
    std::vector<UnpackToDestMode> unpack_modes(max_dfbs, UnpackToDestMode::Default);
    for (const auto& [dfb_name, mode] : user_modes) {
        // Indexed by device slot: this vector is consumed by the HLK alongside the CB-indexed data
        // formats, which set_dfb_data_fmt_and_tile also keys by slot.
        uint32_t dfb_slot = dfb_name_to_slot.at(dfb_name);
        // This TT_FATAL is unreachable, provided that validation wasn't skipped.
        TT_FATAL(
            dfb_slot < max_dfbs,
            "Internal Error: DFB '{}' has device slot {} which exceeds the JIT data-format "
            "slot count ({}); compute kernels cannot reference DFBs past this limit",
            dfb_name,
            dfb_slot,
            max_dfbs);
        // Public UnpackMode -> internal UnpackToDestMode. UnpackToDest keeps full FP32 by
        // unpacking straight to Dest; UnpackToSrc is the SrcA/B path (the internal "Default").
        unpack_modes[dfb_slot] =
            (mode == UnpackMode::UnpackToDest) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    }
    return unpack_modes;
}

// ----------------------------------------------------------------------------
// MakeGen1ComputeConfig: Create a ComputeConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

ComputeConfig MakeGen1ComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToSlotMap& dfb_name_to_slot, const Hal& hal) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);

    std::vector<UnpackToDestMode> unpack_dst_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_modes, dfb_name_to_slot, hal);

    // bfp_pack_precision_mode is TT-1.x.x-only. If config_1xx is not engaged, use the
    // historical default (Approximate).
    const Precision bfp_pack_precision_mode = compute_config.config_1xx.has_value()
                                                  ? compute_config.config_1xx->bfp_pack_precision_mode
                                                  : Precision::Approximate;

    return ComputeConfig{
        .math_fidelity = compute_config.fpu_math_fidelity,
        .fp32_dest_acc_en = compute_config.enable_32_bit_dest,
        .dst_full_sync_en = !compute_config.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .bfp8_pack_precise = (bfp_pack_precision_mode == Precision::Precise),
        .math_approx_mode = (compute_config.sfpu_precision_mode == Precision::Approximate),
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeQuasarDataMovementConfig: Create a QuasarDataMovementConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");

    return experimental::quasar::QuasarDataMovementConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .is_legacy_kernel = false,
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeGen2ComputeConfig: Create a QuasarComputeConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarComputeConfig MakeGen2ComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToSlotMap& dfb_name_to_slot, const Hal& hal) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);

    std::vector<UnpackToDestMode> unpack_dst_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_modes, dfb_name_to_slot, hal);

    return experimental::quasar::QuasarComputeConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .math_fidelity = compute_config.fpu_math_fidelity,
        .fp32_dest_acc_en = compute_config.enable_32_bit_dest,
        .dst_full_sync_en = !compute_config.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .math_approx_mode = (compute_config.sfpu_precision_mode == Precision::Approximate),
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// --------------------------------------------------------------------------------
// GetDMProcessorSet: Convert a DMProcessorMask to a set of DataMovementProcessor
// --------------------------------------------------------------------------------

std::set<DataMovementProcessor> GetDMProcessorSet(DMProcessorMask mask) {
    std::set<DataMovementProcessor> processors;
    for (uint8_t i = 0; i < QUASAR_DM_CORES_PER_NODE; ++i) {
        if (mask.is_idx_in_use(i)) {
            processors.insert(static_cast<DataMovementProcessor>(i));
        }
    }
    return processors;
}

// ------------------------------------------------------------------------------------------
// GetComputeProcessorSet: Convert a ComputeEngineMask to a set of QuasarComputeProcessor
// ------------------------------------------------------------------------------------------
//
// The ComputeEngineMask represents the active Tensix engines on a node.
// (Based on the number of compute kernel threads running on that node.)
// Each Tensix engine has 4 compute processors.
// So if bit i is set in the mask, we include all 4 processors for that engine:
//   Engine 0 -> NEO_0_COMPUTE_{0,1,2,3}
//   Engine 1 -> NEO_1_COMPUTE_{0,1,2,3}
//   etc.

std::set<experimental::quasar::QuasarComputeProcessor> GetComputeProcessorSet(ComputeEngineMask mask) {
    using QuasarComputeProcessor = experimental::quasar::QuasarComputeProcessor;
    constexpr uint8_t PROCESSORS_PER_ENGINE = experimental::quasar::QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE;

    std::set<QuasarComputeProcessor> processors;
    for (uint8_t engine = 0; engine < QUASAR_TENSIX_ENGINES_PER_NODE; ++engine) {
        if (mask.is_idx_in_use(engine)) {
            // Add all 4 compute processors for this engine
            for (uint8_t proc = 0; proc < PROCESSORS_PER_ENGINE; ++proc) {
                uint8_t processor_id = (engine * PROCESSORS_PER_ENGINE) + proc;
                processors.insert(static_cast<QuasarComputeProcessor>(processor_id));
            }
        }
    }
    return processors;
}

namespace {

// ----------------------------------------------------------------------------
// ReservePrefetcherPipeSlots: PrefetcherPipeParameters -> Program slots
// ----------------------------------------------------------------------------
//
// One Program slot per accessor group (a KernelAdvancedOptions::PrefetcherPipeBinding), reserved on the
// kernel's nodes from spec geometry alone. The group's role (sender / receiver; validated exact
// by ValidateProgramSpec) decides the slot's receiver cores and credit lanes P (the receiver
// kernel's num_threads). A relay DFB whose prefetcher_pipe_relays equals the group's pipe set is
// registered against the slot (its base address is supplied when the pipe binds).
//
// Every parameter named by the group is recorded with the slot and the cores it owns inside the
// kernel's nodes (its sender node or its receivers -- the group tiles the nodes, so this is one
// pipe per node). SetProgramRunArgs later binds the supplied pipe object onto exactly those cores,
// so a multi-pipe accessor resolves per node on the host and the kernel binary sees one slot.
using PrefetcherPipeHandlesByKernel =
    std::unordered_map<const KernelSpec*, std::vector<tt::tt_metal::PrefetcherPipeBindingHandle>>;

PrefetcherPipeHandlesByKernel ReservePrefetcherPipeSlots(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    detail::ProgramImpl& program_impl,
    const DFBNameToIdMap& dfb_name_to_id) {
    PrefetcherPipeHandlesByKernel handles;
    if (spec.advanced_options.prefetcher_pipe_parameters.empty()) {
        return handles;
    }

    // Per-parameter placement, accumulated across the accessor groups that name it.
    std::unordered_map<PrefetcherPipeParamName, detail::ProgramImpl::PrefetcherPipeParameterBinding> placements;
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        placements[pipe.unique_id] = detail::ProgramImpl::PrefetcherPipeParameterBinding{
            .device = &mesh_device,
            .receivers = to_node_range_set(pipe.receivers),
            .ring_size = pipe.ring_size,
            .slots = {},
            .bound_pipe = nullptr};
    }

    // Relay DFBs keyed by their (sorted) relayed pipe set, so a group can find its relay.
    auto sorted_names = [](std::vector<PrefetcherPipeParamName> names) {
        std::sort(names.begin(), names.end());
        return names;
    };
    std::map<std::vector<PrefetcherPipeParamName>, const DataflowBufferSpec*> relay_by_pipe_set;
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            continue;
        }
        auto [it, inserted] =
            relay_by_pipe_set.try_emplace(sorted_names(dfb.advanced_options.prefetcher_pipe_relays), &dfb);
        TT_FATAL(
            inserted,
            "DFBs '{}' and '{}' both relay the same PrefetcherPipe set; a pipe set has at most one relay DFB",
            it->second->unique_id,
            dfb.unique_id);
    }
    std::unordered_set<const DataflowBufferSpec*> relays_registered;

    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.advanced_options.prefetcher_pipe_bindings.empty()) {
            continue;
        }
        const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            const PrefetcherPipeParameter* first =
                collected.prefetcher_pipe_by_name.at(binding.pipe_parameter_names[0]);
            NodeRangeSet group_receivers;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                group_receivers = group_receivers.merge(to_node_range_set(pipe->receivers));
            }
            const bool is_sender_role =
                !is_prefetcher_pipe_receiver_role(nodes, group_receivers) &&
                is_prefetcher_pipe_sender_role(nodes, group_receivers, binding.pipe_parameter_names.size());

            const NodeRangeSet receiver_cores = is_sender_role ? NodeRangeSet() : nodes;
            const uint32_t num_credit_lanes = is_sender_role ? 1u : kernel.num_threads;
            const uint8_t prefetcher_pipe_id = program_impl.reserve_prefetcher_pipe_slot(
                nodes, receiver_cores, first->ring_size, first->entry_size, num_credit_lanes);
            handles[&kernel].push_back(
                {.accessor_name = binding.accessor_name, .prefetcher_pipe_id = prefetcher_pipe_id});

            // Sender role: the spec does not say which of the kernel's nodes hosts which pipe, so
            // every pipe's placement names all of them; the bind narrows it to the pipe's sender.
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                placements.at(pipe_name).slots.push_back(
                    {.prefetcher_pipe_id = prefetcher_pipe_id,
                     .cores = is_sender_role ? nodes : to_node_range_set(pipe->receivers),
                     .sender_role = is_sender_role});
            }

            // A relay over exactly this group's pipes hangs off the receiver kernel's slot.
            if (!is_sender_role) {
                auto relay_it = relay_by_pipe_set.find(sorted_names(binding.pipe_parameter_names));
                if (relay_it != relay_by_pipe_set.end()) {
                    const DataflowBufferSpec* relay = relay_it->second;
                    TT_FATAL(
                        relays_registered.insert(relay).second,
                        "Relay DFB '{}' matches PrefetcherPipe accessor groups in more than one receiver kernel",
                        relay->unique_id);
                    program_impl.register_prefetcher_pipe_relay_dfb(
                        prefetcher_pipe_id, dfb_name_to_id.at(relay->unique_id));
                }
            }
        }
    }

    for (const auto& dfb : spec.dataflow_buffers) {
        if (!dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            TT_FATAL(
                relays_registered.contains(&dfb),
                "Relay DFB '{}' has no data-movement kernel binding its relayed PrefetcherPipe set as receiver; the "
                "relay's PRODUCER must bind those pipes under one accessor",
                dfb.unique_id);
        }
    }

    for (auto& [pipe_name, placement] : placements) {
        program_impl.register_prefetcher_pipe_parameter(pipe_name.get(), std::move(placement));
    }
    return handles;
}

}  // namespace

Program BuildProgram(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    [[maybe_unused]] MetalContext& metal_ctx) {
    // Arch and the Emule check come from the mesh device's env, not the default context.
    MetalEnvImpl& metal_env = mesh_device.impl().metal_env();
    const Hal& hal = metal_env.get_hal();
    // Step 2a: Build kernel risc masks (arch-specific)
    //  - Gen2: backtracking solver assigns DM cores automatically
    //  - Gen1: processor is user-specified in Gen1Config
    KernelRiscMaskMap kernel_to_risc_mask =
        is_gen2_arch(hal) ? SolveGen2KernelRiscMasks(spec, collected) : BuildGen1KernelRiscMasks(spec);

    // Step 2b: For multi-binding DFBs, all KernelSpecs on the same role must end up with
    // identical risc_masks. The DFB has a single producer_risc_mask / consumer_risc_mask in
    // its hardware config.
    //
    // Gen1: the mask is a deterministic function of the user's KernelSpec hw_config (compute
    //   placement is fixed; DM processor is user-specified via Gen1Config). A
    //   mismatch is a user error — incompatible processor placement across multi-bound kernels.
    // Gen2 (Quasar): the mask is solver-assigned, with the solver constrained to give every
    //   member of a DM coupling-group equivalence class the same DM mask. Compute kernel masks
    //   are deterministic from num_threads, which is uniform per role. So on Gen2 the uniformity
    //   property is guaranteed by construction; the check is retained as a defensive assertion.
    for (const auto& dfb : spec.dataflow_buffers) {
        // Instance-multi-binding (Gen1-only) intentionally binds same-role kernels on distinct RISCs
        // (e.g. a BRISC producer and an NCRISC producer on one node), so their risc_masks differ by
        // design and the uniform-mask requirement does not apply. On Gen1 the DFB lowers to a plain
        // circular buffer where the mask is inert (it never reaches the device blob), so the single
        // representative mask MakeDataflowBufferConfig takes from the first binding is harmless. (The
        // flag is rejected on Gen2 in ValidateProgramSpec, so under normal validation any DFB reaching
        // here with it set is Gen1.)
        if (dfb.advanced_options.allow_instance_multi_binding) {
            continue;
        }
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        auto check_uniform_mask = [&](const auto& records, std::string_view role) {
            if (records.size() < 2) {
                return;
            }
            const uint16_t first_mask = kernel_to_risc_mask.at(records[0].kernel);
            const auto* first_kernel = records[0].kernel;
            for (size_t i = 1; i < records.size(); ++i) {
                const uint16_t mask = kernel_to_risc_mask.at(records[i].kernel);
                if (mask == first_mask) {
                    continue;
                }
                if (is_gen2_arch(hal)) {
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
                } else {
                    TT_FATAL(
                        false,
                        "DFB '{}' has multiple {} KernelSpecs ('{}', '{}') with mismatched "
                        "processor placement (risc_mask 0x{:x} vs 0x{:x}). Multi-binding "
                        "requires all same-role kernels to share processor placement (for DM "
                        "kernels, check Gen1Config::processor; for compute, the "
                        "placement is determined by the KernelSpec's config type).",
                        dfb.unique_id,
                        role,
                        first_kernel->unique_id,
                        records[i].kernel->unique_id,
                        first_mask,
                        mask);
                }
            }
        };
        check_uniform_mask(endpoints.producers, "PRODUCER");
        check_uniform_mask(endpoints.consumers, "CONSUMER");
    }

    // Step 2c: Resolve TensorParameters against the MeshDevice into static CTA payloads.
    //
    // TensorBindings ride two existing kernel-arg channels:
    //   - Static layout (rank, shape, bank coords, ...) flows through the kernel's positional
    //     CTA buffer, after any user CTA-vararg prefix (KernelAdvancedOptions::compile_time_varargs).
    //   - Per-enqueue base address flows through a reserved-prefix named CRTA, appended to
    //     the kernel's user-named CRTAs and filled by SetProgramRunArgs from the
    //     corresponding TensorArgument entry. TensorParameters that opt into a dynamic accessor
    //     field (currently: dynamic_tensor_shape, sharded only) carry additional CRTA words
    //     immediately after the address slot, also filled at enqueue time.
    std::unordered_map<TensorParamName, ResolvedTensorParameter> resolved_tensor_parameters;
    resolved_tensor_parameters.reserve(spec.tensor_parameters.size());
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        resolved_tensor_parameters.emplace(
            tensor_parameter.unique_id, ResolveTensorParameterStaticCTAs(tensor_parameter, mesh_device));
    }

    // Step 3: Build the Program
    auto program_impl = std::make_shared<detail::ProgramImpl>(extract_context_id(&mesh_device));
    program_impl->mark_created_from_spec();  // mark as Metal 2.0 ProgramSpec-created (for legality checks)

    // Register TensorParameters with the program for ValidateProgramRunArgs to consult at enqueue.
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        program_impl->register_tensor_parameter(
            tensor_parameter.unique_id.get(), tensor_parameter.spec, tensor_parameter.relaxations);
    }

    // Create DataflowBuffers and build name -> ID map.
    // NOTE: Iterate over spec.dataflow_buffers (not collected.dfb_endpoints) to ensure
    //       deterministic DFB ID assignment based on user-specified order.
    DFBNameToIdMap dfb_name_to_id;
    DFBNameToSlotMap dfb_name_to_slot;
    std::unordered_map<DFBSpecName, bool> dfb_name_to_is_relay;
    for (const auto& dfb_spec : spec.dataflow_buffers) {
        const DFBSpecName& dfb_name = dfb_spec.unique_id;
        const auto& dfb_endpoint_info = collected.dfb_endpoints.at(dfb_name);
        const experimental::dfb::DataflowBufferConfig config =
            MakeDataflowBufferConfig(&dfb_spec, dfb_endpoint_info, kernel_to_risc_mask);

        // Add the DFB to the ProgramImpl, and register the name -> handle mapping.
        // Allocation nodes are derived from binding kernels' WorkUnitSpec membership.
        // (For borrowed-memory DFBs, config.borrows_memory was set in MakeDataflowBufferConfig;
        // the device-side runtime uses that to skip regular L1 allocation.)
        uint32_t dfb_id = program_impl->add_dataflow_buffer(collected.dfb_node_set.at(dfb_name), config);
        program_impl->register_dfb_spec_name(dfb_name.get(), dfb_id);
        dfb_name_to_id[dfb_name] = dfb_id;
        const auto& created_config = program_impl->get_dataflow_buffer(dfb_id)->config;
        dfb_name_to_slot[dfb_name] = program_impl->get_dataflow_buffer(dfb_id)->device_slot;
        dfb_name_to_is_relay[dfb_name] = created_config.is_relay;

        // Borrowed-memory DFB: record the dfb_id ↔ TensorParamName binding so that
        // SetProgramRunArgs / UpdateTensorArgs can resolve and attach the actual L1 Buffer
        // at runtime (analog of dynamic CB's UpdateDynamicCircularBufferAddress).
        if (dfb_spec.borrowed_from.has_value()) {
            program_impl->register_dfb_borrowed_binding(dfb_id, dfb_spec.borrowed_from->get());
        }
    }

    // Reserve PrefetcherPipe slots (one per kernel accessor group) from the spec geometry, register
    // relay DFBs against them, and record each parameter's placement for SetProgramRunArgs. Must
    // precede kernel creation: the slot id is baked into the kernel's `pipe::<accessor>` token and
    // a relay DFB's `dfb::` token.
    const PrefetcherPipeHandlesByKernel prefetcher_pipe_handles =
        ReservePrefetcherPipeSlots(mesh_device, spec, collected, *program_impl, dfb_name_to_id);

    std::unordered_map<DFBSpecName, uint8_t> dfb_name_to_prefetcher_pipe_id;
    for (const auto& [dfb_name, dfb_id] : dfb_name_to_id) {
        dfb_name_to_prefetcher_pipe_id[dfb_name] =
            program_impl->get_prefetcher_pipe_id_for_relay(dfb_id).value_or(0xFF);
    }

    // Wire alias groups: for each DFB that has alias_with entries, make the first
    // encountered DFB in the group the primary and call set_dfb_alias for each secondary.
    // handled_as_secondary prevents a DFB from being treated as a primary when it was
    // already registered as a secondary by an earlier DFB in the group. Soundness relies
    // on the strict-clique invariant enforced by ValidateProgramSpec: every group member
    // lists every other member, so the primary's alias_with covers the whole group.
    {
        std::unordered_set<DFBSpecName> handled_as_secondary;
        for (const auto& dfb_spec : spec.dataflow_buffers) {
            if (handled_as_secondary.contains(dfb_spec.unique_id)) {
                continue;
            }
            if (dfb_alias_with(dfb_spec).empty()) {
                continue;
            }
            const uint32_t primary_id = dfb_name_to_id.at(dfb_spec.unique_id);
            for (const auto& alias_name : dfb_alias_with(dfb_spec)) {
                if (handled_as_secondary.contains(alias_name)) {
                    continue;
                }
                const uint32_t secondary_id = dfb_name_to_id.at(alias_name);
                program_impl->set_dfb_alias(primary_id, secondary_id);
                handled_as_secondary.insert(alias_name);
            }
        }
    }

    // Create Semaphores and build the name -> ID map.
    // NOTE: Iterate over spec.semaphores to preserve user-provided deterministic ordering.
    SemaphoreNameToIdMap semaphore_name_to_id;
    for (const auto& semaphore_spec : spec.semaphores) {
        const SemaphoreSpecName& semaphore_name = semaphore_spec.unique_id;
        const uint32_t init_value = semaphore_spec.advanced_options.initial_value;
        uint32_t sem_id = program_impl->create_semaphore(
            to_node_range_set(semaphore_spec.target_nodes), init_value, CoreType::WORKER);
        program_impl->register_semaphore_spec_name(semaphore_name.get(), sem_id);
        semaphore_name_to_id[semaphore_name] = sem_id;
    }

    // Pick each semaphore's access mechanism. Resolve against this program's env (the mesh
    // device's), not the default context, so a non-default-context device resolves its own arch
    // and target device.
    const sem_solver::SemaphoreNameToScopeMap semaphore_name_to_scope =
        sem_solver::ResolveSemaphoreScopes(spec, collected.semaphore_binders, metal_env);

    // Create Kernels (arch-specific)
    for (const KernelSpec& kernel_spec : spec.kernels) {
        KernelSource kernel_src = MakeKernelSource(kernel_spec, program_impl->get_context_id());
        const NodeRangeSet& node_ranges = collected.kernel_node_set.at(kernel_spec.unique_id);

        // Make the local accessor name -> DFB device slot map for this kernel
        const tt::tt_metal::DataflowBufferBindingHandleMap dfb_handles = MakeDataflowBufferBindingHandles(
            kernel_spec, dfb_name_to_slot, dfb_name_to_is_relay, dfb_name_to_prefetcher_pipe_id, collected.dfb_by_name);
        const tt::tt_metal::SemaphoreBindingHandleMap semaphore_handles = MakeSemaphoreBindingHandles(
            kernel_spec, collected.semaphore_binders, semaphore_name_to_id, semaphore_name_to_scope);

        // Resolve TensorBindings for this kernel:
        //  - pack each binding's pre-resolved CTA payload into the kernel's positional CTA buffer
        //    (after the user CTA-vararg prefix)
        //  - assign each binding a slot in the kernel's CRTA buffer (TensorBinding address section)
        const auto& user_named_crtas = kernel_spec.runtime_arg_schema.common_runtime_arg_names;
        const auto& cta_varargs = kernel_spec.advanced_options.compile_time_varargs;
        const uint32_t vararg_cta_count = static_cast<uint32_t>(cta_varargs.size());
        TensorBindingsForKernel ta_bindings = ResolveTensorBindingsForKernel(
            kernel_spec,
            resolved_tensor_parameters,
            /*base_named_crta_count=*/user_named_crtas.size(),
            /*base_cta_offset=*/vararg_cta_count);

        // Create TensorBindingHandles for this kernel
        const std::vector<TensorBindingHandle>& tensor_binding_handles = ta_bindings.handles;

        // Resolve scratchpad bindings for this kernel. The scratchpad section follows the TensorBinding
        // section and precedes varargs, so it begins at the tensor-binding resolution's (pre-scratchpad)
        // vararg offset; we then push the vararg section out by the scratchpad section's width so the
        // crta_layout that flows into the kernel ctor reflects all four sections.
        ScratchpadBindingsForKernel sp_bindings = ResolveScratchpadBindingsForKernel(
            kernel_spec,
            collected.scratchpad_by_name,
            /*scratchpad_base_crta_word=*/ta_bindings.crta_layout.vararg_section_offset);
        ta_bindings.crta_layout.scratchpad_section_words = sp_bindings.section_words;
        ta_bindings.crta_layout.vararg_section_offset += sp_bindings.section_words;

        // Named-args schema fields passed to the Kernel ctor. The names are used at JIT time to
        // emit kernel_args_generated.h and factor into the kernel cache key. The TensorBinding
        // address section is tracked separately (via tensor_binding_handles), so we pass the user
        // CRTA list through unchanged.
        const auto& named_rtas = kernel_spec.runtime_arg_schema.runtime_arg_names;

        // Positional CTAs: [ user CTA varargs | TensorBinding CTA payloads ]
        std::vector<uint32_t> compile_args = cta_varargs;
        compile_args.insert(compile_args.end(), ta_bindings.cta_words.begin(), ta_bindings.cta_words.end());

        // Create the kernel object
        std::shared_ptr<Kernel> kernel;

        // Kernel creation APIs accept a "is_metal2_kernel" bool, which fences Metal 2.0 JIT machinery
        constexpr bool is_metal2_kernel = true;

        if (is_gen2_arch(hal)) {
            uint16_t risc_mask = kernel_to_risc_mask.at(&kernel_spec);
            if (kernel_spec.is_data_movement_kernel()) {
                auto config = MakeQuasarDataMovementConfig(kernel_spec);
                config.compile_args = std::move(compile_args);
                auto processors = GetDMProcessorSet(DMProcessorMask{(uint8_t)(risc_mask & 0xFF)});
                kernel = std::make_shared<experimental::quasar::QuasarDataMovementKernel>(
                    program_impl->get_context_id(),
                    kernel_src,
                    node_ranges,
                    config,
                    processors,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            } else {
                auto config = MakeGen2ComputeConfig(kernel_spec, dfb_name_to_slot, hal);
                config.compile_args = std::move(compile_args);
                auto processors = GetComputeProcessorSet(ComputeEngineMask{(uint8_t)(risc_mask >> 8)});
                kernel = std::make_shared<experimental::quasar::QuasarComputeKernel>(
                    program_impl->get_context_id(),
                    kernel_src,
                    node_ranges,
                    config,
                    processors,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            }
        } else {  // gen1
            if (kernel_spec.is_data_movement_kernel()) {
                auto config = MakeGen1DataMovementConfig(kernel_spec);
                config.compile_args = std::move(compile_args);
                kernel = std::make_shared<DataMovementKernel>(
                    program_impl->get_context_id(),
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            } else {
                auto config = MakeGen1ComputeConfig(kernel_spec, dfb_name_to_slot, hal);
                config.compile_args = std::move(compile_args);
                // Bake the compute semaphore's capacity into the kernel (Semaphore::wait_not_full and the
                // SEMINITs read COMPUTE_SEMAPHORE_MAX). At most one compute semaphore per program
                // (ValidateProgramSpec), so at most one define.
                for (const auto& binding : kernel_spec.semaphore_bindings) {
                    if (semaphore_name_to_scope.at(binding.semaphore_spec_name) != SemScope::COMPUTE_ATOMIC) {
                        continue;
                    }
                    const auto sem =
                        std::find_if(spec.semaphores.begin(), spec.semaphores.end(), [&](const SemaphoreSpec& s) {
                            return s.unique_id == binding.semaphore_spec_name;
                        });
                    // The host option is the source of truth: always bake the resolved capacity, overriding
                    // any user-supplied COMPUTE_SEMAPHORE_MAX define. max_value 0 means the default (the 4-bit
                    // hardware ceiling, 15), which is what the kernel assumes when the define is absent.
                    const uint32_t capacity = (sem != spec.semaphores.end() && sem->advanced_options.max_value != 0)
                                                  ? sem->advanced_options.max_value
                                                  : 15u;
                    config.defines["COMPUTE_SEMAPHORE_MAX"] = std::to_string(capacity);
                }
                kernel = std::make_shared<ComputeKernel>(
                    program_impl->get_context_id(),
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            }
        }

        // Attach the resolved scratchpad bindings to the kernel (set post-construction: their sizes are
        // part of the kernel cache key, so this must run before the kernel is compiled). allocate_scratchpads
        // will later fill each handle's allocated_address.
        kernel->set_scratchpad_binding_handles(std::move(sp_bindings.handles));

        // PrefetcherPipe accessors -> program slot ids (also part of the kernel cache key).
        if (auto pipe_it = prefetcher_pipe_handles.find(&kernel_spec); pipe_it != prefetcher_pipe_handles.end()) {
            kernel->set_prefetcher_pipe_binding_handles(pipe_it->second);
        }

        std::vector<TensorBindingSequenceHandle> tensor_binding_sequences;
        tensor_binding_sequences.reserve(kernel_spec.advanced_options.tensor_binding_sequences.size());
        for (const auto& sequence : kernel_spec.advanced_options.tensor_binding_sequences) {
            tensor_binding_sequences.push_back(
                TensorBindingSequenceHandle{.sequence_name = sequence.sequence_name, .members = sequence.members});
        }
        kernel->set_tensor_binding_sequences(std::move(tensor_binding_sequences));

        // Prefix length for device get_compile_time_vararg* bounds (values are in compile_time_args_).
        kernel->set_compile_time_vararg_count(vararg_cta_count);

        // Add the kernel to the ProgramImpl and register the name -> handle mapping
        KernelHandle handle = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
        program_impl->register_kernel_spec_name(kernel_spec.unique_id.get(), handle);

        // Register the RTA+CRTA schema (named lists + vararg counts) with the ProgramImpl.
        // Used by ValidateProgramRunArgs and SetProgramRunArgs to validate and serialize
        // the user-provided values at dispatch time.
        //
        // User-facing vararg RTA specification (see kernel_spec.hpp):
        //   - num_runtime_varargs (scalar): default count applied to every node the kernel
        //     runs on.
        //   - num_runtime_varargs_per_node (optional): sparse per-node overrides on top of
        //     the scalar default. Unlisted nodes fall back to the scalar.
        // We apply the scalar first across target_nodes, then overlay each override entry.
        // An explicit override of 0 erases the scalar-default entry so run-params treats
        // that node as having no varargs (rather than requiring an "empty" value list).
        // Overlapping override entries (two entries covering the same node) are an error.
        const auto& user_schema = kernel_spec.runtime_arg_schema;
        detail::ProgramImpl::KernelRTASchema runtime_schema;
        runtime_schema.runtime_arg_names = user_schema.runtime_arg_names;

        // Pass the user CRTA list through.
        // NOTE: The TensorBinding address section is tracked separately on the Kernel
        // (via tensor_binding_handles) and its slot offsets are baked into each binding handle's
        // addr_crta_offset; SetProgramRunArgs uses BOTH to assemble the per-enqueue CRTA buffer.
        runtime_schema.common_runtime_arg_names = user_named_crtas;

        // Precompute the name -> slot-index maps so the hot UpdateProgramRunArgs path does O(1)
        // lookups rather than rebuilding a map per call.
        for (size_t i = 0; i < runtime_schema.runtime_arg_names.size(); ++i) {
            runtime_schema.runtime_arg_name_to_slot.emplace(runtime_schema.runtime_arg_names[i], i);
        }
        for (size_t i = 0; i < runtime_schema.common_runtime_arg_names.size(); ++i) {
            runtime_schema.common_runtime_arg_name_to_slot.emplace(runtime_schema.common_runtime_arg_names[i], i);
        }

        // Varargs schema now lives on KernelAdvancedOptions.
        const uint32_t num_runtime_varargs = kernel_spec.advanced_options.num_runtime_varargs;
        const uint32_t num_common_runtime_varargs = kernel_spec.advanced_options.num_common_runtime_varargs;
        const bool has_per_node_override = !kernel_spec.advanced_options.num_runtime_varargs_per_node.empty();

        if (num_runtime_varargs > 0) {
            for (const NodeRange& range : node_ranges.ranges()) {
                for (const NodeCoord& node : range) {
                    runtime_schema.num_runtime_varargs_per_node[node] = num_runtime_varargs;
                }
            }
        }
        if (has_per_node_override) {
            std::unordered_set<NodeCoord> seen_overrides;
            for (const auto& [nodes_spec, num_varargs] : kernel_spec.advanced_options.num_runtime_varargs_per_node) {
                const NodeRangeSet expanded = to_node_range_set(nodes_spec);
                for (const NodeRange& range : expanded.ranges()) {
                    for (const NodeCoord& node : range) {
                        const bool inserted = seen_overrides.insert(node).second;
                        TT_FATAL(
                            inserted,
                            "KernelSpec '{}' num_runtime_varargs_per_node has overlapping entries "
                            "for node {}",
                            kernel_spec.unique_id,
                            node.str());
                        if (num_varargs > 0) {
                            runtime_schema.num_runtime_varargs_per_node[node] = num_varargs;
                        } else {
                            // Explicit zero override: drop any scalar-default entry so
                            // run-params treats this node as missing (→ 0 expected).
                            runtime_schema.num_runtime_varargs_per_node.erase(node);
                        }
                    }
                }
            }
        }
        runtime_schema.num_common_runtime_varargs = num_common_runtime_varargs;
        program_impl->register_kernel_rta_schema(kernel_spec.unique_id.get(), runtime_schema);
    }

    return Program(std::move(program_impl));
}

}  // namespace tt::tt_metal::experimental
