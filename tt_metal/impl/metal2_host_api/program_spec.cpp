// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <functional>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

// TODO: These constants should be queriable from the public API (currently HAL, for consistency)
//       They are currently also hardcoded in the temporary Quasar host_api.hpp. Need to clean this up.
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_TENSIX_CORES_PER_NODE = 4;

inline bool is_gen2_arch() {
    tt::ARCH arch = tt::tt_metal::hal::get_arch();
    return arch == tt::ARCH::QUASAR;
}
inline bool is_gen1_arch() {
    tt::ARCH arch = tt::tt_metal::hal::get_arch();
    return arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE;
}

// TODO: Are these NodeRangeSet helpers really needed?
//       I'm sure a convert-to-CoreRangeSet helper already exists.
//       Everything is converted to CoreRangeSet upfront in the iterative API.

// Helper: Convert Nodes variant to NodeRangeSet
NodeRangeSet to_node_range_set(const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes) {
    return std::visit(
        [](const auto& n) -> NodeRangeSet {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, NodeRangeSet>) {
                return n;
            } else if constexpr (std::is_same_v<T, NodeRange>) {
                return NodeRangeSet(n);
            } else {
                // NodeCoord case
                return NodeRangeSet(NodeRange(n, n));
            }
        },
        nodes);
}

// Helper: Check if two Nodes variants intersect
bool nodes_intersect(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& a,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& b) {
    NodeRangeSet a_set = to_node_range_set(a);
    NodeRangeSet b_set = to_node_range_set(b);
    return a_set.intersects(b_set);
}

// Helper: Check if one Nodes variant contains another
bool nodes_contains(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& superset,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& subset) {
    NodeRangeSet superset_node_range_set = to_node_range_set(superset);
    NodeRangeSet subset_node_range_set = to_node_range_set(subset);
    return superset_node_range_set.contains(subset_node_range_set);
}

// Data structure built up from ProgramSpec to enable fast lookups
struct CollectedSpecData {
    // Name -> spec lookups
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;

    // DFB endpoint info lives on the kernel spec
    struct DFBEndpointInfo {
        const KernelSpec* producer = nullptr;
        const KernelSpec* consumer = nullptr;
        const KernelSpec::DFBBinding* producer_binding = nullptr;
        const KernelSpec::DFBBinding* consumer_binding = nullptr;
    };
    std::unordered_map<DFBSpecName, DFBEndpointInfo> dfb_endpoints;
};

// Perform ALL validation checks required on the ProgramSpec.
// If the call succeeds, the ProgramSpec is guaranteed to be valid.
// Return a CollectedSpecData structure (derived from the ProgramSpec) for re-use during program creation.
CollectedSpecData ValidateAndCollectSpecData(const ProgramSpec& spec) {
    CollectedSpecData collected_spec_data;

    // Check target architecture (temporary)
    TT_FATAL(
        tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR,
        "Metal 2.0 API is currently only implemented for Quasar. WH/BH support coming soon.");

    //////////////////////////////
    // Validate KernelSpecs
    //////////////////////////////

    // A Program needs at least one kernel
    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");

    // All KernelSpecs must have unique names
    for (const auto& kernel : spec.kernels) {
        // Populate kernel_by_name lookup while validating
        auto [it, inserted] = collected_spec_data.kernel_by_name.try_emplace(kernel.unique_id, &kernel);
        TT_FATAL(inserted, "Duplicate name '{}' found KernelSpec list", kernel.unique_id);
    }

    // Validate kernel thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (kernel.is_compute_kernel()) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_TENSIX_CORES_PER_NODE,
                "KernelSpec '{}' has too many threads. The architecture supports up to {} for compute kernels.",
                kernel.unique_id,
                QUASAR_TENSIX_CORES_PER_NODE);
        }
        if (kernel.is_dm_kernel()) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_DM_CORES_PER_NODE,
                "KernelSpec '{}' has too many data movement threads. The architecture supports up to {} for data "
                "movement kernels.",
                kernel.unique_id,
                QUASAR_DM_CORES_PER_NODE);
        }
    }

    // Check DM configs
    for (const auto& kernel : spec.kernels) {
        if (kernel.is_dm_kernel()) {
            const auto& data_movement_config = std::get<DataMovementConfiguration>(kernel.config_spec);

            // Both Gen1 and Gen2 configs are optional. But at least one must be specified.
            TT_FATAL(
                data_movement_config.gen1_data_movement_config.has_value() ||
                    data_movement_config.gen2_data_movement_config.has_value(),
                "KernelSpec '{}' must specify a DM config for Gen1, Gen2, or both.",
                kernel.unique_id);

            // The config for the current target architecture must be specified.
            if (is_gen2_arch()) {
                TT_FATAL(
                    data_movement_config.gen2_data_movement_config.has_value(),
                    "KernelSpec '{}' must specify a Gen2 DM config when targeting Quasar.",
                    kernel.unique_id);
            } else if (is_gen1_arch()) {
                TT_FATAL(
                    data_movement_config.gen1_data_movement_config.has_value(),
                    "KernelSpec '{}' must specify a Gen1 DM config when targeting WH or BH.",
                    kernel.unique_id);
            } else {
                TT_FATAL(false, "Unknown architecture");
            }
        }
    }

    //////////////////////////////////
    // DataflowBufferSpec validation
    //////////////////////////////////

    // All DataflowBufferSpecs must have unique names
    for (const auto& dfb : spec.dataflow_buffers) {
        // Populate dfb_by_name lookup while validating
        auto [it, inserted] = collected_spec_data.dfb_by_name.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate name '{}' found in dataflow_buffers", dfb.unique_id);
    }

    // A DFB must have exactly one producer and one consumer
    for (const auto& kernel : spec.kernels) {
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            // Get the DFBEndpointInfo for this DFB (if it doesn't exist, create it)
            CollectedSpecData::DFBEndpointInfo& endpoint_info =
                collected_spec_data.dfb_endpoints[dfb_binding.dfb_spec_name];

            if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::PRODUCER) {
                TT_FATAL(
                    endpoint_info.producer == nullptr, "DFB '{}' has multiple producers", dfb_binding.dfb_spec_name);
                endpoint_info.producer = &kernel;
                endpoint_info.producer_binding = &dfb_binding;
            } else if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::CONSUMER) {
                TT_FATAL(
                    endpoint_info.consumer == nullptr, "DFB '{}' has multiple consumers", dfb_binding.dfb_spec_name);
                endpoint_info.consumer = &kernel;
                endpoint_info.consumer_binding = &dfb_binding;
            } else {
                TT_FATAL(false, "RELAY endpoints are only used for remote DFB, which is not supported yet");
            }
        }
    }
    for (const auto& [dfb_name, endpoint_info] : collected_spec_data.dfb_endpoints) {
        TT_FATAL(endpoint_info.producer != nullptr, "DFB '{}' has no producer", dfb_name);
        TT_FATAL(endpoint_info.consumer != nullptr, "DFB '{}' has no consumer", dfb_name);
    }

    // Check for unbound DFBs
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected_spec_data.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }

    // Remote DFBs are not supported yet
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(!dfb.is_remote_dfb, "Remote DFBs are not supported yet");
        TT_FATAL(!dfb.producer_consumer_map || dfb.producer_consumer_map->empty(), "Remote DFBs are not supported yet");
    }

    // Data format metadata (optional param) MUST be specified a DFB with a compute endpoint
    for (const auto& [dfb_name, endpoint_info] : collected_spec_data.dfb_endpoints) {
        // Does it have a compute kernel endpoint?
        if ((endpoint_info.producer && endpoint_info.producer->is_compute_kernel()) ||
            (endpoint_info.consumer && endpoint_info.consumer->is_compute_kernel())) {
            // Data format metadata is required
            const DataflowBufferSpec* dfb_spec = collected_spec_data.dfb_by_name.at(dfb_name);
            TT_FATAL(
                dfb_spec->data_format_metadata.has_value(),
                "DFB '{}' is used by a compute kernel, but no data_format_metadata is specified",
                dfb_name);
        }
    }

    //////////////////////////////////
    // SemaphoreSpec validation
    //////////////////////////////////

    // Semaphores aren't supported yet for Quasar
    TT_FATAL(spec.semaphores.empty(), "Semaphores are not supported yet");

    // All SemaphoreSpecs must have unique names
    for (const auto& semaphore : spec.semaphores) {
        // Populate semaphore_by_name lookup while validating
        auto [it, inserted] = collected_spec_data.semaphore_by_name.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate name '{}' found in semaphores", semaphore.unique_id);
    }

    //////////////////////////////
    // WorkerSpec validation
    //////////////////////////////

    // Check that WorkerSpecs are provided on Quasar
    // NOTE: WorkerSpec data is redundant, but improves clarity and messaging.
    //       If it's hated, we can make it optional on Gen2 and derive the WorkerSpec if absent.
    TT_FATAL(spec.workers.has_value(), "Workers are required on Gen2+");
    const auto& workers = spec.workers.value();
    TT_FATAL(!workers.empty(), "At least one WorkerSpec is required");

    // WorkerSpecs don't really need unique names, as the names are only used for error messaging.
    // No need to validate uniqueness.

    // WorkerSpecs may not overlap in their target nodes
    for (const auto& worker : workers) {
        for (const auto& other_worker : workers) {
            if (worker.unique_id == other_worker.unique_id) {
                continue;
            }
            if (nodes_intersect(worker.target_nodes, other_worker.target_nodes)) {
                TT_FATAL(
                    false,
                    "WorkerSpecs '{}' and '{}' overlap in target nodes",
                    worker.unique_id,
                    other_worker.unique_id);
            }
        }
    }

    // A WorkerSpec must have at least one kernel
    for (const auto& worker : workers) {
        TT_FATAL(!worker.kernels.empty(), "WorkerSpec '{}' has no kernels", worker.unique_id);
    }

    // Does the Worker have enough cores to run all its kernels?
    for (const auto& worker : workers) {
        uint32_t dm_cores_needed = 0;
        uint32_t compute_cores_needed = 0;
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected_spec_data.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                compute_cores_needed += kernel_spec->num_threads;
            }
            if (kernel_spec->is_dm_kernel()) {
                dm_cores_needed += kernel_spec->num_threads;
            }
        }
        TT_FATAL(
            compute_cores_needed <= QUASAR_TENSIX_CORES_PER_NODE,
            "WorkerSpec '{}' needs {} compute cores, but only {} are available",
            worker.unique_id,
            compute_cores_needed,
            QUASAR_TENSIX_CORES_PER_NODE);
        TT_FATAL(
            dm_cores_needed <= QUASAR_DM_CORES_PER_NODE,
            "WorkerSpec '{}' needs {} data movement cores, but only {} are available",
            worker.unique_id,
            dm_cores_needed,
            QUASAR_DM_CORES_PER_NODE);
    }

    // A worker can have at most one compute kernel
    for (const auto& worker : workers) {
        uint32_t num_compute_kernels = 0;
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected_spec_data.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                num_compute_kernels++;
            }
        }
        TT_FATAL(num_compute_kernels <= 1, "WorkerSpec '{}' has more than one compute kernel", worker.unique_id);
    }

    // Check that WorkerSpec target nodes are valid nodes
    // (TODO once legality rules on Quasar are defined)
    // (For WH/BH, this check should catch attempts to assign kernels to dispatch nodes.)

    // Kernels in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected_spec_data.kernel_by_name.at(kernel_name);
            TT_FATAL(
                nodes_contains(kernel_spec->target_nodes, worker.target_nodes),
                "Kernel '{}' target nodes must contain WorkerSpec '{}' target nodes",
                kernel_name,
                worker.unique_id);
        }
    }

    // DFBs in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& dfb_name : worker.dataflow_buffers) {
            const auto& dfb_spec = collected_spec_data.dfb_by_name.at(dfb_name);
            TT_FATAL(
                nodes_contains(dfb_spec->target_nodes, worker.target_nodes),
                "DFB '{}' target nodes must contain WorkerSpec '{}' target nodes",
                dfb_name,
                worker.unique_id);
        }
    }

    // Semaphores in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& semaphore_name : worker.semaphores) {
            const auto& semaphore_spec = collected_spec_data.semaphore_by_name.at(semaphore_name);
            TT_FATAL(
                nodes_contains(semaphore_spec->target_nodes, worker.target_nodes),
                "Semaphore '{}' target nodes must contain WorkerSpec '{}' target nodes",
                semaphore_name,
                worker.unique_id);
        }
    }

    // Check that all kernel target nodes are "accounted for" by a WorkerSpec
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_ranges;
    for (const auto& worker : workers) {
        for (const auto& kernel_name : worker.kernels) {
            // A kernel may belong to multiple WorkerSpecs.
            // Merge the target nodes of all WorkerSpecs that contain this kernel.
            kernel_node_ranges[kernel_name].merge(to_node_range_set(worker.target_nodes));
        }
    }
    for (const auto& kernel : spec.kernels) {
        KernelSpecName kernel_name = kernel.unique_id;
        NodeRangeSet kernel_target_nodes = to_node_range_set(kernel.target_nodes);

        // The kernel must belong to at least one WorkerSpec
        TT_FATAL(kernel_node_ranges.contains(kernel_name), "Kernel '{}' is not part of any WorkerSpec", kernel_name);

        // All the kernel's target nodes must be accounted for by the WorkerSpecs that contain it
        NodeRangeSet worker_derived_target_nodes = kernel_node_ranges.at(kernel_name);
        TT_FATAL(
            worker_derived_target_nodes == kernel_target_nodes,
            "Kernel '{}' has target nodes that are not accounted for by any WorkerSpec",
            kernel_name);
    }

    // Check that all DFB target nodes are "accounted for" by a WorkerSpec
    // (May revisit this, depending on how remote DFBs are implemented)
    std::unordered_map<DFBSpecName, NodeRangeSet> dfb_node_ranges;
    for (const auto& worker : workers) {
        for (const auto& dfb_name : worker.dataflow_buffers) {
            // A DFB may belong to multiple WorkerSpecs.
            // Merge the target nodes of all WorkerSpecs that have this DFB.
            dfb_node_ranges[dfb_name].merge(to_node_range_set(worker.target_nodes));
        }
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        DFBSpecName dfb_name = dfb.unique_id;
        NodeRangeSet dfb_target_nodes = to_node_range_set(dfb.target_nodes);

        // The DFB must belong to at least one WorkerSpec
        TT_FATAL(dfb_node_ranges.contains(dfb_name), "DFB '{}' is not part of any WorkerSpec", dfb_name);

        // All the DFB's target nodes must be accounted for by the WorkerSpecs that have it
        NodeRangeSet worker_derived_target_nodes = dfb_node_ranges.at(dfb_name);
        TT_FATAL(
            worker_derived_target_nodes == dfb_target_nodes,
            "DFB '{}' has target nodes that are not accounted for by any WorkerSpec",
            dfb_name);
    }

    return collected_spec_data;
}

// Handy struct used for solving kernel->core assignments
template <uint8_t NUM_CORES>
struct ProcessorMask {
    static_assert(NUM_CORES > 0 && NUM_CORES <= 8, "ProcessorMask supports 1-8 processors");

    // Mask of valid bit positions (e.g., 0xFF for 8 processors, 0x0F for 4)
    static constexpr uint8_t VALID_BITS_MASK = (NUM_CORES == 8) ? 0xFF : ((1 << NUM_CORES) - 1);

    // One-hot encoding of processors in use (0 = available, 1 = in use)
    uint8_t bits = 0x00;

    // Boolean operators
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

    // Helpers
    uint8_t num_in_use() const { return std::popcount(bits); }
    uint8_t num_available() const { return NUM_CORES - num_in_use(); }
    bool is_idx_available(uint8_t idx) const { return (bits & (1 << idx)) == 0; }
    bool is_idx_in_use(uint8_t idx) const { return (bits & (1 << idx)) != 0; }
    bool conflicts_with(ProcessorMask other) const { return (bits & other.bits) != 0; }

    // Factory method: create a ProcessorMask from a mask
    static ProcessorMask create(uint8_t mask) {
        TT_FATAL(mask <= VALID_BITS_MASK, "Mask specifies too many cores for ProcessorMask<{}>: {}", NUM_CORES, mask);
        return {mask};
    }

    // Factory method: allocate N available processors from those not already in use
    // Returns a mask of the newly reserved processors, or std::nullopt if not enough are available
    static std::optional<ProcessorMask> reserve_n(uint8_t n, const ProcessorMask& already_in_use) {
        // Enough left?
        if (already_in_use.num_available() < n) {
            return std::nullopt;
        }

        ProcessorMask newly_reserved;
        for (uint8_t i = 0; i < NUM_CORES && n > 0; i++) {
            if (already_in_use.is_idx_available(i)) {
                newly_reserved.bits |= (1 << i);
                n--;
            }
        }
        return newly_reserved;
    }
};

// Type aliases for DM and compute processors on Quasar
using DMProcessorMask = ProcessorMask<QUASAR_DM_CORES_PER_NODE>;
using ComputeProcessorMask = ProcessorMask<QUASAR_TENSIX_CORES_PER_NODE>;

// Helper: Reserve DM processors for a kernel on a WorkerSpec (aka KernelGroup)
// Returns {this_kernel_mask, updated_cumulative_mask}
// Throws TT_FATAL on conflict or allocation failure (see simplifying assumption notes)
std::pair<DMProcessorMask, DMProcessorMask> ReserveDMProcessors(
    const KernelSpec* kernel_spec,
    std::optional<DMProcessorMask> existing_mask,
    DMProcessorMask cumulative_mask,
    const WorkerSpecName& worker_id) {
    // Was this kernel already assigned a mask from a previous WorkerSpec?
    if (existing_mask.has_value()) {
        DMProcessorMask existing = existing_mask.value();

        // Check for conflict with what's already allocated on the current WorkerSpec
        TT_FATAL(
            !existing.conflicts_with(cumulative_mask),
            "Kernel '{}' requires processors already in use on WorkerSpec '{}'. "
            "The \"common DM cores\" assumption has been violated!",
            kernel_spec->unique_id,
            worker_id);

        // Return existing mask and updated cumulative
        return {existing, cumulative_mask | existing};
    }

    // First time seeing this kernel - reserve new processors
    std::optional<DMProcessorMask> reserved = DMProcessorMask::reserve_n(kernel_spec->num_threads, cumulative_mask);
    TT_FATAL(
        reserved.has_value(),
        "Failed to reserve processors for DM kernel '{}' on WorkerSpec '{}'. "
        "The \"common DM cores\" assumption has been violated!",
        kernel_spec->unique_id,
        worker_id);

    DMProcessorMask mask = reserved.value();
    return {mask, cumulative_mask | mask};
}

// Helper: Assign compute processor mask for a kernel.
ComputeProcessorMask AssignComputeProcessors(const KernelSpec* kernel_spec, const KernelSpecName& kernel_name) {
    auto reserved = ComputeProcessorMask::reserve_n(kernel_spec->num_threads, ComputeProcessorMask::create(0x00));
    TT_FATAL(
        reserved.has_value(),
        "Compute kernel '{}' reservation failed. Condition should be unreachable after validation.",
        kernel_name);
    return reserved.value();
}

using DMProcessorMaskMap = std::unordered_map<const KernelSpec*, DMProcessorMask>;
using ComputeProcessorMaskMap = std::unordered_map<const KernelSpec*, ComputeProcessorMask>;

// Helper: Solve kernel-to-core assignments
// NOTE: Despite the earlier legality checks, it is possible for the solver to fail! (with TT_FATAL)
//   The current implementation makes a (likely temporary) simplifying assumption:
//      A given DM kernel will run on the _same_ set of DM cores on every node/cluster.
//   If the input ProgramSpec passes legality checks but fails in the solver, the resulting error
//   message will make it clear what went wrong (i.e. overly restrictive "common DM cores" assumption).
std::pair<DMProcessorMaskMap, ComputeProcessorMaskMap> SolveKernelToProcessorAssignments(
    const ProgramSpec& spec, const CollectedSpecData& derived_data) {
    // Mapping from kernel specs to their processor masks
    DMProcessorMaskMap kernels_to_dm_processor_mask;
    ComputeProcessorMaskMap kernels_to_compute_processor_mask;

    // Solver loop (with simplifying assumption)
    for (const WorkerSpec& worker : spec.workers.value()) {
        // Cumulative DM processor mask for this WorkerSpec
        // (If we decide to reserve DM cores for interal use, just update the initial mask here.)
        DMProcessorMask cumulative_dm_mask = DMProcessorMask::create(0x00);

        // Since we enforce (at most) one compute kernel per WorkerSpec, no need to track cumulative mask.

        for (const KernelSpecName& kernel_name : worker.kernels) {
            const KernelSpec* kernel_spec = derived_data.kernel_by_name.at(kernel_name);

            if (kernel_spec->is_dm_kernel()) {
                // Look up existing DM mask, if any (from previous WorkerSpec iterations)
                std::optional<DMProcessorMask> existing_mask =
                    kernels_to_dm_processor_mask.contains(kernel_spec)
                        ? std::optional(kernels_to_dm_processor_mask.at(kernel_spec))
                        : std::nullopt;  // (redundant lookup, for code readability)

                auto [mask, new_cumulative] =
                    ReserveDMProcessors(kernel_spec, existing_mask, cumulative_dm_mask, worker.unique_id);

                kernels_to_dm_processor_mask[kernel_spec] = mask;
                cumulative_dm_mask = new_cumulative;
            } else {
                // compute kernel
                kernels_to_compute_processor_mask[kernel_spec] = AssignComputeProcessors(kernel_spec, kernel_name);
            }
        }
    }

    return std::make_pair(kernels_to_dm_processor_mask, kernels_to_compute_processor_mask);
}

// Helper: Make a DataflowBufferConfig from a DataflowBufferSpec
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const DMProcessorMaskMap& kernel_to_dm_processor_mask_map,
    const ComputeProcessorMaskMap& kernel_to_compute_processor_mask_map) {
    const KernelSpec* producer = dfb_endpoint_info.producer;
    const KernelSpec* consumer = dfb_endpoint_info.consumer;

    // Create the combined processor mask for producer and consumer
    // (uint16_t, where bits 0-7 = DM riscs, bits 8-15 = Tensix riscs)
    auto get_dfb_risc_mask = [&](const KernelSpec* kernel) -> uint16_t {
        if (kernel->is_dm_kernel()) {
            return kernel_to_dm_processor_mask_map.at(kernel).bits;
        } else {
            return kernel_to_compute_processor_mask_map.at(kernel).bits << 8;
        }
    };
    uint16_t producer_risc_mask = get_dfb_risc_mask(producer);
    uint16_t consumer_risc_mask = get_dfb_risc_mask(consumer);

    // Convert user-facing access pattern enum to hardware interface access pattern enum
    // (TODO: We should merge these enums; it's silly to have separate ones.)
    auto to_hw_access_pattern = [](DFBAccessPattern pattern) -> experimental::dfb::AccessPattern {
        switch (pattern) {
            case DFBAccessPattern::STRIDED: return experimental::dfb::AccessPattern::STRIDED;
            case DFBAccessPattern::BLOCKED: return experimental::dfb::AccessPattern::BLOCKED;
            case DFBAccessPattern::CONTIGUOUS: TT_FATAL(false, "CONTIGUOUS access pattern is not yet supported");
        }
        TT_FATAL(false, "Unknown DFBAccessPattern");
    };
    auto producer_access_pattern = to_hw_access_pattern(dfb_endpoint_info.producer_binding->access_pattern);
    auto consumer_access_pattern = to_hw_access_pattern(dfb_endpoint_info.consumer_binding->access_pattern);

    return experimental::dfb::DataflowBufferConfig{
        .entry_size = dfb_spec->entry_size,
        .num_entries = dfb_spec->num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = dfb_endpoint_info.producer->num_threads,
        .pap = producer_access_pattern,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = dfb_endpoint_info.consumer->num_threads,
        .cap = consumer_access_pattern,
        .enable_implicit_sync = !dfb_spec->disable_implicit_sync,
        .data_format = dfb_spec->data_format_metadata.value_or(tt::DataFormat::Invalid),
        .tile = dfb_spec->tile_format_metadata};
}

Program MakeProgramFromSpec(const ProgramSpec& spec) {
    // Validate and collect derived data
    // (We could add a flag to skip validation at production runtime, to reduce overhead.)
    CollectedSpecData derived_data = ValidateAndCollectSpecData(spec);

    // Solve kernel-to-core assignments
    // NOTE: Current solver assumes that a given DM kernel uses the _same_ set of DM cores on every node/cluster.
    auto [kernel_to_dm_processor_mask_map, kernel_to_compute_processor_mask_map] =
        SolveKernelToProcessorAssignments(spec, derived_data);

    // Create ProgramImpl
    auto program_impl = std::make_shared<detail::ProgramImpl>();

    // Create DataflowBuffers
    std::unordered_map<DFBSpecName, uint32_t> dfb_name_to_id_map;
    for (const auto& [dfb_name, dfb_endpoint_info] : derived_data.dfb_endpoints) {
        // Create the DFB + add it to the ProgramImpl
        const DataflowBufferSpec* dfb_spec = derived_data.dfb_by_name.at(dfb_name);
        const experimental::dfb::DataflowBufferConfig config = MakeDataflowBufferConfig(
            dfb_spec, dfb_endpoint_info, kernel_to_dm_processor_mask_map, kernel_to_compute_processor_mask_map);
        uint32_t dfb_id = program_impl->add_dataflow_buffer(to_node_range_set(dfb_spec->target_nodes), config);

        // Remember the generated ID
        dfb_name_to_id_map[dfb_name] = dfb_id;
    }

    // Create Kernels
    // (TODO... WIP)
    /*

    // Ok, what will we need?

    //  - [DONE] We need a CreateKernel API that will accept our pre-solved processor list
    //     - We'll need some silly helpers to convert from our mask to the official enum
    //  - We need a way to get our consts into the headergen (new CTA mechanism)
    //    ... and later, our location-specific consts into the RTA list...
    //    ... tempted to do this via the experimental Quasar config

    for (const KernelSpec& kernel_spec : spec.kernels) {

        if (kernel_spec.is_dm_kernel()) {

            // Data movement kernel
            KernelSource kernel_source = MakeKernelSource(kernel_spec);
            QuasarDataMovementConfig config = MakeQuasarDataMovementConfig(kernel_spec);
            std::set<DataMovementProcessor> dm_processors = MakeDMProcessorSet(DMProcessorMask);
            // Get DFB local accessor names -> DFB ID
            // (crap, this means walking the dfb bindings vector again...)

            // Create the kernel
            std::shared_ptr<Kernel> kernel = std::make_shared<QuasarDataMovementKernel>(kernel_src, core_ranges, config,
    dm_processors);

            // Add it to the ProgramImpl & save the kernel handle
            KernelHandle kh = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
            // TODO... save kernel handle
        }
        else {
            // Compute kernel
            KernelSource kernel_source = MakeKernelSource(kernel_spec);
            QuasarDataMovementConfig config = MakeQuasarComputeConfig(kernel_spec);
            std::set<DataMovementProcessor> dm_processors = MakeComputeProcessorSet(ComputeProcessorMask);
            // Get DFB local accessor names -> DFB ID

            // Create the kernel
            std::shared_ptr<Kernel> kernel = std::make_shared<QuasarComputeKernel>(kernel_src, core_ranges, config,
    compute_processors);

            // Add it to the ProgramImpl & save the kernel handle
            KernelHandle kh = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
            // TODO... save kernel handle
        }
    }

    // Shoot.
    // In order to implement ProgramRunParams, I'm going to need to store the
    // KernelSpecName and DFBSpecName in the ProgramImpl.
    // I need these to hold the mapping data from spec names to the handles.
    // Luckily, ProgramImpl is not part of the public API, so I can add members to it.

    */

    return Program(std::move(program_impl));
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
