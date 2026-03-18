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
// They are currently hardcoded in the temporary Quasar host_api.hpp too.
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_TENSIX_CORES_PER_NODE = 4;

// TODO: See if these NodeRangeSet helpers are needed. They may already exist (for CoreRangeSet).

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

// Helper: All ProgramSpec validation checks
void ValidateProgramSpec(const ProgramSpec& spec) {
    // Check target architecture
    TT_FATAL(
        tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR,
        "Metal 2.0 API is currently only implemented for Quasar. WH/BH support coming soon.");

    // Data structures
    std::unordered_map<KernelSpecName, const KernelSpec*> kernels;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfbs;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphores;

    //////////////////////////////
    // KernelSpec validation
    //////////////////////////////

    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");

    // All KernelSpecs must have unique names
    for (const auto& kernel : spec.kernels) {
        kernels[kernel.unique_id] = &kernel;
        auto [it, inserted] = kernels.try_emplace(kernel.unique_id, &kernel);
        TT_FATAL(inserted, "Duplicate name '{}' found in data_movement_kernels", kernel.unique_id);
    }

    // Validate kernel thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (std::holds_alternative<ComputeConfiguration>(kernel.config_spec)) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_TENSIX_CORES_PER_NODE,
                "KernelSpec '{}' has too many threads!",
                kernel.unique_id);
        }
        if (std::holds_alternative<DataMovementConfiguration>(kernel.config_spec)) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_DM_CORES_PER_NODE,
                "KernelSpec '{}' has too many threads!",
                kernel.unique_id);
        }
    }

    // Check DM config specs
    for (const auto& kernel : spec.kernels) {
        if (std::holds_alternative<DataMovementConfiguration>(kernel.config_spec)) {
            const auto& data_movement_config = std::get<DataMovementConfiguration>(kernel.config_spec);
            TT_FATAL(
                data_movement_config.gen1_data_movement_config.has_value() ||
                    data_movement_config.gen2_data_movement_config.has_value(),
                "KernelSpec '{}' must specify a DM config for Gen1, Gen2, or both.",
                kernel.unique_id);
        }
    }

    //////////////////////////////////
    // DataflowBufferSpec validation
    //////////////////////////////////

    // All DataflowBufferSpecs must have unique names
    for (const auto& dfb : spec.dataflow_buffers) {
        dfbs[dfb.unique_id] = &dfb;
        auto [it, inserted] = dfbs.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate name '{}' found in dataflow_buffers", dfb.unique_id);
    }

    // A DFB must have exactly one producer and one consumer
    struct DFBProducerConsumerRecord {
        std::vector<const KernelSpec*> producer_kernels = {};
        std::vector<const KernelSpec*> consumer_kernels = {};
    };
    std::unordered_map<DFBSpecName, DFBProducerConsumerRecord> dfb_endpoints;
    for (const auto& kernel : spec.kernels) {
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            // Get the DFBProducerConsumerRecord for this DFB (if it doesn't exist, create it)
            DFBProducerConsumerRecord& pc_record = dfb_endpoints[dfb_binding.dfb_spec_name];

            // Add the kernel to the endpoint list
            if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::PRODUCER) {
                pc_record.producer_kernels.push_back(&kernel);
            } else if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::CONSUMER) {
                pc_record.consumer_kernels.push_back(&kernel);
            } else {
                TT_FATAL(false, "RELAY endpoints are only used for remote DFB, which is not supported yet");
            }
        }
    }
    for (const auto& [dfb_name, pc_record] : dfb_endpoints) {
        TT_FATAL(
            pc_record.producer_kernels.size() == 1,
            "DFB '{}' has {} producer kernels; a DFB must have exactly one producer.",
            dfb_name,
            pc_record.producer_kernels.size());
        TT_FATAL(
            pc_record.consumer_kernels.size() == 1,
            "DFB '{}' has {} consumer kernels; a DFB must have exactly one consumer.",
            dfb_name,
            pc_record.consumer_kernels.size());
    }

    // Check for unbound DFBs
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            dfb_endpoints.contains(dfb.unique_id), "DFB '{}' is defined but not bound by any kernel", dfb.unique_id);
    }

    //////////////////////////////
    // SemaphoreSpec validation
    //////////////////////////////

    TT_FATAL(spec.semaphores.empty(), "Semaphores are not supported yet");

    // All SemaphoreSpecs must have unique names
    for (const auto& semaphore : spec.semaphores) {
        semaphores[semaphore.unique_id] = &semaphore;
        auto [it, inserted] = semaphores.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate name '{}' found in semaphores", semaphore.unique_id);
    }

    // ... TODO

    //////////////////////////////
    // WorkerSpec validation
    //////////////////////////////

    // Check that WorkerSpecs are provided
    TT_FATAL(spec.workers.has_value(), "Workers are required on Gen2+");
    const auto& workers = spec.workers.value();
    TT_FATAL(!workers.empty(), "At least one WorkerSpec is required");

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

    // Check each WorkerSpec
    for (const auto& worker : workers) {
        // A WorkerSpec must have at least one kernel
        TT_FATAL(!worker.kernels.empty(), "WorkerSpec '{}' has no kernels!", worker.unique_id);

        // Each KernelSpec's target nodes must contain the WorkerSpec's target nodes
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = kernels.at(kernel_name);
            TT_FATAL(
                nodes_contains(kernel_spec->target_nodes, worker.target_nodes),
                "Kernel '{}' target nodes must contain WorkerSpec '{}' target nodes",
                kernel_name,
                worker.unique_id);
        }

        // Each DFBSpec's target nodes must contain the WorkerSpec's target nodes
        for (const auto& dfb_name : worker.dataflow_buffers) {
            const auto& dfb_spec = dfbs.at(dfb_name);
            TT_FATAL(
                nodes_contains(dfb_spec->target_nodes, worker.target_nodes),
                "DFB '{}' target nodes must contain WorkerSpec '{}' target nodes",
                dfb_name,
                worker.unique_id);
        }

        // The WorkerSpec's kernels must (together) target a legal number of cores
        uint32_t num_dm_cores = 0;
        uint32_t num_compute_cores = 0;
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = kernels.at(kernel_name);
            if (std::holds_alternative<ComputeConfiguration>(kernel_spec->config_spec)) {
                num_compute_cores += kernel_spec->num_threads;
            }
            if (std::holds_alternative<DataMovementConfiguration>(kernel_spec->config_spec)) {
                num_dm_cores += kernel_spec->num_threads;
            }
        }
        TT_FATAL(
            num_compute_cores <= QUASAR_TENSIX_CORES_PER_NODE,
            "WorkerSpec '{}' has too many compute cores!",
            worker.unique_id);
        TT_FATAL(
            num_dm_cores <= QUASAR_DM_CORES_PER_NODE,
            "WorkerSpec '{}' has too many data movement cores!",
            worker.unique_id);
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
    // (TODO: Revisit once we have remote DFB support)
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

    return;
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

    // Handy helpers
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

// Helper: Assign or verify DM processor mask for a kernel on a worker.
// Returns the kernel's mask (existing or newly reserved).
// Updates cumulative_mask in place. Throws TT_FATAL on conflict or allocation failure.
DMProcessorMask AssignDMProcessors(
    const KernelSpec* kernel_spec,
    const KernelSpecName& kernel_name,
    const WorkerSpecName& worker_id,
    std::unordered_map<const KernelSpec*, DMProcessorMask>& kernel_masks,
    DMProcessorMask& cumulative_mask) {
    // Already assigned from a previous worker?
    if (kernel_masks.contains(kernel_spec)) {
        DMProcessorMask existing = kernel_masks.at(kernel_spec);

        // Check for conflict with what's already allocated
        TT_FATAL(
            !existing.conflicts_with(cumulative_mask),
            "Kernel '{}' requires processors already in use on WorkerSpec '{}'. "
            "The \"common DM cores\" assumption has been violated!",
            kernel_name,
            worker_id);

        // Otherwise, update the cumulative mask
        cumulative_mask |= existing;
        return existing;
    }

    // First time seeing this kernel - reserve new processors
    auto reserved = DMProcessorMask::reserve_n(kernel_spec->num_threads, cumulative_mask);
    TT_FATAL(
        reserved.has_value(),
        "Failed to reserve processors for DM kernel '{}' on WorkerSpec '{}'. "
        "The \"common DM cores\" assumption has been violated!",
        kernel_name,
        worker_id);

    // Update the kernel mask and cumulative mask
    DMProcessorMask mask = reserved.value();
    kernel_masks[kernel_spec] = mask;
    cumulative_mask |= mask;
    return mask;
}

// Helper: Assign compute processor mask for a kernel.
ComputeProcessorMask AssignComputeProcessors(const KernelSpec* kernel_spec, const KernelSpecName& kernel_name) {
    auto reserved = ComputeProcessorMask::reserve_n(kernel_spec->num_threads, ComputeProcessorMask::create(0x00));
    TT_FATAL(
        reserved.has_value(),
        "Compute kernel '{}' reservation failed; should be unreachable after validation.",
        kernel_name);
    return reserved.value();
}

Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation = false) {
    auto impl = std::make_shared<detail::ProgramImpl>();

    //////////////////////////////
    // Legality checks
    //////////////////////////////

    // May want to make an option to skip legality checks in production to reduce runtime overhead.
    // If you skip legality checks, you will get undefined behavior if the ProgramSpec is invalid.
    if (skip_validation) {
        TT_FATAL(false, "For now, bypassing legality checks on Quasar isn't allowed.");
    }
    ValidateProgramSpec(spec);

    //////////////////////////////////////
    // Kernel->core assignments
    /////////////////////////////////////

    // Mapping from kernel names to KernelSpecs (inefficiency -- also created in legality checks)
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    for (const auto& kernel : spec.kernels) {
        kernel_by_name[kernel.unique_id] = &kernel;
    }

    // Processor masks for each kernel in the ProgramSpec
    std::unordered_map<const KernelSpec*, DMProcessorMask> kernel_dm_processor_mask;
    std::unordered_map<const KernelSpec*, ComputeProcessorMask> kernel_compute_processor_mask;

    // NOTE:
    // The current implementation makes a simplifying assumption:
    // A given DM kernel will run on the _same_ set of DM cores on every node/cluster.
    // This assumption is overly restrictive! it is easy to create a counterexample where it doesn't hold.
    // If we encounter a case that violates the simplifying assumption, ValidateProgramSpec will succeed,
    // but kernel->core assignment will fail. The error message will make it clear what happened.
    //
    // While the initial Quasar implementation uses this simplifying assumption, this is probably temporary.
    // When the time comes to relax this assumption, several aspects of the implementation will need to change:
    //   - The simple solver logic here will need to be re-written.
    //   - DFBs will need to be created per-KernelGroup/WorkerSpec, not per-kernel.
    //   - One DFBSpec may map to multiple DFBHandles.
    //   - DFB bindings will no longer have the character of CTAs! They'll act like implicit RTAs.
    //   - Possibly other knock-on effects, TBD.

    for (const auto& worker : spec.workers.value()) {
        // Cumulative DM processor mask for this WorkerSpec
        // (If we decide to reserve DM cores for interal use, just update the initial mask.)
        DMProcessorMask cumulative_dm_mask = DMProcessorMask::create(0x00);

        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = kernel_by_name.at(kernel_name);

            if (std::holds_alternative<DataMovementConfiguration>(kernel_spec->config_spec)) {
                AssignDMProcessors(
                    kernel_spec, kernel_name, worker.unique_id, kernel_dm_processor_mask, cumulative_dm_mask);
            } else {
                kernel_compute_processor_mask[kernel_spec] = AssignComputeProcessors(kernel_spec, kernel_name);
            }
        }
    }

    //////////////////////////////
    // Create DataflowBuffers
    //////////////////////////////

    // Mapping from DFB names to DataflowBufferSpecs (also created in legality checks)
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    for (const auto& dfb : spec.dataflow_buffers) {
        dfb_by_name[dfb.unique_id] = &dfb;
    }

    // Add DFBs to the Program
    //

    // Generate DFBs

    // Generate kernels

    return Program(std::move(impl));
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
