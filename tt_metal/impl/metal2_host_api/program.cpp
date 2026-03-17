// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

// TODO: These constants should be queriable from the public API (currently HAL, for consistency)
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_TENSIX_CORES_PER_NODE = 4;


// Helper to convert Nodes variant to NodeRangeSet
NodeRangeSet to_node_range_set(const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes) {
    return std::visit([](const auto& n) -> NodeRangeSet {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, NodeRangeSet>) {
            return n;
        } else if constexpr (std::is_same_v<T, NodeRange>) {
            return NodeRangeSet(n);
        } else {
            // NodeCoord case
            return NodeRangeSet(NodeRange(n, n));
        }
    }, nodes);
}

// Helper to check if two Nodes variants intersect
bool nodes_intersect(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& a,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& b) {
    NodeRangeSet a_set = to_node_range_set(a);
    NodeRangeSet b_set = to_node_range_set(b);
    return a_set.intersects(b_set);
}

// Helper to check if one Nodes variant contains another
bool nodes_contains(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& superset,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& subset) {
    NodeRangeSet superset_node_range_set = to_node_range_set(superset);
    NodeRangeSet subset_node_range_set = to_node_range_set(subset);
    return superset_node_range_set.contains(subset_node_range_set);
}

// Forward declaration
void ValidateProgramSpec(const ProgramSpec& spec);


Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation = false) {
    auto impl = std::make_shared<detail::ProgramImpl>();

    // Legality checks
    if (!skip_validation) {
        ValidateProgramSpec(spec);
    }

    // Solve for kernel cores



    // Generate DFBs



    // Generate kernels



    return Program(std::move(impl));
}


void ValidateProgramSpec(const ProgramSpec& spec) {

    // Check target architecture
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR, 
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

    // Check thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (std::holds_alternative<ComputeConfiguration>(kernel.config_spec)) {
            TT_FATAL(kernel.num_threads <= QUASAR_TENSIX_CORES_PER_NODE, "KernelSpec '{}' has too many threads!", kernel.unique_id);
        }
        if (std::holds_alternative<DataMovementConfiguration>(kernel.config_spec)) {
            TT_FATAL(kernel.num_threads <= QUASAR_DM_CORES_PER_NODE, "KernelSpec '{}' has too many threads!", kernel.unique_id);
        }
    }

    // Check config specs
    for (const auto& kernel : spec.kernels) {
        if (std::holds_alternative<DataMovementConfiguration>(kernel.config_spec)) {    
            const auto& data_movement_config = std::get<DataMovementConfiguration>(kernel.config_spec);
            TT_FATAL(data_movement_config.gen1_data_movement_config.has_value() || data_movement_config.gen2_data_movement_config.has_value(), 
              "KernelSpec '{}' must specify a DM config for Gen1, Gen2, or both.", kernel.unique_id);
        }
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
            if (worker.unique_id == other_worker.unique_id) { continue; }
            if (nodes_intersect(worker.target_nodes, other_worker.target_nodes)) {
                TT_FATAL(false, "WorkerSpecs '{}' and '{}' overlap in target nodes", worker.unique_id, other_worker.unique_id);
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
            TT_FATAL(nodes_contains(kernel_spec->target_nodes, worker.target_nodes), 
               "Kernel '{}' target nodes must contain WorkerSpec '{}' target nodes", kernel_name, worker.unique_id);
        }

        // Each DFBSpec's target nodes must contain the WorkerSpec's target nodes
        for (const auto& dfb_name : worker.dataflow_buffers) {
            const auto& dfb_spec = dfbs.at(dfb_name);
            TT_FATAL(nodes_contains(dfb_spec->target_nodes, worker.target_nodes), 
               "DFB '{}' target nodes must contain WorkerSpec '{}' target nodes", dfb_name, worker.unique_id);
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
        TT_FATAL(num_compute_cores <= QUASAR_TENSIX_CORES_PER_NODE, "WorkerSpec '{}' has too many compute cores!", worker.unique_id);
        TT_FATAL(num_dm_cores <= QUASAR_DM_CORES_PER_NODE, "WorkerSpec '{}' has too many data movement cores!", worker.unique_id);
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
        TT_FATAL(worker_derived_target_nodes == kernel_target_nodes, 
            "Kernel '{}' has target nodes that are not accounted for by any WorkerSpec", kernel_name);
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
        TT_FATAL(worker_derived_target_nodes == dfb_target_nodes, 
            "DFB '{}' has target nodes that are not accounted for by any WorkerSpec", dfb_name);
    }  

    return;
}




}  // namespace tt::tt_metal::experimental::metal2_host_api
