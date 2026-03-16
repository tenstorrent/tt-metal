// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

Program MakeProgramFromSpec(const ProgramSpec& spec) {
    (void)spec;  // Suppress unused parameter warning for stub implementation
    auto impl = std::make_shared<detail::ProgramImpl>();


    // Data structures
    std::unordered_map<const KernelSpec*, NodeRangeSet> all_kernels;
    std::unordered_map<const DataflowBufferSpec*, NodeRangeSet> all_dfbs;


    /////////////////////
    // LEGALITY CHECKS //
    /////////////////////

    // Ensure that WorkerSpecs don't overlap in target nodes
    for (const auto& worker : spec.workers) {
        for (const auto& other_worker : spec.workers) {
            if (worker.unique_id == other_worker.unique_id) { continue; }
            if (worker.target_nodes.intersects(other_worker.target_nodes)) {
                TT_FATAL("WorkerSpecs '{}' and '{}' overlap in target nodes", worker.unique_id, other_worker.unique_id);
            }
        }
    }

    // WorkerSpec legality checks
    for (const auto& worker : spec.workers) {
        if (worker.kernels.empty()) { TT_FATAL("WorkerSpec has no kernels!"); }
        if (!worker.semaphores.empty()) { TT_FATAL("Semaphores aren't supported yet (TODO)"); }

        // A KernelSpec must be specified only once per WorkerSpec (but is allowed in multiple WorkerSpecs)
        std::unordered_map<KernelSpecName, const KernelSpec*> worker_kernels;
        for (const auto& kernel : worker.kernels) {
            auto [it, inserted] = worker_kernels.try_emplace(kernel.unique_id, &kernel);
            if (!inserted) { TT_FATAL("Duplicate KernelSpec '{}' in WorkerSpec", kernel.unique_id); }
        }

        // A DFB must be specified only once per WorkerSpec (but is allowed in multiple WorkerSpecs)
        std::unordered_map<DFBSpecName, const DataflowBufferSpec*> worker_dfbs;
        for (const auto& dfb : worker.dataflow_buffers) {
            auto [it, inserted] = worker_dfbs.try_emplace(dfb.unique_id, &dfb);
            if (!inserted) { TT_FATAL("Duplicate DFB '{}' in WorkerSpec", dfb.unique_id); }
        }

        // Each KernelSpec's target nodes must contain the WorkerSpec's target nodes
        for (const auto& kernel : worker.kernels) {
            if (!kernel.target_nodes.contains(worker.target_nodes)) {
                TT_FATAL("Kernel '{}' target nodes must contain its WorkerSpec target nodes", kernel.unique_id);
            }
        }

        // Each DFB's target nodes must contain the WorkerSpec's target nodes
        for (const auto& dfb : worker.dataflow_buffers) {
            if (!dfb.target_nodes.contains(worker.target_nodes)) {
                TT_FATAL("DFB '{}' target nodes must contain its WorkerSpec target nodes", dfb.unique_id);
            }
        }

        // Check that this WorkerSpec's kernels target a legal number of cores
        uint32_t num_dm_cores = 0;
        uint32_t num_compute_cores = 0;
        for (const auto& kernel : worker.kernels) {
            if (compute_kernel) num_compute_cores += kernel.num_threads;
            if (data_movement_kernel) num_dm_cores += kernel.num_threads;
        }


        // TODO -- need to merge the target nodes of all the kernels and dfbs into a single NodeRangeSet

        all_kernels.insert(worker_kernels.begin(), worker_kernels.end()); // Add to global list (de-duplicated)
        all_dfbs.insert(worker_dfbs.begin(), worker_dfbs.end()); // Add to global list (de-duplicated)
    }





    // Remove duplicates from kernel_specs



    // Solve for kernel cores



    // Generate DFBs



    // Generate kernels



        // TODO: Add semaphores
        for (const auto& semaphore_spec : worker.semaphores) {
            (void)semaphore_spec;  // Placeholder
        }

        // TODO: Add dataflow buffers
        for (const auto& dfb_spec : worker.dataflow_buffers) {
            (void)dfb_spec;  // Placeholder
        }

        // TODO: Add kernels
        for (const auto& kernel_spec : worker.kernels) {
            (void)kernel_spec;  // Placeholder
        }
    }


    return Program(std::move(impl));
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
