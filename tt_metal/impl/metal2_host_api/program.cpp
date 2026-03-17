// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/llrt/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

Program MakeProgramFromSpec(const ProgramSpec& spec) {
    auto impl = std::make_shared<detail::ProgramImpl>();

    // Legality checks
    // Can make these bypassable with a flag to reduce production runtime overhead
    ValidateProgramSpec(spec);

    // Data structures
    std::unordered_map<const KernelSpec*, NodeRangeSet> all_kernels;
    std::unordered_map<const DataflowBufferSpec*, NodeRangeSet> all_dfbs;




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

// Helper for uniqueness validation
namespace {
    template <typename Range>
    void check_unique_ids(std::unordered_set<std::string>& seen, const Range& items, std::string_view context) {
        for (const auto& item : items) {
            auto [it, inserted] = seen.insert(item.unique_id);
            TT_FATAL(inserted, "Duplicate name '{}' found in {}", item.unique_id, context);
        }
    }
}  // namespace

void ValidateUniqueIDs(const ProgramSpec& spec) {
    // All KernelSpecs have unique names
    {
        std::unordered_set<std::string> kernel_names;
        check_unique_ids(kernel_names, spec.data_movement_kernels, "data_movement_kernels");
        check_unique_ids(kernel_names, spec.compute_kernels, "compute_kernels");
    }

    // All DFBSpecs have unique names
    {
        std::unordered_set<std::string> dfb_names;
        check_unique_ids(dfb_names, spec.dataflow_buffers, "DataflowBufferSpecs");
    }

    // All SemaphoreSpecs have unique names
    {
        std::unordered_set<std::string> semaphore_names;
        check_unique_ids(semaphore_names, spec.semaphores, "SemaphoreSpecs");
    }

    // All WorkerSpecs have unique names
    {
        std::unordered_set<std::string> worker_names;
        check_unique_ids(worker_names, spec.workers.value(), "WorkerSpecs");
    }
}


void ValidateProgramSpec(const ProgramSpec& spec) {

    // Check target architecture
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR, "Metal 2.0 API is currently only for Quasar. WH/BH support coming soon.");

    //////////////////////////////
    // Uniqueness checks
    //////////////////////////////

    // All KernelSpecs have unique names
    {
        std::unordered_set<KernelSpecName> kernel_names;
        for (const auto& kernel : spec.data_movement_kernels) {
            auto [it, inserted] = kernel_names.insert(kernel.unique_id);
            TT_FATAL(inserted, "Duplicate KernelSpec name '{}' found in data_movement_kernels", kernel.unique_id);
        }
        for (const auto& kernel : spec.compute_kernels) {
            auto [it, inserted] = kernel_names.insert(kernel.unique_id);
            TT_FATAL(inserted, "Duplicate KernelSpec name '{}' found in compute_kernels", kernel.unique_id);
        }
    }

    // All DFBSpecs have unique names
    {
        std::unordered_set<DFBSpecName> dfb_names;
        for (const auto& dfb : spec.dataflow_buffers) {
            auto [it, inserted] = dfb_names.insert(dfb.unique_id);
            TT_FATAL(inserted, "Duplicate DataflowBufferSpec name '{}' found", dfb.unique_id);
        }
    }
    // All SemaphoreSpecs have unique names
    {
        std::unordered_set<SemaphoreSpecName> semaphore_names;
        for (const auto& semaphore : spec.semaphores) {
            auto [it, inserted] = semaphore_names.insert(semaphore.unique_id);
            TT_FATAL(inserted, "Duplicate SemaphoreSpec name '{}' found", semaphore.unique_id);
        }
    }
    // All WorkerSpecs have unique names
    {
        std::unordered_set<WorkerSpecName> worker_names;
        for (const auto& worker : spec.workers) {
            auto [it, inserted] = worker_names.insert(worker.unique_id);
            TT_FATAL(inserted, "Duplicate WorkerSpec name '{}' found", worker.unique_id);
        }
    }

    
    //////////////////////////////
    // WorkerSpec validation
    //////////////////////////////

     // WorkerSpecs must not overlap in their target nodes

    // WorkerSpecs must not overlap in their target nodes
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
        // WorkerSpec must have at least one kernel
        if (worker.kernels.empty()) { TT_FATAL("WorkerSpec has no kernels!"); }

        // TODO: Add semaphore support
        if (!worker.semaphores.empty()) { TT_FATAL("Semaphores aren't supported yet"); }

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





    //////////////////////////////
    // KernelSpec validation

    //////////////////////////////
    // DFBSpec validation

    //////////////////////////////
    // SemaphoreSpec validation
    TT_FATAL(spec.semaphores.empty(), "Semaphores are not supported yet");






    }


}

}  // namespace tt::tt_metal::experimental::metal2_host_api
