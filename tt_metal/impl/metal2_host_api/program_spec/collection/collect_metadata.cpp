// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Phase 1 -- "Name -> spec lookups". Builds every *_by_name map.
// Establishes: the *_by_name invariants (non-null values, key == unique_id, and the
// cross_node_dfb_by_name <-> dfb_by_name link). Name uniqueness falls out of try_emplace.
void CollectNameLookups(const ProgramSpec& spec, CollectedSpecData& collected) {
    for (const auto& kernel : spec.kernels) {
        auto [it, inserted] = collected.kernel_by_name.try_emplace(kernel.unique_id, &kernel);
        TT_FATAL(inserted, "Duplicate KernelSpec name '{}'", kernel.unique_id);
    }

    // Local DFBs.
    for (const auto& dfb : spec.dataflow_buffers) {
        auto [it, inserted] = collected.dfb_by_name.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate DataflowBufferSpec name '{}'", dfb.unique_id);
    }

    // Cross-node DFBs share the DFB name space with local DFBs, since kernel bindings
    // refer to either kind by the same DFBSpecName.
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        auto [it1, inserted1] = collected.dfb_by_name.try_emplace(name, &cross_node_dfb.dfb_spec);
        TT_FATAL(inserted1, "Duplicate DataflowBufferSpec name '{}' (across local and cross-node DFBs)", name);
        auto [it2, inserted2] = collected.cross_node_dfb_by_name.try_emplace(name, &cross_node_dfb);
        TT_FATAL(inserted2, "Duplicate CrossNodeDataflowBufferSpec name '{}'", name);
    }

    for (const auto& semaphore : spec.semaphores) {
        auto [it, inserted] = collected.semaphore_by_name.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate SemaphoreSpec name '{}'", semaphore.unique_id);
    }

    for (const auto& scratchpad : spec.scratchpads) {
        auto [it, inserted] = collected.scratchpad_by_name.try_emplace(scratchpad.unique_id, &scratchpad);
        TT_FATAL(inserted, "Duplicate ScratchpadSpec name '{}'", scratchpad.unique_id);
    }

    for (const auto& tensor_parameter : spec.tensor_parameters) {
        auto [it, inserted] =
            collected.tensor_parameter_by_name.try_emplace(tensor_parameter.unique_id, &tensor_parameter);
        TT_FATAL(inserted, "Duplicate TensorParameter name '{}'", tensor_parameter.unique_id);
    }

    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        auto [it, inserted] = collected.prefetcher_pipe_by_name.try_emplace(pipe_parameter.unique_id, &pipe_parameter);
        TT_FATAL(inserted, "Duplicate PrefetcherPipeParameter name '{}'", pipe_parameter.unique_id);
    }
}

// Phase 2 -- "Resource -> Users lookups". Needs phase 1 (every reference is resolved against a *_by_name map).
// Establishes: the invariants of dfb_endpoints, scratchpad_binders and prefetcher_pipe_relays.
// Bindings to resources that have no stored index (semaphores, tensor parameters, pipes) are only
// checked for dangling references here.
void CollectResourceUsers(const ProgramSpec& spec, CollectedSpecData& collected) {
    // dfb_endpoints: one record per kernel DFB binding.
    for (const auto& kernel : spec.kernels) {
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            TT_FATAL(
                collected.dfb_by_name.contains(dfb_binding.dfb_spec_name),
                "Kernel '{}' references unknown DFB '{}'",
                kernel.unique_id,
                dfb_binding.dfb_spec_name);

            CollectedSpecData::DFBEndpointInfo& endpoint_info = collected.dfb_endpoints[dfb_binding.dfb_spec_name];

            if (dfb_binding.endpoint_type == DFBEndpointType::PRODUCER) {
                endpoint_info.producers.push_back({&kernel, &dfb_binding});
            } else if (dfb_binding.endpoint_type == DFBEndpointType::CONSUMER) {
                endpoint_info.consumers.push_back({&kernel, &dfb_binding});
            }
        }
    }

    // dfb_endpoints invariant: neither list is empty. (Cross-role coverage matching and within-role
    // binding-site uniformity are checked later, after kernel node coverage is computed.)
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        TT_FATAL(!endpoint_info.producers.empty(), "DFB '{}' has no producer", dfb_name);
        TT_FATAL(!endpoint_info.consumers.empty(), "DFB '{}' has no consumer", dfb_name);
    }

    // dfb_endpoints invariant: every local DFB has an entry. (The cross-node equivalent is in
    // ValidateProgramMisc.)
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }

    // scratchpad_binders: more than one kernel may bind the same ScratchpadSpec (the node-set
    // placement check is in ValidateProgramSpec), but a kernel binds a given scratchpad at most once,
    // which keeps each vector free of repeated kernels.
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<ScratchpadSpecName> bound_specs;
        for (const auto& binding : kernel.scratchpad_bindings) {
            TT_FATAL(
                collected.scratchpad_by_name.contains(binding.scratchpad_spec_name),
                "Kernel '{}' references unknown scratchpad '{}'",
                kernel.unique_id,
                binding.scratchpad_spec_name);
            auto [sit, sinserted] = bound_specs.insert(binding.scratchpad_spec_name);
            TT_FATAL(
                sinserted,
                "Kernel '{}' binds scratchpad '{}' more than once (latest under accessor_name '{}'). A "
                "kernel may bind a given scratchpad at most once.",
                kernel.unique_id,
                binding.scratchpad_spec_name,
                binding.accessor_name);
            collected.scratchpad_binders[binding.scratchpad_spec_name].push_back(&kernel);
        }
    }

    // prefetcher_pipe_relays: each named pipe must exist and be named once per DFB. (Geometry checks
    // against the pipe are in ValidateProgramSpec.) Only local DFBs can relay: the ring lives in
    // receiver-node L1.
    for (const auto& dfb : spec.dataflow_buffers) {
        std::unordered_set<PrefetcherPipeParamName> relayed;
        for (const auto& pipe_name : dfb.advanced_options.prefetcher_pipe_relays) {
            TT_FATAL(
                collected.prefetcher_pipe_by_name.contains(pipe_name),
                "DFB '{}' relays unknown PrefetcherPipeParameter '{}'",
                dfb.unique_id,
                pipe_name);
            auto [it, inserted] = relayed.insert(pipe_name);
            TT_FATAL(
                inserted,
                "DFB '{}' lists PrefetcherPipeParameter '{}' more than once in prefetcher_pipe_relays",
                dfb.unique_id,
                pipe_name);
            collected.prefetcher_pipe_relays[pipe_name].push_back(&dfb);
        }
    }

    // Dangling-reference checks for bindings that are not indexed (nothing downstream reads a
    // reverse index for them).
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.semaphore_bindings) {
            TT_FATAL(
                collected.semaphore_by_name.contains(binding.semaphore_spec_name),
                "Kernel '{}' references unknown semaphore '{}'",
                kernel.unique_id,
                binding.semaphore_spec_name);
        }
        for (const auto& binding : kernel.tensor_bindings) {
            TT_FATAL(
                collected.tensor_parameter_by_name.contains(binding.tensor_parameter_name),
                "Kernel '{}' references unknown TensorParameter '{}'",
                kernel.unique_id,
                binding.tensor_parameter_name);
        }
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                TT_FATAL(
                    collected.prefetcher_pipe_by_name.contains(pipe_name),
                    "Kernel '{}' accessor '{}' references unknown PrefetcherPipeParameter '{}'",
                    kernel.unique_id,
                    binding.accessor_name,
                    pipe_name);
            }
        }
    }
}

// Phase 3 -- "Derived node sets". Needs phases 1 and 2.
// Establishes: the invariants of kernel_node_set and dfb_node_set.
void DeriveNodeSets(const ProgramSpec& spec, CollectedSpecData& collected) {
    // kernel_node_set: union of the target_nodes of every WorkUnitSpec containing the kernel (a kernel
    // may belong to multiple WorkUnitSpecs). (WorkUnitSpec.name is debug-only; no uniqueness invariant.)
    for (const auto& work_unit : spec.work_units) {
        for (const auto& kernel_name : work_unit.kernels) {
            TT_FATAL(
                collected.kernel_by_name.contains(kernel_name),
                "WorkUnitSpec '{}' references unknown kernel '{}'",
                work_unit.name,
                kernel_name);
            NodeRangeSet& node_set = collected.kernel_node_set[kernel_name];
            node_set = node_set.merge(to_node_range_set(work_unit.target_nodes));
        }
    }

    // kernel_node_set invariant: exactly the kernels in kernel_by_name have an entry (a kernel in no
    // WorkUnitSpec would have no node placement and never run).
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            collected.kernel_node_set.contains(kernel.unique_id),
            "Kernel '{}' is not referenced by any WorkUnitSpec",
            kernel.unique_id);
    }

    // dfb_node_set: union of the endpoint kernels' node sets, for each local DFB.
    // (Collected, but unvalidated. Semantic integrity checks for DFB take place in ValidateProgramSpec.
    //  Once those pass, producer and consumer coverages are guaranteed equal; here we union both
    //  sides for safety before that guarantee holds.)
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        NodeRangeSet node_set;
        for (const auto& rec : endpoints.producers) {
            node_set = node_set.merge(collected.kernel_node_set.at(rec.kernel->unique_id));
        }
        for (const auto& rec : endpoints.consumers) {
            node_set = node_set.merge(collected.kernel_node_set.at(rec.kernel->unique_id));
        }
        collected.dfb_node_set[dfb.unique_id] = node_set;
    }
}

}  // namespace

CollectedSpecData CollectSpecData(const ProgramSpec& spec) {
    CollectedSpecData collected;

    // Phases follow the section order of CollectedSpecData; each one establishes that section's invariants.
    CollectNameLookups(spec, collected);
    CollectResourceUsers(spec, collected);
    DeriveNodeSets(spec, collected);

    // semaphore_binders: needs the kernel node sets. Also rejects a kernel that binds the same
    // semaphore twice.
    collected.semaphore_binders = sem_solver::CollectSemaphoreBinders(spec, collected.kernel_node_set);

    return collected;
}

}  // namespace tt::tt_metal::experimental
