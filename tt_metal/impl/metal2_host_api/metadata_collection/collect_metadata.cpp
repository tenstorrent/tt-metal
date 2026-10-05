// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"

namespace tt::tt_metal::experimental {

template <typename KernelId>
void ValidateAccessorNameLength(const KernelId& kernel_id, std::string_view kind, std::string_view name) {
    TT_FATAL(
        name.size() <= MAX_ACCESSOR_NAME_LENGTH,
        "Kernel '{}' {} accessor_name '{}' is {} characters; an accessor_name must be at most {} characters",
        kernel_id,
        kind,
        name,
        name.size(),
        MAX_ACCESSOR_NAME_LENGTH);
}

CollectedSpecData CollectSpecData(const ProgramSpec& spec) {
    CollectedSpecData collected;

    // Collect KernelSpecs
    for (const auto& kernel : spec.kernels) {
        auto [it, inserted] = collected.kernel_by_name.try_emplace(kernel.unique_id, &kernel);
        TT_FATAL(inserted, "Duplicate KernelSpec name '{}'", kernel.unique_id);
    }

    // Collect DataflowBufferSpecs (local DFBs)
    for (const auto& dfb : spec.dataflow_buffers) {
        auto [it, inserted] = collected.dfb_by_name.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate DataflowBufferSpec name '{}'", dfb.unique_id);
    }

    // Collect CrossNodeDataflowBufferSpecs (cross-node DFBs).
    // Cross-node DFBs share the DFB name space with local DFBs, since kernel bindings
    // refer to either kind by the same DFBSpecName.
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        auto [it1, inserted1] = collected.dfb_by_name.try_emplace(name, &cross_node_dfb.dfb_spec);
        TT_FATAL(inserted1, "Duplicate DataflowBufferSpec name '{}' (across local and cross-node DFBs)", name);
        auto [it2, inserted2] = collected.cross_node_dfb_by_name.try_emplace(name, &cross_node_dfb);
        TT_FATAL(inserted2, "Duplicate CrossNodeDataflowBufferSpec name '{}'", name);
    }

    // Build DFB endpoint info from kernel bindings
    for (const auto& kernel : spec.kernels) {
        // Track per-accessor-name signatures within this kernel. Reusing a single
        // accessor_name across two DFBBindings is permitted as a "self-loop pair":
        // both bindings target the same DFB with opposite endpoint types (one PRODUCER,
        // one CONSUMER). This lets a kernel that both produces and consumes the same DFB
        // use a single device-side accessor name instead of two aliasing wrappers.
        struct AccessorBindingInfo {
            DFBSpecName dfb_spec_name;
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<std::string, AccessorBindingInfo> accessor_bindings;
        // Track, per DFB, which endpoint roles this kernel has already bound. Within a kernel a DFB
        // may be bound at most once per role; the only multi-binding form is the self-loop pair (one
        // PRODUCER + one CONSUMER, whose accessor names may differ). A second binding of the same
        // role under a different accessor name is the forbidden "one buffer, two names" aliasing
        // (see the check below). Scoped to the kernel so it resets per iteration — a DFB legitimately
        // carries different accessor names on different kernels (producer 'out', consumer 'in'), so
        // this must not be global.
        struct DFBBoundRoles {
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<DFBSpecName, DFBBoundRoles> dfb_bound_roles;
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            auto [it, inserted] = accessor_bindings.try_emplace(
                dfb_binding.accessor_name, AccessorBindingInfo{dfb_binding.dfb_spec_name});
            AccessorBindingInfo& info = it->second;
            if (inserted) {
                TT_FATAL(
                    IsValidCppIdentifier(dfb_binding.accessor_name),
                    "Kernel '{}' DFB accessor_name '{}' must be a valid C++ identifier",
                    kernel.unique_id,
                    dfb_binding.accessor_name);
                ValidateAccessorNameLength(kernel.unique_id, "DFB", dfb_binding.accessor_name);
            } else {
                TT_FATAL(
                    info.dfb_spec_name == dfb_binding.dfb_spec_name,
                    "Kernel '{}' uses accessor_name '{}' for two different DFBs ('{}' and '{}'). "
                    "Reusing a name is only permitted when both bindings target the same DFB (self-loop pair).",
                    kernel.unique_id,
                    dfb_binding.accessor_name,
                    info.dfb_spec_name,
                    dfb_binding.dfb_spec_name);
            }
            const bool is_producer = (dfb_binding.endpoint_type == DFBEndpointType::PRODUCER);
            bool& seen_this_type = is_producer ? info.has_producer : info.has_consumer;
            TT_FATAL(
                !seen_this_type,
                "Kernel '{}' has duplicate {} binding for accessor_name '{}'",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.accessor_name);
            seen_this_type = true;

            // Forbid binding the same DFB twice in the same role within this kernel (e.g. two CONSUMER
            // bindings under different accessor names). The legitimate multi-binding form is the
            // self-loop pair — one PRODUCER + one CONSUMER — which this allows regardless of whether
            // the two bindings share an accessor name. The same-role same-name case is already caught
            // above (duplicate {PRODUCER,CONSUMER} binding for accessor_name); this closes the
            // different-name gap. "One buffer, two names" in kernel code must be a handle alias
            // (constexpr auto x = dfb::y) over a single binding, not a second binding — two accessors /
            // DataflowBuffer objects for one FIFO break the object<->DFB identity that device-side
            // debug tooling relies on.
            DFBBoundRoles& bound_roles = dfb_bound_roles[dfb_binding.dfb_spec_name];
            bool& role_already_bound = is_producer ? bound_roles.has_producer : bound_roles.has_consumer;
            TT_FATAL(
                !role_already_bound,
                "Kernel '{}' has two {} bindings to DFB '{}' under different accessor names. Within a "
                "kernel a DFB may be bound at most once per role (the only multi-binding form is the "
                "self-loop pair: one PRODUCER + one CONSUMER). To refer to one buffer by multiple names "
                "in kernel code, alias the handle (constexpr auto x = dfb::y) instead of adding a second binding.",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.dfb_spec_name);
            role_already_bound = true;

            // Referential integrity: the DFB must exist
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

    // Completeness: every DFB must have at least one producer and one consumer.
    // (Cross-role coverage matching and within-role binding-site uniformity are checked
    // later, after kernel node coverage is computed.)
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        TT_FATAL(!endpoint_info.producers.empty(), "DFB '{}' has no producer", dfb_name);
        TT_FATAL(!endpoint_info.consumers.empty(), "DFB '{}' has no consumer", dfb_name);
    }

    // Referential integrity: every declared DFB (local or cross-node) must be bound by some kernel
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        TT_FATAL(
            collected.dfb_endpoints.contains(name),
            "CrossNodeDataflowBufferSpec '{}' is defined but not bound by any kernel",
            name);
    }

    // Collect SemaphoreSpecs
    for (const auto& semaphore : spec.semaphores) {
        auto [it, inserted] = collected.semaphore_by_name.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate SemaphoreSpec name '{}'", semaphore.unique_id);
    }

    // Validate semaphore bindings
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.semaphore_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate semaphore accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' semaphore accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "semaphore", binding.accessor_name);
            TT_FATAL(
                collected.semaphore_by_name.contains(binding.semaphore_spec_name),
                "Kernel '{}' references unknown semaphore '{}'",
                kernel.unique_id,
                binding.semaphore_spec_name);
        }
    }

    // Collect ScratchpadSpecs
    for (const auto& scratchpad : spec.scratchpads) {
        auto [it, inserted] = collected.scratchpad_by_name.try_emplace(scratchpad.unique_id, &scratchpad);
        TT_FATAL(inserted, "Duplicate ScratchpadSpec name '{}'", scratchpad.unique_id);
        TT_FATAL(
            scratchpad.size_per_node != 0,
            "ScratchpadSpec '{}' has size_per_node == 0; a scratchpad must reserve a non-zero number of bytes "
            "(did you forget to set size_per_node?).",
            scratchpad.unique_id);
    }

    // Collect scratchpad bindings (structural checks here; the node-set placement check is in
    // ValidateProgramSpec, which has the derived kernel node sets).
    // A scratchpad is private, node-local L1. More than one KernelSpec may bind the same
    // ScratchpadSpec, but only on disjoint node sets — the same node-local-resource discipline as a
    // local DFB. Same-node co-binding (true sharing) is gated behind a future AdvancedOption and is
    // rejected by the per-node census in ValidateProgramSpec.
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        std::unordered_set<ScratchpadSpecName> bound_specs;
        for (const auto& binding : kernel.scratchpad_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate scratchpad accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' scratchpad accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "scratchpad", binding.accessor_name);
            TT_FATAL(
                collected.scratchpad_by_name.contains(binding.scratchpad_spec_name),
                "Kernel '{}' references unknown scratchpad '{}'",
                kernel.unique_id,
                binding.scratchpad_spec_name);
            // A kernel may bind a given scratchpad at most once. Two bindings would request two
            // separate per-node allocations of the same spec under one kernel — muddy semantics, and
            // a node-level violation of "one binding instance per node". This is structural (no node
            // info needed), so it is caught here rather than in the placement census.
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
    // Every declared scratchpad must be bound by some kernel: an unbound scratchpad would reserve L1
    // that no kernel can reach.
    for (const auto& scratchpad : spec.scratchpads) {
        TT_FATAL(
            collected.scratchpad_binders.contains(scratchpad.unique_id),
            "ScratchpadSpec '{}' is declared but not bound by any kernel.",
            scratchpad.unique_id);
    }

    // Collect TensorParameters
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        auto [it, inserted] =
            collected.tensor_parameter_by_name.try_emplace(tensor_parameter.unique_id, &tensor_parameter);
        TT_FATAL(inserted, "Duplicate TensorParameter name '{}'", tensor_parameter.unique_id);
    }

    // Validate kernel tensor bindings
    for (const auto& kernel : spec.kernels) {
        // A tensor binding is legal on both DM and compute kernels:
        //   - a DM kernel can use the binding token to construct a TensorAccessor or LocalTensorAccessor
        //   - a compute kernel can only use LocalTensorAccessor (NOC-free, local-L1 only)

        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.tensor_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate tensor accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' tensor accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "tensor", binding.accessor_name);
            TT_FATAL(
                collected.tensor_parameter_by_name.contains(binding.tensor_parameter_name),
                "Kernel '{}' references unknown TensorParameter '{}'",
                kernel.unique_id,
                binding.tensor_parameter_name);

            collected.tensor_parameter_users[binding.tensor_parameter_name].push_back(&kernel);
        }

        std::unordered_set<std::string> reserved_type_aliases;
        reserved_type_aliases.reserve(accessor_names.size());
        for (const auto& binding_name : accessor_names) {
            reserved_type_aliases.insert(binding_name + "_t");
        }

        std::unordered_set<std::string> sequence_names;
        for (const auto& sequence : kernel.advanced_options.tensor_binding_sequences) {
            TT_FATAL(
                IsValidCppIdentifier(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                sequence.sequence_name);
            TT_FATAL(
                !accessor_names.contains(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' collides with a TensorBinding accessor_name",
                kernel.unique_id,
                sequence.sequence_name);
            TT_FATAL(
                !reserved_type_aliases.contains(sequence.sequence_name),
                "Kernel '{}' tensor binding sequence_name '{}' collides with generated type alias '{}'",
                kernel.unique_id,
                sequence.sequence_name,
                sequence.sequence_name);
            auto [sit, sinserted] = sequence_names.insert(sequence.sequence_name);
            TT_FATAL(
                sinserted,
                "Kernel '{}' has duplicate tensor binding sequence_name '{}'",
                kernel.unique_id,
                sequence.sequence_name);

            std::unordered_set<std::string> member_names;
            for (const auto& member : sequence.members) {
                TT_FATAL(
                    accessor_names.contains(member),
                    "Kernel '{}' tensor binding sequence '{}' references unknown tensor accessor_name '{}'",
                    kernel.unique_id,
                    sequence.sequence_name,
                    member);
                auto [mit, minserted] = member_names.insert(member);
                TT_FATAL(
                    minserted,
                    "Kernel '{}' tensor binding sequence '{}' has duplicate member '{}'",
                    kernel.unique_id,
                    sequence.sequence_name,
                    member);
            }
        }
    }

    // A borrowed-memory DFB uses its backing TensorParameter via DataflowBufferSpec::borrowed_from
    // (the DFB resolves its L1 address from that parameter's TensorArgument at runtime) even when no
    // kernel binds the parameter directly. Count that as a use so the completeness check below doesn't
    // reject a borrowed-only parameter. Existence of the referent is validated separately in the
    // borrowed-DFB checks. Only local DFBs are walked here: borrowed memory is a local-L1 feature,
    // so spec.dataflow_buffers is the relevant set (cross-node DFBs are runtime-unsupported).
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.borrowed_from.has_value()) {
            collected.tensor_parameter_users[*dfb.borrowed_from];  // register as used (no kernel user)
        }
    }

    // Referential integrity: every declared TensorParameter must be referenced by some kernel
    // binding or a DFB borrowed_from. (Same usage requirement as DFBs; an unused tensor parameter
    // is a user error.)
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        TT_FATAL(
            collected.tensor_parameter_users.contains(tensor_parameter.unique_id),
            "TensorParameter '{}' is defined but not bound by any kernel",
            tensor_parameter.unique_id);
    }

    // Collect PrefetcherPipeParameters
    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        auto [it, inserted] = collected.prefetcher_pipe_by_name.try_emplace(pipe_parameter.unique_id, &pipe_parameter);
        TT_FATAL(inserted, "Duplicate PrefetcherPipeParameter name '{}'", pipe_parameter.unique_id);
    }

    // Validate kernel PrefetcherPipe bindings (structural). Kernel-kind and role checks need the
    // derived node sets and live in ValidateProgramSpec.
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        std::unordered_set<PrefetcherPipeParamName> bound_pipes;
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate PrefetcherPipe accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' PrefetcherPipe accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            ValidateAccessorNameLength(kernel.unique_id, "PrefetcherPipe", binding.accessor_name);
            TT_FATAL(
                !binding.pipe_parameter_names.empty(),
                "Kernel '{}' PrefetcherPipe accessor '{}' names no PrefetcherPipeParameter",
                kernel.unique_id,
                binding.accessor_name);
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                TT_FATAL(
                    collected.prefetcher_pipe_by_name.contains(pipe_name),
                    "Kernel '{}' accessor '{}' references unknown PrefetcherPipeParameter '{}'",
                    kernel.unique_id,
                    binding.accessor_name,
                    pipe_name);
                // One binding per pipe per kernel, within and across accessors: a second binding
                // would be a second device object over the same credit counters (two names for one
                // pipe is a handle alias, not a binding).
                auto [pit, pinserted] = bound_pipes.insert(pipe_name);
                TT_FATAL(
                    pinserted,
                    "Kernel '{}' binds PrefetcherPipeParameter '{}' more than once (latest under accessor_name '{}'). "
                    "A kernel may bind a given pipe at most once.",
                    kernel.unique_id,
                    pipe_name,
                    binding.accessor_name);
                collected.prefetcher_pipe_users[pipe_name].binders.push_back({&kernel, &binding});
            }
        }
    }

    // Collect relay DFBs: each named pipe must exist and be named once per DFB. (Geometry checks
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
            collected.prefetcher_pipe_users[pipe_name].relays.push_back(&dfb);
        }
    }
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        TT_FATAL(
            cross_node_dfb.dfb_spec.advanced_options.prefetcher_pipe_relays.empty(),
            "CrossNodeDataflowBufferSpec '{}' sets prefetcher_pipe_relays; only a local DFB can relay a "
            "PrefetcherPipe",
            cross_node_dfb.dfb_spec.unique_id);
    }

    // Referential integrity: every declared PrefetcherPipeParameter must be used by a kernel binding
    // or a relay DFB. (An unused pipe parameter would demand a run arg nothing reads.)
    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        TT_FATAL(
            collected.prefetcher_pipe_users.contains(pipe_parameter.unique_id),
            "PrefetcherPipeParameter '{}' is defined but not bound by any kernel or relay DFB",
            pipe_parameter.unique_id);
    }

    // Build WorkUnitSpec membership for each kernel, validating references along the way.
    // (WorkUnitSpec.name is debug-only; no uniqueness invariant.)
    // A kernel may belong to multiple WorkUnitSpecs; its effective target node set is the union.
    for (const auto& work_unit : spec.work_units) {
        for (const auto& kernel_name : work_unit.kernels) {
            TT_FATAL(
                collected.kernel_by_name.contains(kernel_name),
                "WorkUnitSpec '{}' references unknown kernel '{}'",
                work_unit.name,
                kernel_name);
            collected.kernel_work_units[kernel_name].push_back(&work_unit);
        }
    }

    // Every declared kernel must be referenced by at least one WorkUnitSpec
    // (otherwise, it has no node placement and would never run).
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            collected.kernel_work_units.contains(kernel.unique_id),
            "Kernel '{}' is not referenced by any WorkUnitSpec",
            kernel.unique_id);
    }

    // Derive each kernel's effective target node set: union of containing WorkUnitSpec target_nodes.
    for (const auto& [kernel_name, work_units] : collected.kernel_work_units) {
        NodeRangeSet node_set;
        for (const WorkUnitSpec* work_unit : work_units) {
            node_set = node_set.merge(to_node_range_set(work_unit->target_nodes));
        }
        collected.kernel_node_set[kernel_name] = node_set;
    }

    // Derive each local DFB's allocation node set: union of binding-kernels' node sets.
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

    // Census the semaphore binders. Needs the kernel node sets derived above. Also rejects a kernel
    // that binds the same semaphore twice.
    collected.semaphore_binders = sem_solver::CollectSemaphoreBinders(spec, collected.kernel_node_set);

    return collected;
}

}  // namespace tt::tt_metal::experimental
