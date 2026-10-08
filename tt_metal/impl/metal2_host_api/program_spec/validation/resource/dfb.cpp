// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Validate borrowed-memory DFBs.
//
// A borrowed-memory DFB names a TensorParameter via DataflowBufferSpec::borrowed_from. The
// backing MeshTensor flows through ProgramRunArgs::tensor_args at execution time.
// We enforce only the safety-relevant checks:
//  - the named parameter exists
//  - the TensorSpec places storage in L1,
//  - the spec is large enough
// We don't validate any layout considerations (interleaved vs sharded, page / tile sizes, etc.)
// That's on the user; this is an advanced feature.
void ValidateBorrowedMemory(
    const DataflowBufferSpec& dfb,
    const CollectedSpecData& collected,
    uint32_t l1_alignment,
    const NumBanksFromBufferType& num_banks_from_buffer_type) {
    if (!dfb.borrowed_from.has_value()) {
        return;
    }
    const TensorParamName& tp_name = *dfb.borrowed_from;
    auto it = collected.tensor_parameter_by_name.find(tp_name);
    TT_FATAL(
        it != collected.tensor_parameter_by_name.end(),
        "DFB '{}' borrows memory from TensorParameter '{}', but no such TensorParameter is declared in the "
        "ProgramSpec.",
        dfb.unique_id,
        tp_name);
    const TensorSpec& tensor_spec = it->second->spec;
    TT_FATAL(
        tensor_spec.memory_config().is_l1(),
        "DFB '{}' borrows memory from TensorParameter '{}', but its TensorSpec is not L1-resident (L1 is "
        "required). Both L1 and L1_SMALL are accepted.",
        dfb.unique_id,
        tp_name);
    // Spec-time sizing check. A borrowed DFB lives in ONE core's slice of the backing buffer,
    // so the bound is that buffer's per-bank allocation.
    //
    // TensorSpec yields the per-bank figure without a Buffer: sharded specs take pages-per-bank
    // from the shard spec or the distribution spec, interleaved specs divide their page count by
    // num_banks.
    // The attach-time check in AttachBorrowedDFBBuffers (program_run_args.cpp) stays
    // authoritative; this one just stops deferring a rejection it can already make.
    //
    // Caveat: this is the default allocator, which has no sub-device context. A sub-device
    // allocator owns fewer banks, so an interleaved tensor allocated there has a LARGER per-bank
    // slice than what we compute, and a DFB sized to it would be rejected here even though
    // attach time would take it. Sharded specs ignore num_banks and so are unaffected. Thread a
    // SubDeviceId in here if that combination ever shows up.
    const uint32_t num_banks = num_banks_from_buffer_type(tensor_spec.memory_config().buffer_type());
    const size_t dfb_bytes = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
    const size_t tensor_bytes = tensor_spec.compute_consumed_memory_bytes_per_bank(l1_alignment, num_banks);
    TT_FATAL(
        dfb_bytes <= tensor_bytes,
        "DFB '{}' (entry_size {} * num_entries {} = {} bytes) is larger than the per-bank allocation of its "
        "borrowed TensorParameter '{}' ({} bytes).",
        dfb.unique_id,
        dfb.entry_size,
        dfb.num_entries,
        dfb_bytes,
        tp_name,
        tensor_bytes);
}

}  // namespace

void ValidateDFBSpec(
    const DataflowBufferSpec& dfb,
    const CollectedSpecData& collected,
    uint32_t l1_alignment,
    tt::ARCH arch,
    const NumBanksFromBufferType& num_banks_from_buffer_type) {
    // Validate per-DFB sizing: entry_size and num_entries must be set to non-zero values.
    // (Sizes may still be overridden at runtime via ProgramRunArgs, but a ProgramSpec value is required.)
    TT_FATAL(
        dfb.entry_size > 0,
        "DataflowBufferSpec '{}' has entry_size = 0. entry_size must be set to a non-zero value.",
        dfb.unique_id);
    TT_FATAL(
        dfb.num_entries > 0,
        "DataflowBufferSpec '{}' has num_entries = 0. num_entries must be set to a non-zero value.",
        dfb.unique_id);

    // Data format must be valid for the architecture
    if (dfb.data_format_metadata.has_value()) {
        TT_FATAL(
            tt::is_data_format_supported(dfb.data_format_metadata.value(), arch),
            "DFB '{}' has data format '{}' which is not supported on architecture {}",
            dfb.unique_id,
            dfb.data_format_metadata.value(),
            arch);
    }

    // The allow_instance_multi_binding escape hatch is Gen1-only. On Gen2 the DFB's per-RISC
    // tile-counter / remapper machinery is driven by the producer/consumer masks, so a multi-bound
    // instance cannot be lowered. Reject the flag itself on Gen2, independent of whether any instance
    // is actually multi-bound — a Gen2 spec carrying it is never valid.
    if (is_gen2_arch(arch)) {
        TT_FATAL(
            !dfb.advanced_options.allow_instance_multi_binding,
            "DFB '{}' sets allow_instance_multi_binding, which is only supported on Gen1 (WH/BH) "
            "architectures. On Gen2 a DFB instance must have exactly one producer and one consumer.",
            dfb.unique_id);
    }

    // Every name in alias_with must refer to a real DFB and must not be self-referential.
    for (const auto& alias_name : dfb_alias_with(dfb)) {
        TT_FATAL(
            collected.dfb_by_name.contains(alias_name),
            "DFB '{}' lists unknown alias '{}' in alias_with",
            dfb.unique_id,
            alias_name);
        TT_FATAL(alias_name != dfb.unique_id, "DFB '{}' lists itself in alias_with", dfb.unique_id);
    }

    TT_FATAL(
        dfb.advanced_options.prefetcher_pipe_relays.empty() || !dfb.borrowed_from.has_value(),
        "DFB '{}' sets both prefetcher_pipe_relays and borrowed_from; a relay DFB's backing memory is the "
        "pipe ring",
        dfb.unique_id);

    ValidateBorrowedMemory(dfb, collected, l1_alignment, num_banks_from_buffer_type);
}

// DFB bindings: accessor names, self-loop pairs, role aliasing
void ValidateDFBBindings(const KernelSpec& kernel, const ValidationContext& ctx) {
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
    // (see the check below). Scoped to the kernel — a DFB legitimately carries different accessor
    // names on different kernels (producer 'out', consumer 'in'), so this must not be global.
    struct DFBBoundRoles {
        bool has_producer = false;
        bool has_consumer = false;
    };
    std::unordered_map<DFBSpecName, DFBBoundRoles> dfb_bound_roles;
    for (const auto& dfb_binding : kernel.dfb_bindings) {
        auto [it, inserted] =
            accessor_bindings.try_emplace(dfb_binding.accessor_name, AccessorBindingInfo{dfb_binding.dfb_spec_name});
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
    }

    // Data format metadata (optional param) MUST be specified for a DFB with a compute endpoint
    if (kernel.is_compute_kernel()) {
        for (const auto& binding : kernel.dfb_bindings) {
            const DataflowBufferSpec* dfb_spec = ctx.collected.dfb_by_name.at(binding.dfb_spec_name);
            TT_FATAL(
                dfb_spec->data_format_metadata.has_value(),
                "DFB '{}' is used by a compute kernel, but no data_format_metadata is specified",
                binding.dfb_spec_name);
        }
    }
}

namespace {

// Implicit sync is a Gen2-only, DM-only mechanism (ISR-based credit posting from NoC
// transaction completion). A DM kernel can opt out per-DFB by listing the DFB's name in
// config_2xx->disable_dfb_implicit_sync_for, or opt out of all the DFBs it binds at
// once via config_2xx->disable_dfb_implicit_sync_for_all. If config_2xx is not
// engaged, implicit sync stays at its default (on for every bound DFB). Either way the
// opt-out applies to the side(s) of the DFB this kernel binds (producer, consumer, or
// both for a self-loop).
//
// Cross-kernel rule (per DFB): on each side independently, all DM kernels must agree on the
// opt-out — either all disable it (by list or by _all), or none do. (Producer-side and
// consumer-side are checked separately; the underlying hardware mechanism is per-side, with
// one mask per side.)
void ValidateImplicitSyncAgreement(const CollectedSpecData& collected, tt::ARCH arch) {
    // Note: a single DFB can be bound by multiple producer KernelSpecs and multiple
    // consumer KernelSpecs — ops sometimes specialize the same kernel source by CTAs,
    // producing several KernelSpecs that share a DFB.
    auto check_side_agreement = [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints,
                                    const DFBSpecName& dfb_name,
                                    std::string_view side_label) {
        const KernelSpec* canonical = nullptr;
        bool canonical_disables = false;
        for (const auto& ep : endpoints) {
            if (!ep.kernel->is_data_movement_kernel()) {
                continue;
            }
            const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
            if (!is_gen2_arch(arch)) {
                // Gen1 device — can't physically participate in Gen2 implicit sync; abstains.
                continue;
            }
            const bool disables = DmKernelDisablesImplicitSync(dm_config, dfb_name);
            if (canonical == nullptr) {
                canonical = ep.kernel;
                canonical_disables = disables;
                continue;
            }
            TT_FATAL(
                disables == canonical_disables,
                "DFB '{}' has disagreeing implicit-sync opt-out state on the {} side",
                dfb_name,
                side_label);
        }
    };
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        check_side_agreement(endpoint_info.producers, dfb_name, "producer");
        check_side_agreement(endpoint_info.consumers, dfb_name, "consumer");
    }
}

// Validate local DFB endpoint placement and multi-binding consistency.
//
// The hardware invariant is local: a local DFB lives in shared SRAM on each node, so at every
// node where the DFB is instantiated, exactly one producer kernel instance and exactly one
// consumer kernel instance must run on that node. Metal 2.0 permits multiple PRODUCER
// KernelSpecs (and multiple CONSUMER KernelSpecs) per DFB, so we enforce that invariant directly
// as a per-node census, plus per-role uniformity of the binding-site config:
//   1./2. Placement: every node hosting the DFB runs exactly one producer instance and exactly
//         one consumer instance (the per-node census below). This subsumes both "no node has two
//         same-role instances" and "producer and consumer node coverage coincide".
//   3. All bindings on the same role have matching `access_pattern` (the DFB scheduler
//      config is shared per role).
//   4. All KernelSpecs on the same role have matching `num_threads` (the per-side
//      credit-tracking config is shared per role).
// Self-loop (a kernel that appears in both producers and consumers of a DFB) is currently
// restricted to the simple single-producer-single-consumer case.
void ValidateDFBEndpointPlacement(const DataflowBufferSpec& dfb, const CollectedSpecData& collected, tt::ARCH arch) {
    const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);

    // allow_instance_multi_binding (Gen1-only; rejected on Gen2 in ValidateDFBSpec) turns
    // the per-node DFB into a plain shared circular buffer, which has no per-role hardware config
    // to share — no processor mask, DFB scheduler, or credit machinery. Every per-role uniformity
    // requirement below exists solely to guarantee such a shared config, so none apply under the
    // flag: the role-uniformity checks are skipped and the per-node census relaxes its "exactly
    // one" counts to "at least one".
    const bool allow_multi = dfb.advanced_options.allow_instance_multi_binding;

    // (3) and (4): per-role uniformity of binding-site parameters, plus kernel kind.
    // Kind (compute vs DM) must agree because the DFB's hardware config carries a single
    // processor mask per role, and compute / DM masks live in disjoint bit ranges (bits
    // 0-7 vs 8-15 on Gen2; orthogonal RISC encodings on Gen1) — mismatched kinds cannot
    // share a mask. (All three are skipped under allow_multi — see above.)
    auto check_role_uniformity = [&](const auto& records, std::string_view role) {
        if (records.size() < 2) {
            return;
        }
        const auto first_pattern = records[0].binding->access_pattern;
        const auto first_threads = records[0].kernel->num_threads;
        const bool first_is_compute = records[0].kernel->is_compute_kernel();
        const auto& first_kernel = records[0].kernel->unique_id;
        for (size_t i = 1; i < records.size(); ++i) {
            TT_FATAL(
                records[i].binding->access_pattern == first_pattern,
                "DFB '{}' has multiple {} bindings with mismatched access_pattern (kernel '{}' vs kernel '{}')",
                dfb.unique_id,
                role,
                first_kernel,
                records[i].kernel->unique_id);
            TT_FATAL(
                records[i].kernel->num_threads == first_threads,
                "DFB '{}' has multiple {} KernelSpecs with mismatched num_threads (kernel '{}' = {} vs kernel '{}' "
                "= {})",
                dfb.unique_id,
                role,
                first_kernel,
                first_threads,
                records[i].kernel->unique_id,
                records[i].kernel->num_threads);
            TT_FATAL(
                records[i].kernel->is_compute_kernel() == first_is_compute,
                "DFB '{}' has multiple {} KernelSpecs mixing compute and data-movement kinds "
                "('{}' is a {} kernel; '{}' is a {} kernel). All KernelSpecs bound to the same "
                "DFB role must be of the same kind — the DFB's hardware config carries a single "
                "processor mask per role.",
                dfb.unique_id,
                role,
                first_kernel,
                first_is_compute ? "compute" : "data-movement",
                records[i].kernel->unique_id,
                first_is_compute ? "data-movement" : "compute");
        }
    };
    if (!allow_multi) {
        check_role_uniformity(endpoints.producers, "PRODUCER");
        check_role_uniformity(endpoints.consumers, "CONSUMER");
    }

    // (1)/(2) Placement — per-node census. A local DFB lives in shared SRAM on each node, so
    // every node it is instantiated on must run exactly one producer instance and exactly one
    // consumer instance. Tally instances per node directly from the bindings: this subsumes the
    // old within-role disjointness check (a node with >1 same-role instance) and the cross-role
    // coverage check (a node with one role but not the other), and reports the offending node in
    // node terms rather than WorkUnitSpec terms. It counts actual node occupancy, so overlapping
    // same-role placements are caught regardless of how the WorkUnitSpec bookkeeping produced
    // them. (Self-loops fall out naturally: a self-looping kernel tallies as one producer AND
    // one consumer on each of its nodes.)
    std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> producers_on_node;
    std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> consumers_on_node;
    auto tally_role = [&](const auto& records, auto& on_node) {
        for (const auto& rec : records) {
            for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(rec.kernel->unique_id))) {
                on_node[node].push_back(rec.kernel);
            }
        }
    };
    tally_role(endpoints.producers, producers_on_node);
    tally_role(endpoints.consumers, consumers_on_node);

    // Footprint = every node hosting any instance of either role. A std::set gives deterministic
    // iteration order, hence deterministic error messages.
    std::set<NodeCoord> footprint;
    for (const auto& [node, kernels] : producers_on_node) {
        footprint.insert(node);
    }
    for (const auto& [node, kernels] : consumers_on_node) {
        footprint.insert(node);
    }

    auto names_at = [](const std::unordered_map<NodeCoord, std::vector<const KernelSpec*>>& on_node,
                       const NodeCoord& node) -> std::string {
        auto it = on_node.find(node);
        if (it == on_node.end() || it->second.empty()) {
            return "none";
        }
        std::string names;
        for (const KernelSpec* k : it->second) {
            names += (names.empty() ? "'" : ", '") + k->unique_id.get() + "'";
        }
        return names;
    };

    // Per-node census. Under allow_multi (see top of function) the "exactly one" upper bound relaxes
    // to "at least one": a node may host multiple instances of a role, but must still host at
    // least one producer AND one consumer, or the FIFO is half-wired.
    for (const NodeCoord& node : footprint) {
        auto p_it = producers_on_node.find(node);
        auto c_it = consumers_on_node.find(node);
        const size_t num_producers = p_it == producers_on_node.end() ? 0 : p_it->second.size();
        const size_t num_consumers = c_it == consumers_on_node.end() ? 0 : c_it->second.size();
        const bool node_ok =
            allow_multi ? (num_producers >= 1 && num_consumers >= 1) : (num_producers == 1 && num_consumers == 1);
        if (node_ok) {
            continue;
        }
        std::string_view guidance;
        if (num_producers == 0) {
            guidance =
                "This node has a consumer but no producer — ensure a producer kernel covers it "
                "(via its WorkUnitSpec membership).";
        } else if (num_consumers == 0) {
            guidance =
                "This node has a producer but no consumer — ensure a consumer kernel covers it "
                "(via its WorkUnitSpec membership).";
        } else {
            guidance =
                "Multiple same-role kernel instances land on this node — their placements overlap; "
                "give each disjoint nodes.";
        }
        TT_FATAL(
            false,
            "Local DFB '{}' is malformed at node {}: {} producer instance(s) ({}) and {} consumer "
            "instance(s) ({}). A local DFB lives in shared SRAM on each node, so every node it is "
            "instantiated on must run exactly one producer and one consumer kernel instance. {}",
            dfb.unique_id,
            node.str(),
            num_producers,
            names_at(producers_on_node, node),
            num_consumers,
            names_at(consumers_on_node, node),
            guidance);
    }

    // Find a self-loop participant: a kernel bound to this DFB as both producer and consumer.
    // Stays nullptr if the DFB is not self-looped. Iterating producers in vector order keeps the
    // pick — and any resulting error message — deterministic across runs.
    const KernelSpec* self_loop_kernel = nullptr;
    for (const auto& p : endpoints.producers) {
        for (const auto& c : endpoints.consumers) {
            if (p.kernel == c.kernel) {
                self_loop_kernel = p.kernel;
                break;
            }
        }
        if (self_loop_kernel != nullptr) {
            break;
        }
    }

    if (self_loop_kernel != nullptr) {
        // A data-movement kernel may self-loop a DFB (bind it as both PRODUCER and CONSUMER) only
        // on Gen1 (WH/BH), where a DFB lowers to a plain circular buffer that a single DM RISC can
        // both fill and drain. On Gen2 the DFB's tile-counter credit machinery requires disjoint
        // producer/consumer RISCs, so a DM self-loop cannot be lowered. Catch it here (with a clear
        // message) rather than let it fall through to a confusing "producer_risc_mask and
        // consumer_risc_mask must not overlap" error in the DFB backend. (Compute self-loops are
        // always legal: they lower to the intra-Tensix packer->unpacker flow.)
        TT_FATAL(
            !(is_gen2_arch(arch) && self_loop_kernel->is_data_movement_kernel()),
            "DataflowBuffer '{}' is self-looped by data-movement kernel '{}' (bound as both PRODUCER "
            "and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 "
            "architectures. Consider using a scratchpad or LocalTensorAccessor instead.",
            dfb.unique_id,
            self_loop_kernel->unique_id);

        // Self-loop interplay with multi-binding: the producer set must equal the consumer set
        // as sets of KernelSpec*. This permits the natural pattern of multiple same-source
        // KernelSpecs each self-looping the DFB on their disjoint node ranges, while rejecting
        // the case where a self-looping kernel shares the DFB with an unrelated kernel (which
        // would make the producer/consumer mask and lowering semantics ambiguous).
        std::unordered_set<const KernelSpec*> producer_kernels;
        std::unordered_set<const KernelSpec*> consumer_kernels;
        for (const auto& p : endpoints.producers) {
            producer_kernels.insert(p.kernel);
        }
        for (const auto& c : endpoints.consumers) {
            consumer_kernels.insert(c.kernel);
        }
        TT_FATAL(
            producer_kernels == consumer_kernels,
            "DFB '{}' is self-looped (some kernel appears as both producer and consumer), but "
            "the set of producer KernelSpecs differs from the set of consumer KernelSpecs. "
            "When a DFB is self-looped, every same-side binding must come from a self-loop "
            "participant (i.e. a kernel that appears on both sides).",
            dfb.unique_id);
    }

    // (5) Gen1: same-role DM kernels must share config_1xx->processor. The DFB's hardware config
    // carries a single processor per role, and on Gen1 the processor is user-specified (compute
    // kernels are always placed on the same Tensix processor, and kinds agree by (3)).
    // (Skipped under allow_multi — see above.)
    auto check_role_processor = [&](const auto& records, std::string_view role) {
        if (records.size() < 2 || !records[0].kernel->is_data_movement_kernel()) {
            return;
        }
        const KernelSpec* first_kernel = records[0].kernel;
        const DataMovementProcessor first_processor =
            std::get<DataMovementHardwareConfig>(first_kernel->hw_config).config_1xx->processor;
        for (size_t i = 1; i < records.size(); ++i) {
            const DataMovementProcessor processor =
                std::get<DataMovementHardwareConfig>(records[i].kernel->hw_config).config_1xx->processor;
            TT_FATAL(
                processor == first_processor,
                "DFB '{}' has multiple {} KernelSpecs ('{}', '{}') with mismatched processor placement "
                "(RISCV_{} vs RISCV_{}). Multi-binding requires all same-role data-movement kernels to "
                "share DataMovement1XXConfig::processor.",
                dfb.unique_id,
                role,
                first_kernel->unique_id,
                records[i].kernel->unique_id,
                static_cast<int>(first_processor),
                static_cast<int>(processor));
        }
    };
    if (!allow_multi && is_gen1_arch(arch)) {
        check_role_processor(endpoints.producers, "PRODUCER");
        check_role_processor(endpoints.consumers, "CONSUMER");
    }
}

}  // namespace

void ValidateDFBEndpoints(const ValidationContext& ctx, tt::ARCH arch) {
    ValidateImplicitSyncAgreement(ctx.collected, arch);
    for (const auto& dfb : ctx.spec.dataflow_buffers) {
        ValidateDFBEndpointPlacement(dfb, ctx.collected, arch);
    }
}

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
