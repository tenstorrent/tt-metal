// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"

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

}  // namespace tt::tt_metal::experimental
