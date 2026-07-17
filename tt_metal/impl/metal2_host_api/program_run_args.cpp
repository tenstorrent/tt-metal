// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental {

// ============================================================================
// Validation Helpers
// ============================================================================

// Type alias for readability
using KernelRTASchema = detail::ProgramImpl::KernelRTASchema;

// Helpers for vararg-value access (now living on AdvancedKernelRunArgs).
const Table<NodeCoord, std::vector<uint32_t>>& kernel_runtime_varargs(const ProgramRunArgs::KernelRunArgs& kp) {
    return kp.advanced_options.runtime_varargs;
}

const AdvancedKernelRunArgs::Varargs& kernel_common_runtime_varargs(const ProgramRunArgs::KernelRunArgs& kp) {
    return kp.advanced_options.common_runtime_varargs;
}

// Internal validation function - validates a TensorArgument list against the Program's TensorParameters.
// Shared by SetProgramRunArgs (full path) and UpdateTensorArgs (partial path).
//   - No duplicate tensor_parameter_name entries
//   - Every entry references a TensorParameter declared in the ProgramSpec
//   - The supplied MeshTensor's TensorSpec matches the binding's expected TensorSpec, with the
//     match relaxed according to the TensorParameter's loosening flags. The three cases form a
//     lattice from strictest to loosest (dynamic_tensor_shape strictly subsumes
//     match_padded_shape_only; when both are set, dynamic wins):
//       - Neither flag set (default): full TensorSpec equality.
//       - match_padded_shape_only=true (only): tensor_layout() must match exactly, and
//         padded_shape() must match exactly. logical_shape() may differ.
//       - dynamic_tensor_shape=true: tensor_layout() must match exactly, and the logical_shape
//         rank must match. Both logical_shape and padded_shape per-dim values may differ.
//     See the field doc comments in tensor_parameter.hpp for the full contracts.
//   - Every declared TensorParameter must be set
//   - Every declared TensorParameter must be set, UNLESS require_all is false (partial update),
//     in which case TensorParameters declared enqueue-loop invariant may be omitted (their
//     previously-bound MeshTensor is retained).
void ValidateTensorArgs(
    const Program& program,
    const Table<TensorParamName, ProgramRunArgs::TensorArgument>& tensor_args,
    bool require_all = true) {
    const detail::ProgramImpl& program_impl = program.impl();

    std::unordered_set<std::string> tensor_parameters_with_params;
    for (const auto& [param_name, tensor_arg] : tensor_args) {
        tensor_parameters_with_params.insert(param_name.get());
        const TensorSpec* expected_spec = program_impl.get_tensor_parameter_layout(param_name.get());
        TT_FATAL(expected_spec != nullptr, "TensorArgument references unknown TensorParameter '{}'.", param_name);
        const TensorSpec& runtime_spec = mesh_tensor_of(tensor_arg).tensor_spec();
        const bool dyn_shape = program_impl.get_tensor_parameter_dynamic_tensor_shape(param_name.get());
        const bool padded_only = program_impl.get_tensor_parameter_match_padded_shape_only(param_name.get());
        if (dyn_shape) {
            // dynamic_tensor_shape: tensor_layout must match exactly; logical_shape may differ in
            // per-dim values, but rank must still match. (Wins over match_padded_shape_only if both
            // are set: dynamic is strictly more permissive.)
            TT_FATAL(
                runtime_spec.tensor_layout() == expected_spec->tensor_layout(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose tensor_layout does not match the "
                "binding's "
                "declared layout. dynamic_tensor_shape loosens the match only along logical_shape; dtype, "
                "page_config, memory_config, and alignment must still match exactly.",
                param_name);
            TT_FATAL(
                runtime_spec.logical_shape().rank() == expected_spec->logical_shape().rank(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose logical_shape rank ({}) differs from the "
                "declared rank ({}). dynamic_tensor_shape lets the per-dim shape values vary, but the rank must "
                "remain constant.",
                param_name,
                runtime_spec.logical_shape().rank(),
                expected_spec->logical_shape().rank());
        } else if (padded_only) {
            // match_padded_shape_only: tensor_layout must match exactly, and padded_shape must
            // match exactly. logical_shape may differ provided it produces the same padded_shape.
            // Purely a host-side validation loosening: the accessor's CTAs/CRTAs are unchanged
            // (tensor_shape_in_pages is derived from padded_shape, which is fixed across binds).
            TT_FATAL(
                runtime_spec.tensor_layout() == expected_spec->tensor_layout(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose tensor_layout does not match the "
                "binding's "
                "declared layout. match_padded_shape_only loosens the match only along logical_shape (within the "
                "constraint that padded_shape is preserved); dtype, page_config, memory_config, and alignment must "
                "still match exactly.",
                param_name);
            TT_FATAL(
                runtime_spec.padded_shape() == expected_spec->padded_shape(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose padded_shape does not match the binding's "
                "declared padded_shape. match_padded_shape_only requires padded_shape to be preserved across binds; "
                "use dynamic_tensor_shape if you need padded_shape to vary as well.",
                param_name);
        } else {
            TT_FATAL(
                runtime_spec == *expected_spec,
                "TensorArgument for binding '{}' supplied a MeshTensor whose TensorSpec does not match the binding's "
                "declared spec. The binding declaration in ProgramSpec is the single source of truth for layout; "
                "the supplied tensor must conform to it.",
                param_name);
        }
    }
    for (const std::string& declared : program_impl.get_registered_tensor_parameter_names()) {
        if (!require_all && program_impl.get_tensor_parameter_enqueue_invariant(declared)) {
            // Partial update: an enqueue-invariant TensorParameter may be omitted; its previously
            // bound MeshTensor is retained.
            continue;
        }
        TT_FATAL(
            tensor_parameters_with_params.contains(declared),
            "TensorParameter '{}' is declared in the Program but has no TensorArgument entry.",
            declared);
    }
}

// Internal validation function - validates ProgramRunArgs against the Program's schema.
//
// Named RTAs/CRTAs and vararg RTAs/CRTAs are validated separately:
//   - Named args must have EVERY declared name set (and no extras). Named RTA values are
//     required for every node the kernel runs on.
//   - Vararg counts per-node must match the schema. A node with no vararg entry in the schema
//     expects zero varargs (but may still have named RTAs, if the schema declares them).
void ValidateProgramRunArgs(const Program& program, const ProgramRunArgs& params) {
    const detail::ProgramImpl& program_impl = program.impl();

    // Track which kernels we've seen parameters for
    std::unordered_set<KernelSpecName> kernels_with_params;

    // Validate kernel runtime parameters
    for (const auto& kernel_params : params.kernel_run_args) {
        const auto& kernel_name = kernel_params.kernel;
        // kernel_run_args is a Group (no structural key), so uniqueness must be validated here.
        auto [it, inserted] = kernels_with_params.insert(kernel_name);
        TT_FATAL(
            inserted,
            "Duplicate kernel '{}' in ProgramRunArgs.kernel_run_args. Each kernel must appear exactly once.",
            kernel_name);

        // Check that the kernel exists
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name.get());
        TT_FATAL(
            schema != nullptr,
            "Kernel '{}' has no RTA schema registered. Was the Program created from a ProgramSpec?",
            kernel_name);

        // Nodes the kernel runs on — the required domain for named-RTA coverage.
        const std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name.get());
        const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();

        // Validate vararg RTA counts per node
        std::unordered_set<NodeCoord> nodes_with_vararg_params;
        for (const auto& [node_coord, args] : kernel_runtime_varargs(kernel_params)) {
            nodes_with_vararg_params.insert(node_coord);

            TT_FATAL(
                kernel_nodes.contains(node_coord),
                "Kernel '{}' is setting runtime_varargs for node {}, but the kernel does not run on that node.",
                kernel_name,
                node_coord.str());

            auto it_schema = schema->num_runtime_varargs_per_node.find(node_coord);
            const size_t expected_varargs =
                (it_schema != schema->num_runtime_varargs_per_node.end()) ? it_schema->second : 0;
            TT_FATAL(
                args.size() == expected_varargs,
                "Kernel '{}' node {} expects {} vararg runtime args, but {} were provided",
                kernel_name,
                node_coord.str(),
                expected_varargs,
                args.size());
        }
        // Every node with a non-zero schema vararg entry must have values provided.
        // Zero-count entries should already be filtered out during schema expansion
        for (const auto& [node_coord, expected_count] : schema->num_runtime_varargs_per_node) {
            if (expected_count == 0) {
                continue;
            }
            TT_FATAL(
                nodes_with_vararg_params.contains(node_coord),
                "Kernel '{}' is missing vararg runtime args for node {} (expected {} args)",
                kernel_name,
                node_coord.str(),
                expected_count);
        }

        // Validate vararg CRTA count
        TT_FATAL(
            kernel_common_runtime_varargs(kernel_params).size() == schema->num_common_runtime_varargs,
            "Kernel '{}' expects {} vararg common runtime args, but {} were provided",
            kernel_name,
            schema->num_common_runtime_varargs,
            kernel_common_runtime_varargs(kernel_params).size());

        // Validate named RTAs: every declared name set per-node, no extras.
        const auto& named_rta_names = schema->runtime_arg_names;
        const std::unordered_set<std::string> named_rta_name_set(named_rta_names.begin(), named_rta_names.end());

        std::unordered_map<NodeCoord, size_t> named_rta_count_per_node;
        for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
            for (const auto& [node, _value] : per_node) {
                (void)_value;
                TT_FATAL(
                    kernel_nodes.contains(node),
                    "Kernel '{}' is setting runtime_arg_values for node {}, but the kernel does not run on that node.",
                    kernel_name,
                    node.str());
                named_rta_count_per_node[node]++;
            }
        }
        for (const auto& [node, provided] : named_rta_count_per_node) {
            TT_FATAL(
                provided == named_rta_names.size(),
                "Kernel '{}' node {} expects {} named RTAs, but {} were provided",
                kernel_name,
                node.str(),
                named_rta_names.size(),
                provided);
        }
        for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
            for (const auto& [node, _value] : per_node) {
                (void)_value;
                TT_FATAL(
                    named_rta_name_set.contains(name),
                    "Kernel '{}' node {} sets named RTA '{}' which is not declared in the schema.",
                    kernel_name,
                    node.str(),
                    name);
            }
        }
        if (!named_rta_names.empty()) {
            for (const auto& node : kernel_nodes) {
                TT_FATAL(
                    named_rta_count_per_node.contains(node),
                    "Kernel '{}' has named RTAs declared but no runtime_arg_values provided for node {}.",
                    kernel_name,
                    node.str());
            }
        }

        // Validate named CRTAs: every declared name supplied, no extras.
        // The TensorBinding address section lives in its own structurally-separate part of the
        // kernel's CRTA buffer and is filled from TensorArgument at enqueue.
        // (This is separate from the schema->common_runtime_arg_names.)
        const auto& named_crta_names = schema->common_runtime_arg_names;
        for (const auto& name : named_crta_names) {
            TT_FATAL(
                kernel_params.common_runtime_arg_values.get(name).has_value(),
                "Kernel '{}' is missing named CRTA '{}'.",
                kernel_name,
                name);
        }
        TT_FATAL(
            kernel_params.common_runtime_arg_values.size() == named_crta_names.size(),
            "Kernel '{}' expects {} user-named CRTAs, but {} were provided",
            kernel_name,
            named_crta_names.size(),
            kernel_params.common_runtime_arg_values.size());
    }

    // Validate that all registered kernels with a non-empty RTA/CRTA schema have parameters.
    // Kernels whose schema declares no named RTAs, no named CRTAs, no vararg RTAs, and no vararg
    // CRTAs have no scalar runtime arguments to supply per enqueue and do not need a kernel_run_args
    // entry. This holds even if the kernel binds tensors: a TensorBinding's base address (and any
    // dynamic accessor fields) is filled automatically by SetProgramRunArgs for every binding-bearing
    // kernel — including a binding-only kernel omitted from kernel_run_args, via its second pass —
    // so tensor bindings alone no longer force an (otherwise-empty) kernel_run_args entry.
    auto registered_names = program_impl.get_registered_kernel_names();
    for (const auto& name : registered_names) {
        if (kernels_with_params.contains(KernelSpecName{name})) {
            continue;
        }
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(name);
        if (schema == nullptr) {
            continue;
        }
        const bool has_anything_to_supply =
            !schema->runtime_arg_names.empty() || !schema->common_runtime_arg_names.empty() ||
            !schema->num_runtime_varargs_per_node.empty() || schema->num_common_runtime_varargs > 0;
        TT_FATAL(
            !has_anything_to_supply,
            "Kernel '{}' is registered in the Program with a non-empty RTA/CRTA schema but has no "
            "runtime parameters specified in ProgramRunArgs.",
            name);
    }

    // Validate DFB runtime parameters
    std::unordered_set<DFBSpecName> dfbs_with_params;
    for (const auto& dfb_params : params.dfb_run_overrides) {
        const auto& dfb_spec_name = dfb_params.dfb;
        // dfb_run_overrides is a Group (no structural key), so uniqueness must be validated here.
        auto [it, inserted] = dfbs_with_params.insert(dfb_spec_name);
        TT_FATAL(
            inserted,
            "Duplicate DFB '{}' in ProgramRunArgs.dfb_run_overrides. Each DFB must appear at most once.",
            dfb_spec_name);
        if (dfb_params.entry_size.has_value()) {
            TT_FATAL(
                dfb_params.entry_size.value() > 0,
                "dfb_run_overrides entry for DFB '{}' has entry_size override = 0. entry_size must be set to a "
                "non-zero value.",
                dfb_spec_name);
        }
        if (dfb_params.num_entries.has_value()) {
            TT_FATAL(
                dfb_params.num_entries.value() > 0,
                "dfb_run_overrides entry for DFB '{}' has num_entries override = 0. num_entries must be set to a "
                "non-zero value.",
                dfb_spec_name);
        }
    }

    // Unlike kernels, DFBs don't require DFBRunOverrides.
    // (Borrowed-memory DFBs don't need a DFBRunOverrides entry either — they identify their backing
    // MeshTensor by name via DataflowBufferSpec::borrowed_from, and the tensor flows through
    // tensor_args.)

    // Validate tensor runtime parameters (delegated to shared helper).
    ValidateTensorArgs(program, params.tensor_args);
}

// Emit the CRTA words for a single tensor binding, in order:
//   [base_address_word, optional runtime_field_words...]
// invoking emit(uint32_t) once per word. Total = 1 + handle.num_runtime_field_crta_words.
//
// The base address always lives in CRTAs (per-enqueue, since the bound MeshTensor's
// address can change between binds). Additional runtime field words appear immediately
// after, when the TensorParameter opts into a dynamic accessor field. Two kinds exist,
// mutually exclusive per binding (discriminated by handle.runtime_field_is_page_size):
//   - tensor_shape_in_pages, for sharded TensorParameters with dynamic_tensor_shape=true
//     (`rank` shape words); or
//   - the aligned page size, for interleaved row-major TensorParameters with
//     dynamic_tensor_shape=true (one word).
//
// Allocation-free by design: this runs once per binding on every enqueue, so callers
// emit straight into their destination (push_back onto the assembled CRTA vector on the
// full path, or write into the kernel's existing CRTA buffer slot on the partial paths)
// rather than through a per-binding temporary vector.
//
// `emit` is taken by const-ref (not a forwarding reference): it is invoked multiple times
// here, so it must not be forwarded/moved-from.
template <typename Emit>
void EmitBindingCrtaValues(const TensorBindingHandle& handle, const MeshTensor& tensor, const Emit& emit) {
    const auto address = tensor.address();
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "Tensor argument for TensorParameter '{}' base address {} exceeds uint32_t max",
        handle.tensor_parameter_name,
        address);
    emit(static_cast<uint32_t>(address));

    if (handle.num_runtime_field_crta_words == 0) {
        return;
    }

    // Both runtime-field kinds source their values from the bound MeshTensor's reference buffer.
    const tt::tt_metal::Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
    TT_FATAL(
        buffer != nullptr,
        "Tensor argument for TensorParameter '{}' has runtime accessor field CRTA words but no backing Buffer to "
        "source them from.",
        handle.tensor_parameter_name);

    if (handle.runtime_field_is_page_size) {
        // Page-size runtime field (interleaved row-major): emit the buffer's aligned page size, re-derived
        // each dispatch so it tracks a width-varying tensor across program-cache hits. Exactly one
        // word by construction (ResolveTensorParameterStaticCTAs reserves a single slot). No
        // BufferDistributionSpec is involved -- interleaved tensors have none, which is exactly why
        // the field-kind discriminator exists (the shape path below would FATAL on a missing BDS).
        emit(static_cast<uint32_t>(buffer->aligned_page_size()));
        return;
    }

    // dynamic_tensor_shape (sharded): the runtime tensor's shape-in-pages, one word per dim.
    const auto& bds_opt = buffer->buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "Tensor argument for TensorParameter '{}' has no BufferDistributionSpec.",
        handle.tensor_parameter_name);
    const auto& tensor_shape = bds_opt->tensor_shape_in_pages();
    TT_FATAL(
        tensor_shape.rank() == handle.num_runtime_field_crta_words,
        "Tensor argument for TensorParameter '{}' supplied a MeshTensor whose shape rank ({}) differs from the rank "
        "({}) reserved at ProgramSpec resolution time. Rank must remain constant across binds; "
        "only the per-dim shape values may vary.",
        handle.tensor_parameter_name,
        tensor_shape.rank(),
        handle.num_runtime_field_crta_words);
    for (size_t i = 0; i < tensor_shape.rank(); ++i) {
        emit(static_cast<uint32_t>(tensor_shape[i]));
    }
}

uint32_t ValidateBorrowedDFBBackingBuffer(
    uint32_t dfb_id,
    const std::string& tp_name,
    uint32_t entry_size,
    uint32_t num_entries,
    const tt::tt_metal::Buffer& buffer) {
    TT_FATAL(
        buffer.is_l1(),
        "Borrowed-memory DFB id {} (from TensorParameter '{}') requires an L1-resident backing memory.",
        dfb_id,
        tp_name);
    const uint32_t dfb_total_bytes = dfb::detail::checked_total_size(
        entry_size, num_entries, fmt::format("Borrowed-memory DFB {} from TensorParameter '{}'", dfb_id, tp_name));
    TT_FATAL(
        dfb_total_bytes <= buffer.aligned_size_per_bank(),
        "Borrowed-memory DFB id {} (from TensorParameter '{}') has total size {} B, which exceeds the borrowed "
        "Buffer's per-bank size of {} B.",
        dfb_id,
        tp_name,
        dfb_total_bytes,
        buffer.aligned_size_per_bank());

    const auto address = buffer.address();
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "Borrowed Buffer base address {} for DFB id {} (TensorParameter '{}') exceeds uint32_t max.",
        address,
        dfb_id,
        tp_name);
    return static_cast<uint32_t>(address);
}

struct ValidatedBorrowedAttachment {
    std::shared_ptr<dfb::detail::DataflowBufferImpl> dfb;
    uint32_t address;
};

std::vector<ValidatedBorrowedAttachment> PrepareBorrowedDFBAttachments(
    detail::ProgramImpl& program_impl,
    const std::unordered_map<std::string, const MeshTensor*>& tensor_by_param,
    const std::vector<detail::ProgramImpl::DfbSizeOverride>& overrides,
    bool require_all = true) {
    std::unordered_map<uint32_t, const detail::ProgramImpl::DfbSizeOverride*> override_by_id;
    override_by_id.reserve(overrides.size());
    for (const auto& override : overrides) {
        auto [it, inserted] = override_by_id.emplace(override.dfb_id, &override);
        TT_FATAL(inserted, "DFB {} has more than one size override in the same batch.", override.dfb_id);
    }

    std::vector<ValidatedBorrowedAttachment> attachments;
    attachments.reserve(program_impl.get_dfb_borrowed_bindings().size());

    for (const auto& [dfb_id, tp_name] : program_impl.get_dfb_borrowed_bindings()) {
        auto tensor = tensor_by_param.find(tp_name);
        if (tensor == tensor_by_param.end()) {
            TT_FATAL(
                !require_all && !override_by_id.contains(dfb_id),
                "Internal error: borrowed-memory DFB {} was resized or required without supplying TensorParameter "
                "'{}'.",
                dfb_id,
                tp_name);
            continue;
        }

        auto dfb_impl = program_impl.get_dataflow_buffer(dfb_id);
        TT_FATAL(
            dfb_impl->borrows_memory(),
            "Internal error: borrowed-memory binding for DFB {} refers to a DFB that does not borrow memory.",
            dfb_id);
        const auto override = override_by_id.find(dfb_id);
        const uint32_t entry_size = override == override_by_id.end()
                                        ? dfb_impl->config.entry_size
                                        : override->second->entry_size.value_or(dfb_impl->config.entry_size);
        const uint32_t num_entries = override == override_by_id.end()
                                         ? dfb_impl->config.num_entries
                                         : override->second->num_entries.value_or(dfb_impl->config.num_entries);
        const tt::tt_metal::Buffer* buffer = tensor->second->mesh_buffer().get_reference_buffer();
        const uint32_t address = ValidateBorrowedDFBBackingBuffer(dfb_id, tp_name, entry_size, num_entries, *buffer);
        attachments.push_back({std::move(dfb_impl), address});
    }
    return attachments;
}

void CommitBorrowedDFBAttachments(const std::vector<ValidatedBorrowedAttachment>& attachments) {
    for (const auto& attachment : attachments) {
        attachment.dfb->set_borrowed_memory_base_addr(attachment.address);
    }
}

// ============================================================================
// PUBLIC ENTRY POINTS: SetProgramRunArgs + UpdateTensorArgs
// ============================================================================

void SetProgramRunArgs(Program& program, const ProgramRunArgs& params, bool skip_validation) {
    log_debug(tt::LogMetal, "Setting ProgramRunArgs");

    // Validate parameters against the schema (can be skipped for trusted inputs, e.g. cached-program
    // re-enqueue inner loops where the args have already been validated once).
    if (!skip_validation) {
        ValidateProgramRunArgs(program, params);
    }

    detail::ProgramImpl& program_impl = program.impl();

    // Build a tensor_parameter_name -> MeshTensor lookup from the user's TensorArgument entries.
    // Used below to fill each kernel's TensorBinding CRTA section: at minimum, the binding's
    // base address (always present); additionally, any runtime accessor field words the
    // TensorParameter opts into via dynamic_tensor_shape.
    // NOTE: We assume lockstep mesh allocation, so a device-independent set of values per binding.
    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(params.tensor_args.size());
    for (const auto& [param_name, tensor_arg] : params.tensor_args) {
        tensor_by_param.emplace(param_name.get(), &mesh_tensor_of(tensor_arg));
    }

    // Append a kernel's TensorBinding CRTA section to `out`, in binding-handle order. Each binding
    // contributes (1 + num_runtime_field_crta_words) words: [base address, optional runtime accessor
    // fields...], sourced from the bound MeshTensor via tensor_by_param. Shared by the main
    // kernel_run_args loop below and the binding-only second pass after it.
    auto append_binding_crtas = [&](const auto& binding_handles, std::vector<uint32_t>& out) {
        for (const auto& handle : binding_handles) {
            auto t_it = tensor_by_param.find(handle.tensor_parameter_name);
            TT_FATAL(
                t_it != tensor_by_param.end(),
                "Internal error: tensor binding '{}' has no resolved MeshTensor (validation should have caught "
                "this).",
                handle.tensor_parameter_name);
            EmitBindingCrtaValues(handle, *t_it->second, [&out](uint32_t w) { out.push_back(w); });
        }
    };

    // Append a kernel's scratchpad CRTA section to `out`, in binding order: one word per scratchpad
    // binding, holding the scratchpad's allocated L1 base address. The address is 0 here on the first
    // SetProgramRunArgs (the scratchpad is allocated later, at program-compile time, and the slot is
    // then patched in place — see ProgramImpl::allocate_scratchpads); on any later re-assembly the
    // handle already carries the allocated address, so it is filled directly. The section is always
    // present (sized by the kernel's scratchpad bindings), so the buffer's word count is stable across
    // re-set calls (install_crtas asserts that).
    auto append_scratchpad_crtas = [](const auto& scratchpad_handles, std::vector<uint32_t>& out) {
        for (const auto& handle : scratchpad_handles) {
            out.push_back(handle.allocated_address);
        }
    };

    // Install a kernel's assembled CRTA buffer. set_common_runtime_args allocates storage but fatals
    // if called twice, so it can only be used the first time; on subsequent SetProgramRunArgs calls
    // (e.g. re-enqueue with new args) the already-allocated buffer is patched in place. Shared by the
    // main kernel_run_args loop and the binding-only second pass so both handle the re-set case.
    auto install_crtas = [](const std::shared_ptr<Kernel>& kernel,
                            const std::vector<uint32_t>& combined_crtas,
                            std::string_view kernel_name) {
        if (combined_crtas.empty()) {
            return;
        }
        if (kernel->common_runtime_args().empty()) {
            kernel->set_common_runtime_args(combined_crtas);
        } else {
            RuntimeArgsData& crta = kernel->common_runtime_args_data();
            TT_FATAL(
                crta.size() == combined_crtas.size(),
                "Kernel '{}' common runtime args count cannot change from {} to {}",
                kernel_name,
                crta.size(),
                combined_crtas.size());
            std::memcpy(crta.data(), combined_crtas.data(), combined_crtas.size() * sizeof(uint32_t));
        }
    };

    // Process kernel runtime arguments.
    // Named RTAs/CRTAs and vararg RTAs/CRTAs share a single dispatch buffer each (RTA + CRTA).
    //
    // Layout:
    //   RTA per-node:  [named_rta_0 ... named_rta_N-1, vararg_0 ... vararg_M-1]
    //   CRTA:          [named_crta_0 ... named_crta_K-1, ta_addr_0 ... ta_addr_B-1, scratch_addr_0 ...
    //   scratch_addr_S-1,
    //                   vararg_0 ... vararg_L-1]
    //
    // RTA layout has two sections: named RTAs and RTA varargs.
    // CRTA layout has four sections: named CRTAs, TensorBinding addresses, Scratchpad addresses, and CRTA varargs.
    //
    // TensorBinding address section is used by headergen to emit the `tensor::` namespace tokens.
    // The device-side get_vararg / get_common_vararg helpers invisibly add the combined named-arg + binding
    // offset, so kernel code indexes varargs from 0.
    for (const auto& kernel_params : params.kernel_run_args) {
        const auto& kernel_name = kernel_params.kernel;
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name.get());
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name.get());
        TT_FATAL(schema != nullptr, "Kernel '{}' has no RTA schema registered.", kernel_name);

        const bool kernel_has_named_rtas = !schema->runtime_arg_names.empty();
        const size_t num_named_rtas = schema->runtime_arg_names.size();
        const auto& slot_of = schema->runtime_arg_name_to_slot;

        // RTA layout per node: [named_rta_0 ... named_rta_{N-1}, vararg_0 ... vararg_{M-1}].
        // Named RTAs are scattered to their declaration slot via the schema's name->slot index
        // (one hash lookup per supplied arg, O(N)/node — matching how UpdateProgramRunArgs
        // serializes); varargs are positional and follow the named section.
        if (!kernel->cores_with_runtime_args().empty()) {
            // ---- Fast path (subsequent call): the per-node RTA buffers already exist, sized and
            // allocated by a prior call. Patch them in place, iterating ONLY the supplied args —
            // exactly as UpdateProgramRunArgs does (and as ValidateProgramRunArgs iterates). This
            // deliberately avoids the first-call machinery below: no per-call node->args lookup
            // maps and no walk over the kernel's full logical-core set. That bookkeeping exists
            // only to *join* a node's named + vararg sections into one combined buffer for the
            // first allocation; once the buffer exists there is nothing to join or size, so the
            // named and vararg sections are patched independently and the join is pure overhead.
            // (This is the per-node fixed cost that made Set ~5us/call slower than the otherwise
            // identical Update path, flat in N.)
            for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
                const auto s = slot_of.find(name);
                for (const auto& [node, value] : per_node) {
                    TT_FATAL(
                        s != slot_of.end(),
                        "Internal error: named RTA '{}' not in schema for kernel '{}' node {}.",
                        name,
                        kernel_name,
                        node.str());
                    RuntimeArgsData& rta = kernel->runtime_args_data(node);
                    TT_FATAL(
                        rta.data() != nullptr,
                        "SetProgramRunArgs fast path: kernel '{}' node {} has no allocated RTA buffer though the "
                        "kernel reports prior runtime args. Internal invariant violation.",
                        kernel_name,
                        node.str());
                    rta.data()[s->second] = value;
                }
            }
            for (const auto& [node, vals] : kernel_runtime_varargs(kernel_params)) {
                if (vals.empty()) {
                    continue;
                }
                RuntimeArgsData& rta = kernel->runtime_args_data(node);
                TT_FATAL(
                    rta.data() != nullptr,
                    "SetProgramRunArgs fast path: kernel '{}' node {} has varargs but no allocated RTA buffer.",
                    kernel_name,
                    node.str());
                uint32_t* vdst = rta.data() + num_named_rtas;
                for (const uint32_t v : vals) {
                    *vdst++ = v;
                }
            }
        } else {
            // ---- First call: no RTA buffers exist yet. Each node's buffer must be allocated by
            // set_runtime_args, which can be called only once per node and sizes the buffer to the
            // node's combined named+vararg width — so we must join a node's named and vararg values
            // before allocating. Build the node->values lookups and walk the kernel's logical cores
            // (the node coverage validation has confirmed) to assemble each combined buffer. This
            // runs once; every subsequent call takes the fast path above.
            std::unordered_map<NodeCoord, std::vector<std::pair<size_t, uint32_t>>> named_rtas_by_node;
            for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
                const auto s = slot_of.find(name);
                for (const auto& [node, value] : per_node) {
                    TT_FATAL(
                        s != slot_of.end(),
                        "Internal error: named RTA '{}' not in schema for kernel '{}' node {}.",
                        name,
                        kernel_name,
                        node.str());
                    named_rtas_by_node[node].emplace_back(s->second, value);
                }
            }
            std::unordered_map<NodeCoord, const std::vector<uint32_t>*> varargs_by_node;
            for (const auto& [node, args] : kernel_runtime_varargs(kernel_params)) {
                varargs_by_node[node] = &args;
            }

            const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();
            for (const auto& node : kernel_nodes) {
                auto named_it = named_rtas_by_node.find(node);
                auto vararg_it = varargs_by_node.find(node);
                const bool has_named = named_it != named_rtas_by_node.end();
                const bool has_varargs = vararg_it != varargs_by_node.end();
                if (!kernel_has_named_rtas && !has_varargs) {
                    continue;  // Nothing to set on this node.
                }
                const size_t num_varargs = has_varargs ? vararg_it->second->size() : 0;

                // Value-initialized to the exact combined width so the scatter can assign by slot.
                std::vector<uint32_t> combined(num_named_rtas + num_varargs, 0u);
                if (kernel_has_named_rtas && has_named) {
                    for (const auto& [slot, value] : named_it->second) {
                        combined[slot] = value;
                    }
                }
                if (has_varargs) {
                    std::copy(
                        vararg_it->second->begin(), vararg_it->second->end(), combined.begin() + num_named_rtas);
                }
                kernel->set_runtime_args(node, combined);
            }
        }

        // Assemble the kernel's per-enqueue CRTA buffer in four structurally-separate sections:
        //   1. User-named CRTAs, in schema order, sourced from common_runtime_arg_values.
        //   2. TensorBinding section, in binding-handle order, sourced from TensorArgument via the
        //      tensor_by_param lookup. Each binding occupies (1 + num_runtime_field_crta_words)
        //      words: [address, optional shape...]. The handle's addr_crta_offset lines up with
        //      the address slot position chosen here.
        //   3. Scratchpad section, in binding order, one word each (the allocated L1 base address,
        //      0 here until allocate_scratchpads patches it at program-compile time).
        //   4. Common runtime varargs, in caller-supplied order.
        const auto& binding_handles = kernel->tensor_binding_handles();
        std::size_t binding_section_words = 0;
        for (const auto& handle : binding_handles) {
            binding_section_words += 1u + handle.num_runtime_field_crta_words;
        }
        std::vector<uint32_t> combined_crtas;
        combined_crtas.reserve(
            schema->common_runtime_arg_names.size() + binding_section_words +
            kernel->scratchpad_binding_handles().size() + kernel_common_runtime_varargs(kernel_params).size());
        for (const auto& name : schema->common_runtime_arg_names) {
            auto v_it = kernel_params.common_runtime_arg_values.find(name);
            TT_FATAL(
                v_it != kernel_params.common_runtime_arg_values.end(),
                "Internal error: named CRTA '{}' missing for kernel '{}'.",
                name,
                kernel_name);
            combined_crtas.push_back(v_it->second);
        }
        append_binding_crtas(binding_handles, combined_crtas);
        append_scratchpad_crtas(kernel->scratchpad_binding_handles(), combined_crtas);
        combined_crtas.insert(
            combined_crtas.end(),
            kernel_common_runtime_varargs(kernel_params).begin(),
            kernel_common_runtime_varargs(kernel_params).end());

        install_crtas(kernel, combined_crtas, kernel_name.get());
    }

    // Second pass: kernels that bind tensors and/or a scratchpad but were omitted from kernel_run_args.
    // A kernel whose only per-enqueue state is TensorBinding addresses and/or a scratchpad (no named
    // RTAs/CRTAs and no varargs — a scratchpad's address is framework-supplied, not user-supplied) is
    // allowed to be absent from kernel_run_args (see ValidateProgramRunArgs). The loop above only visits
    // kernels present in kernel_run_args, so such a kernel's binding + scratchpad CRTA sections would
    // never be written. Fill them here. Kernels with neither need no CRTA buffer.
    std::unordered_set<std::string> kernels_in_run_args;
    kernels_in_run_args.reserve(params.kernel_run_args.size());
    for (const auto& kernel_params : params.kernel_run_args) {
        kernels_in_run_args.insert(kernel_params.kernel.get());
    }
    for (const auto& kernel_name : program_impl.get_registered_kernel_names()) {
        if (kernels_in_run_args.contains(kernel_name)) {
            continue;  // Already filled (and its CRTA buffer allocated) by the loop above.
        }
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name);
        const auto& binding_handles = kernel->tensor_binding_handles();
        const auto& scratchpad_handles = kernel->scratchpad_binding_handles();
        if (binding_handles.empty() && scratchpad_handles.empty()) {
            continue;  // No bindings => nothing to supply; no CRTA dispatch buffer needed.
        }
        // Binding-only kernel: its CRTA buffer is exactly the TensorBinding + scratchpad sections (it has
        // no named CRTAs or varargs, else validation would have required a kernel_run_args entry).
        // install_crtas allocates the buffer on the first SetProgramRunArgs call and patches it in place
        // on later ones (e.g. re-enqueue with a new tensor), mirroring the main loop.
        std::vector<uint32_t> combined_crtas;
        append_binding_crtas(binding_handles, combined_crtas);
        append_scratchpad_crtas(scratchpad_handles, combined_crtas);
        install_crtas(kernel, combined_crtas, kernel_name);
    }

    // Resolve and validate every borrowed attachment against the requested candidate sizes before
    // mutating either live DFB sizes or borrowed addresses. The two commit phases below cannot fail.
    std::vector<detail::ProgramImpl::DfbSizeOverride> size_overrides;
    size_overrides.reserve(params.dfb_run_overrides.size());
    for (const auto& dfb_params : params.dfb_run_overrides) {
        if (!dfb_params.entry_size.has_value() && !dfb_params.num_entries.has_value()) {
            continue;
        }
        size_overrides.push_back(
            {program_impl.get_dfb_handle(*dfb_params.dfb), dfb_params.entry_size, dfb_params.num_entries});
    }
    const auto borrowed_attachments = PrepareBorrowedDFBAttachments(program_impl, tensor_by_param, size_overrides);
    program_impl.apply_dfb_size_overrides(size_overrides);
    CommitBorrowedDFBAttachments(borrowed_attachments);
}

void UpdateTensorArgs(Program& program, const Table<TensorParamName, ProgramRunArgs::TensorArgument>& tensor_args) {
    log_debug(tt::LogMetal, "Updating tensor args (partial fast-path)");

    // Validate the TensorArgument list (shared with the full-path validator).
    ValidateTensorArgs(program, tensor_args);

    detail::ProgramImpl& program_impl = program.impl();

    // Build a tensor_parameter_name -> MeshTensor lookup.
    // As in SetProgramRunArgs, this assumes lockstep mesh allocation:
    // a single device-independent value set per binding.
    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(tensor_args.size());
    for (const auto& [param_name, tensor_arg] : tensor_args) {
        tensor_by_param.emplace(param_name.get(), &mesh_tensor_of(tensor_arg));
    }
    const auto borrowed_attachments = PrepareBorrowedDFBAttachments(program_impl, tensor_by_param, {});

    // For every kernel with tensor bindings, patch the binding slots in its CRTA buffer in
    // place. Each binding occupies (1 + num_runtime_field_crta_words) words starting at
    // handle.addr_crta_offset: the always-present address word, plus any runtime accessor
    // field words (shape, when dynamic_tensor_shape is set). The rest of the buffer (named
    // CRTAs, vararg CRTAs) is left untouched and retains the values installed by the most
    // recent SetProgramRunArgs call. The kernel's RTA buffer is also left untouched
    // (tensor binding state lives in CRTAs only).
    for (const auto& kernel_name : program_impl.get_registered_kernel_names()) {
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name);
        const auto& binding_handles = kernel->tensor_binding_handles();
        if (binding_handles.empty()) {
            continue;
        }

        // Pre-condition: SetProgramRunArgs must have been called previously to size and
        // populate this kernel's CRTA buffer. Without it, there is no buffer to patch into.
        TT_FATAL(
            !kernel->common_runtime_args().empty(),
            "UpdateTensorArgs called on Program before SetProgramRunArgs: kernel '{}' has tensor "
            "bindings but its CRTA buffer has not been allocated. Call SetProgramRunArgs at least "
            "once first.",
            kernel_name);

        RuntimeArgsData& crta = kernel->common_runtime_args_data();
        for (const auto& handle : binding_handles) {
            auto t_it = tensor_by_param.find(handle.tensor_parameter_name);
            TT_FATAL(
                t_it != tensor_by_param.end(),
                "Internal error: tensor binding '{}' has no resolved MeshTensor (validation should have "
                "caught this).",
                handle.tensor_parameter_name);
            // addr_crta_offset is a byte offset; data() is uint32_t*.
            uint32_t* dst = crta.data() + (handle.addr_crta_offset / sizeof(uint32_t));
            EmitBindingCrtaValues(handle, *t_it->second, [&dst](uint32_t w) { *dst++ = w; });
        }
    }

    CommitBorrowedDFBAttachments(borrowed_attachments);
}

// Union the args of `src` into `dst` for a single kernel (same kernel name). Used by
// MergeProgramRunArgs. Named RTAs (per node) and named CRTAs union by name; a given name may
// appear in at most one input (disjoint), else error unless skip_validation. Positional varargs
// cannot be name-merged: at most one input may supply them for a given node / for common varargs.
void MergeKernelRunArgsInto(
    ProgramRunArgs::KernelRunArgs& dst, const ProgramRunArgs::KernelRunArgs& src, bool skip_validation) {
    const auto& kernel_name = dst.kernel;

    // Named CRTAs.
    for (const auto& [name, value] : src.common_runtime_arg_values) {
        if (!skip_validation) {
            TT_FATAL(
                !dst.common_runtime_arg_values.get(name).has_value(),
                "MergeProgramRunArgs: kernel '{}' common runtime arg '{}' is specified in more than one "
                "ProgramRunArgs.",
                kernel_name,
                name);
        }
        dst.common_runtime_arg_values[name] = value;
    }

    // Per-node named RTAs (keyed by name, then node). A given (name, node) may appear in at most
    // one input; disjoint names/nodes union together.
    for (const auto& [name, src_per_node] : src.runtime_arg_values) {
        auto& dst_per_node = dst.runtime_arg_values[name];
        for (const auto& [node, value] : src_per_node) {
            if (!skip_validation) {
                TT_FATAL(
                    !dst_per_node.get(node).has_value(),
                    "MergeProgramRunArgs: kernel '{}' node {} runtime arg '{}' is specified in more than one "
                    "ProgramRunArgs.",
                    kernel_name,
                    node.str(),
                    name);
            }
            dst_per_node[node] = value;
        }
    }

    // Positional RTA varargs (per node): cannot be merged; at most one input may supply them.
    for (const auto& [node, vals] : src.advanced_options.runtime_varargs) {
        if (dst.advanced_options.runtime_varargs.get(node).has_value()) {
            if (!skip_validation) {
                TT_FATAL(
                    false,
                    "MergeProgramRunArgs: kernel '{}' node {} runtime varargs are specified in more than one "
                    "ProgramRunArgs (positional varargs cannot be merged).",
                    kernel_name,
                    node.str());
            }
            continue;  // skip_validation: keep dst's existing varargs.
        }
        dst.advanced_options.runtime_varargs[node] = vals;
    }

    // Positional common varargs: at most one input may be non-empty.
    if (!src.advanced_options.common_runtime_varargs.empty()) {
        if (!dst.advanced_options.common_runtime_varargs.empty()) {
            if (!skip_validation) {
                TT_FATAL(
                    false,
                    "MergeProgramRunArgs: kernel '{}' common runtime varargs are specified in more than one "
                    "ProgramRunArgs (positional varargs cannot be merged).",
                    kernel_name);
            }
        } else {
            dst.advanced_options.common_runtime_varargs = src.advanced_options.common_runtime_varargs;
        }
    }
}

// Validation for the partial fast-path UpdateProgramRunArgs.
//
// Differs from ValidateProgramRunArgs in exactly one dimension: named RTAs/CRTAs and tensor
// parameters declared enqueue-loop invariant MAY be omitted — the value installed by the prior
// SetProgramRunArgs is retained. All other ("regular") args must still be fully specified.
// Varargs are positional and can never be invariant, so they must always be supplied when the
// schema declares them.
void ValidateUpdateProgramRunArgs(const Program& program, const ProgramRunArgs& params) {
    const detail::ProgramImpl& program_impl = program.impl();

    std::unordered_set<KernelSpecName> kernels_with_params;
    for (const auto& kernel_params : params.kernel_run_args) {
        const auto& kernel_name = kernel_params.kernel;
        auto [it, inserted] = kernels_with_params.insert(kernel_name);
        TT_FATAL(
            inserted,
            "Duplicate kernel '{}' in ProgramRunArgs.kernel_run_args. Each kernel must appear exactly once.",
            kernel_name);

        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name.get());
        TT_FATAL(
            schema != nullptr,
            "Kernel '{}' has no RTA schema registered. Was the Program created from a ProgramSpec?",
            kernel_name);

        const std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name.get());
        const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();

        // --- Vararg RTA counts per node (varargs are never invariant; identical to the full path) ---
        std::unordered_set<NodeCoord> nodes_with_vararg_params;
        for (const auto& [node_coord, args] : kernel_runtime_varargs(kernel_params)) {
            nodes_with_vararg_params.insert(node_coord);
            TT_FATAL(
                kernel_nodes.contains(node_coord),
                "Kernel '{}' is setting runtime_varargs for node {}, but the kernel does not run on that node.",
                kernel_name,
                node_coord.str());
            auto it_schema = schema->num_runtime_varargs_per_node.find(node_coord);
            const size_t expected_varargs =
                (it_schema != schema->num_runtime_varargs_per_node.end()) ? it_schema->second : 0;
            TT_FATAL(
                args.size() == expected_varargs,
                "Kernel '{}' node {} expects {} vararg runtime args, but {} were provided",
                kernel_name,
                node_coord.str(),
                expected_varargs,
                args.size());
        }
        for (const auto& [node_coord, expected_count] : schema->num_runtime_varargs_per_node) {
            if (expected_count == 0) {
                continue;
            }
            TT_FATAL(
                nodes_with_vararg_params.contains(node_coord),
                "Kernel '{}' is missing vararg runtime args for node {} (expected {}). Varargs cannot be "
                "enqueue-invariant and must be supplied to UpdateProgramRunArgs.",
                kernel_name,
                node_coord.str(),
                expected_count);
        }

        // --- Common vararg count (never invariant) ---
        TT_FATAL(
            kernel_common_runtime_varargs(kernel_params).size() == schema->num_common_runtime_varargs,
            "Kernel '{}' expects {} vararg common runtime args, but {} were provided",
            kernel_name,
            schema->num_common_runtime_varargs,
            kernel_common_runtime_varargs(kernel_params).size());

        // --- Named RTAs: supplied names must be declared (no extras); every non-invariant name
        //     must be supplied for every node; invariant names may be omitted. ---
        const auto& named_rta_names = schema->runtime_arg_names;
        const std::unordered_set<std::string> named_rta_name_set(named_rta_names.begin(), named_rta_names.end());
        std::vector<std::string> regular_rta_names;
        for (const auto& n : named_rta_names) {
            if (!schema->enqueue_invariant_runtime_arg_names.contains(n)) {
                regular_rta_names.push_back(n);
            }
        }
        std::unordered_set<NodeCoord> nodes_with_named_params;
        for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
            for (const auto& [node, _value] : per_node) {
                (void)_value;
                nodes_with_named_params.insert(node);
                TT_FATAL(
                    kernel_nodes.contains(node),
                    "Kernel '{}' is setting runtime_arg_values for node {}, but the kernel does not run on that node.",
                    kernel_name,
                    node.str());
                TT_FATAL(
                    named_rta_name_set.contains(name),
                    "Kernel '{}' node {} sets named RTA '{}' which is not declared in the schema.",
                    kernel_name,
                    node.str(),
                    name);
            }
        }
        if (!regular_rta_names.empty()) {
            for (const auto& node : kernel_nodes) {
                TT_FATAL(
                    nodes_with_named_params.contains(node),
                    "Kernel '{}' has non-invariant named RTAs but no runtime_arg_values provided for node {}.",
                    kernel_name,
                    node.str());
            }
        }
        // Every non-invariant named RTA must be supplied for every node the kernel runs on.
        for (const auto& rname : regular_rta_names) {
            auto per_node = kernel_params.runtime_arg_values.get(rname);
            for (const auto& node : kernel_nodes) {
                TT_FATAL(
                    per_node.has_value() && per_node->get(node).has_value(),
                    "Kernel '{}' node {} is missing named RTA '{}', which is not declared enqueue-invariant and so "
                    "must be supplied to UpdateProgramRunArgs.",
                    kernel_name,
                    node.str(),
                    rname);
            }
        }

        // --- Named CRTAs: regular ones must be supplied; invariant may be omitted; no extras. ---
        const auto& named_crta_names = schema->common_runtime_arg_names;
        const std::unordered_set<std::string> named_crta_name_set(named_crta_names.begin(), named_crta_names.end());
        for (const auto& [name, _value] : kernel_params.common_runtime_arg_values) {
            (void)_value;
            TT_FATAL(
                named_crta_name_set.contains(name),
                "Kernel '{}' sets named CRTA '{}' which is not declared in the schema.",
                kernel_name,
                name);
        }
        for (const auto& name : named_crta_names) {
            if (schema->enqueue_invariant_common_runtime_arg_names.contains(name)) {
                continue;
            }
            TT_FATAL(
                kernel_params.common_runtime_arg_values.get(name).has_value(),
                "Kernel '{}' is missing named CRTA '{}', which is not declared enqueue-invariant and so must be "
                "supplied to UpdateProgramRunArgs.",
                kernel_name,
                name);
        }
    }

    // A registered kernel may be omitted from kernel_run_args only if it has nothing regular to
    // supply: no non-invariant named RTAs, no non-invariant named CRTAs, and no varargs.
    for (const auto& name : program_impl.get_registered_kernel_names()) {
        if (kernels_with_params.contains(KernelSpecName{name})) {
            continue;
        }
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(name);
        if (schema == nullptr) {
            continue;
        }
        bool has_regular_rta = false;
        for (const auto& n : schema->runtime_arg_names) {
            if (!schema->enqueue_invariant_runtime_arg_names.contains(n)) {
                has_regular_rta = true;
                break;
            }
        }
        bool has_regular_crta = false;
        for (const auto& n : schema->common_runtime_arg_names) {
            if (!schema->enqueue_invariant_common_runtime_arg_names.contains(n)) {
                has_regular_crta = true;
                break;
            }
        }
        const bool has_varargs =
            !schema->num_runtime_varargs_per_node.empty() || schema->num_common_runtime_varargs > 0;
        TT_FATAL(
            !(has_regular_rta || has_regular_crta || has_varargs),
            "Kernel '{}' has non-invariant runtime args (or varargs) but was omitted from UpdateProgramRunArgs. Only "
            "kernels whose every runtime arg is enqueue-invariant — and which have no varargs — may be omitted.",
            name);
    }

    // DFB run overrides: same checks as the full path (duplicates; non-zero size overrides), plus a
    // partial-path-only guard for borrowed-memory DFBs (below). A resized borrowed DFB must have its
    // backing TensorParameter supplied in this same update, so borrowed-attachment preparation re-runs the
    // per-bank fit check against the new size. The full Set path gets this for free (require_all=true);
    // on the partial path an invariant backing tensor may be omitted, which would otherwise let a grown
    // DFB overflow its borrowed buffer's per-bank region unchecked at execution.
    std::unordered_map<uint32_t, std::string> borrowed_backing;  // dfb_id -> backing TensorParameter name
    for (const auto& [dfb_id, tp_name] : program_impl.get_dfb_borrowed_bindings()) {
        borrowed_backing.emplace(dfb_id, tp_name);
    }
    std::unordered_set<DFBSpecName> dfbs_with_params;
    for (const auto& dfb_params : params.dfb_run_overrides) {
        auto [it, inserted] = dfbs_with_params.insert(dfb_params.dfb);
        TT_FATAL(
            inserted,
            "Duplicate DFB '{}' in ProgramRunArgs.dfb_run_overrides. Each DFB must appear at most once.",
            dfb_params.dfb);
        if (dfb_params.entry_size.has_value()) {
            TT_FATAL(
                dfb_params.entry_size.value() > 0,
                "dfb_run_overrides entry for DFB '{}' has entry_size override = 0. entry_size must be set to a "
                "non-zero value.",
                dfb_params.dfb);
        }
        if (dfb_params.num_entries.has_value()) {
            TT_FATAL(
                dfb_params.num_entries.value() > 0,
                "dfb_run_overrides entry for DFB '{}' has num_entries override = 0. num_entries must be set to a "
                "non-zero value.",
                dfb_params.dfb);
        }
        // A resized borrowed-memory DFB needs its backing tensor supplied here so the per-bank fit check
        // during borrowed-attachment preparation re-runs against the new size (see the block comment above).
        const bool resizes = dfb_params.entry_size.has_value() || dfb_params.num_entries.has_value();
        if (resizes) {
            if (auto b = borrowed_backing.find(program_impl.get_dfb_handle(dfb_params.dfb.get()));
                b != borrowed_backing.end()) {
                TT_FATAL(
                    params.tensor_args.contains(TensorParamName{b->second}),
                    "dfb_run_overrides resizes borrowed-memory DFB '{}', but its backing TensorParameter '{}' was "
                    "not supplied in this UpdateProgramRunArgs. Resizing a borrowed DFB on the partial-update fast "
                    "path requires supplying its backing tensor, so the per-bank fit check can re-validate against "
                    "the new size (the full SetProgramRunArgs path enforces this by requiring all tensors).",
                    dfb_params.dfb,
                    b->second);
            }
        }
    }

    // Tensor args: non-invariant TensorParameters must be supplied; invariant ones may be omitted.
    ValidateTensorArgs(program, params.tensor_args, /*require_all=*/false);
}

void UpdateProgramRunArgs(Program& program, const ProgramRunArgs& params, bool skip_validation) {
    log_debug(tt::LogMetal, "Updating ProgramRunArgs (partial fast-path)");

    if (!skip_validation) {
        ValidateUpdateProgramRunArgs(program, params);
    }

    detail::ProgramImpl& program_impl = program.impl();

    // Patch the supplied args in place. Omitted (enqueue-invariant) named args and tensor params
    // are left untouched, retaining the value installed by the most recent SetProgramRunArgs.
    // Positional varargs are never invariant, so a supplied vararg section refreshes wholesale.
    for (const auto& kernel_params : params.kernel_run_args) {
        const auto& kernel_name = kernel_params.kernel;
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name.get());
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name.get());
        TT_FATAL(schema != nullptr, "Kernel '{}' has no RTA schema registered.", kernel_name);

        const size_t num_named_rtas = schema->runtime_arg_names.size();

        // ---- Per-node RTA buffer: patch supplied named RTAs at their declaration-order slot ----
        if (!kernel_params.runtime_arg_values.empty()) {
            const auto& rta_index = schema->runtime_arg_name_to_slot;
            for (const auto& [name, per_node] : kernel_params.runtime_arg_values) {
                const auto it = rta_index.find(name);
                TT_FATAL(
                    it != rta_index.end(),
                    "Internal error: named RTA '{}' not in schema for kernel '{}'.",
                    name,
                    kernel_name);
                for (const auto& [node, value] : per_node) {
                    TT_FATAL(
                        kernel->cores_with_runtime_args().contains(node),
                        "UpdateProgramRunArgs: kernel '{}' has no runtime-arg buffer for node {}. Call "
                        "SetProgramRunArgs at least once before a partial update.",
                        kernel_name,
                        node.str());
                    RuntimeArgsData& rta = kernel->runtime_args_data(node);
                    rta.data()[it->second] = value;
                }
            }
        }

        // ---- Per-node RTA buffer: patch supplied varargs (positional, after the named section) ----
        for (const auto& [node, vals] : kernel_runtime_varargs(kernel_params)) {
            if (vals.empty()) {
                continue;
            }
            TT_FATAL(
                kernel->cores_with_runtime_args().contains(node),
                "UpdateProgramRunArgs: kernel '{}' has no runtime-arg buffer for node {}. Call SetProgramRunArgs at "
                "least once before a partial update.",
                kernel_name,
                node.str());
            RuntimeArgsData& rta = kernel->runtime_args_data(node);
            for (size_t j = 0; j < vals.size(); ++j) {
                rta.data()[num_named_rtas + j] = vals[j];
            }
        }

        // ---- CRTA buffer: patch supplied named CRTAs + supplied common varargs ----
        const auto& cvarargs = kernel_common_runtime_varargs(kernel_params);
        const bool touches_crta = !kernel_params.common_runtime_arg_values.empty() || !cvarargs.empty();
        if (touches_crta) {
            TT_FATAL(
                !kernel->common_runtime_args().empty(),
                "UpdateProgramRunArgs: kernel '{}' CRTA buffer not allocated. Call SetProgramRunArgs at least once "
                "before a partial update.",
                kernel_name);
            RuntimeArgsData& crta = kernel->common_runtime_args_data();

            if (!kernel_params.common_runtime_arg_values.empty()) {
                const auto& crta_index = schema->common_runtime_arg_name_to_slot;
                for (const auto& [name, value] : kernel_params.common_runtime_arg_values) {
                    const auto it = crta_index.find(name);
                    TT_FATAL(
                        it != crta_index.end(),
                        "Internal error: named CRTA '{}' not in schema for kernel '{}'.",
                        name,
                        kernel_name);
                    crta.data()[it->second] = value;
                }
            }
            if (!cvarargs.empty()) {
                // Common varargs live after the named CRTAs, the tensor-binding address section, and the
                // scratchpad address section.
                const auto& binding_handles = kernel->tensor_binding_handles();
                size_t binding_section_words = 0;
                for (const auto& h : binding_handles) {
                    binding_section_words += 1u + h.num_runtime_field_crta_words;
                }
                const size_t scratchpad_section_words = kernel->scratchpad_binding_handles().size();
                const size_t crta_vararg_base =
                    schema->common_runtime_arg_names.size() + binding_section_words + scratchpad_section_words;
                for (size_t j = 0; j < cvarargs.size(); ++j) {
                    crta.data()[crta_vararg_base + j] = cvarargs[j];
                }
            }
        }
    }

    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(params.tensor_args.size());
    for (const auto& [param_name, tensor_arg] : params.tensor_args) {
        tensor_by_param.emplace(param_name.get(), &mesh_tensor_of(tensor_arg));
    }

    // Resolve candidate DFB sizes and every supplied borrowed attachment before changing either live DFB sizes or
    // borrowed addresses.
    std::vector<detail::ProgramImpl::DfbSizeOverride> size_overrides;
    size_overrides.reserve(params.dfb_run_overrides.size());
    for (const auto& dfb_params : params.dfb_run_overrides) {
        if (!dfb_params.entry_size.has_value() && !dfb_params.num_entries.has_value()) {
            continue;
        }
        size_overrides.push_back(
            {program_impl.get_dfb_handle(*dfb_params.dfb), dfb_params.entry_size, dfb_params.num_entries});
    }
    const auto borrowed_attachments =
        PrepareBorrowedDFBAttachments(program_impl, tensor_by_param, size_overrides, /*require_all=*/false);

    // ---- Tensor bindings: patch CRTA address slots for SUPPLIED tensors only ----
    // (Invariant tensors omitted from params keep their previously-patched binding slots.)
    if (!params.tensor_args.empty()) {
        for (const auto& kernel_name : program_impl.get_registered_kernel_names()) {
            std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name);
            const auto& binding_handles = kernel->tensor_binding_handles();
            if (binding_handles.empty()) {
                continue;
            }
            bool any_supplied = false;
            for (const auto& handle : binding_handles) {
                if (tensor_by_param.contains(handle.tensor_parameter_name)) {
                    any_supplied = true;
                    break;
                }
            }
            if (!any_supplied) {
                continue;  // none of this kernel's bindings are being updated.
            }
            TT_FATAL(
                !kernel->common_runtime_args().empty(),
                "UpdateProgramRunArgs: kernel '{}' CRTA buffer not allocated; cannot patch tensor bindings. Call "
                "SetProgramRunArgs at least once before a partial update.",
                kernel_name);
            RuntimeArgsData& crta = kernel->common_runtime_args_data();
            for (const auto& handle : binding_handles) {
                auto t_it = tensor_by_param.find(handle.tensor_parameter_name);
                if (t_it == tensor_by_param.end()) {
                    continue;  // invariant tensor omitted → binding slot retained.
                }
                uint32_t* dst = crta.data() + (handle.addr_crta_offset / sizeof(uint32_t));
                EmitBindingCrtaValues(handle, *t_it->second, [&dst](uint32_t w) { *dst++ = w; });
            }
        }
    }
    // Keep the DFB size and borrowed-address commits adjacent. Tensor-binding CRTA patching above can reject an
    // UpdateProgramRunArgs call made before the initial SetProgramRunArgs; that failure must not leave a new DFB size
    // paired with the old borrowed address.
    program_impl.apply_dfb_size_overrides(size_overrides);
    CommitBorrowedDFBAttachments(borrowed_attachments);
}

ProgramRunArgs MergeProgramRunArgs(ProgramRunArgs base, std::span<const ProgramRunArgs> rest, bool skip_validation) {
    for (const ProgramRunArgs& other : rest) {
        // Tensor args: union by TensorParameter name (disjoint).
        for (const auto& [name, arg] : other.tensor_args) {
            if (!skip_validation) {
                TT_FATAL(
                    !base.tensor_args.contains(name),
                    "MergeProgramRunArgs: TensorParameter '{}' is specified in more than one ProgramRunArgs.",
                    name);
            }
            // Table::operator[] default-constructs its value; TensorArgument (a variant over a
            // reference_wrapper) is not default-constructible, so insert the pair directly. After
            // the disjoint check the key is absent, so insert takes effect (under skip_validation a
            // collision keeps base's existing entry).
            base.tensor_args.insert({name, arg});
        }

        // DFB run overrides: union by DFB name (disjoint).
        for (const auto& dfb : other.dfb_run_overrides) {
            if (!skip_validation) {
                for (const auto& existing : base.dfb_run_overrides) {
                    TT_FATAL(
                        existing.dfb != dfb.dfb,
                        "MergeProgramRunArgs: DFB '{}' overrides are specified in more than one ProgramRunArgs.",
                        dfb.dfb);
                }
            }
            base.dfb_run_overrides.push_back(dfb);
        }

        // Kernel run args: union per kernel (a kernel may appear in multiple inputs as long as
        // the actual args it carries are disjoint — e.g. one input's invariant args, another's
        // volatile args for the same kernel).
        for (const auto& kra : other.kernel_run_args) {
            ProgramRunArgs::KernelRunArgs* dst = nullptr;
            for (auto& existing : base.kernel_run_args) {
                if (existing.kernel == kra.kernel) {
                    dst = &existing;
                    break;
                }
            }
            if (dst == nullptr) {
                base.kernel_run_args.push_back(kra);
                continue;
            }
            MergeKernelRunArgsInto(*dst, kra, skip_validation);
        }
    }
    return base;
}

}  // namespace tt::tt_metal::experimental
