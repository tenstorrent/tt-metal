// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>

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
const std::vector<AdvancedKernelRunArgs::NodeVarargs>& kernel_runtime_varargs(const ProgramRunArgs::KernelRunArgs& kp) {
    return kp.advanced_options.runtime_varargs;
}

const AdvancedKernelRunArgs::CommonVarargs& kernel_common_runtime_varargs(const ProgramRunArgs::KernelRunArgs& kp) {
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
void ValidateTensorArgs(const Program& program, std::span<const ProgramRunArgs::TensorArgument> tensor_args) {
    const detail::ProgramImpl& program_impl = program.impl();

    std::unordered_set<std::string> tensor_parameters_with_params;
    for (const auto& tensor_params : tensor_args) {
        auto [it, inserted] = tensor_parameters_with_params.insert(tensor_params.tensor_parameter_name);
        TT_FATAL(
            inserted,
            "Duplicate tensor_parameter_name '{}' in tensor_args. Each TensorParameter must appear at most once.",
            tensor_params.tensor_parameter_name);
        const TensorSpec* expected_spec = program_impl.get_tensor_parameter_layout(tensor_params.tensor_parameter_name);
        TT_FATAL(
            expected_spec != nullptr,
            "TensorArgument references unknown TensorParameter '{}'.",
            tensor_params.tensor_parameter_name);
        const TensorSpec& runtime_spec = tensor_params.tensor.get().tensor_spec();
        const bool dyn_shape =
            program_impl.get_tensor_parameter_dynamic_tensor_shape(tensor_params.tensor_parameter_name);
        const bool padded_only =
            program_impl.get_tensor_parameter_match_padded_shape_only(tensor_params.tensor_parameter_name);
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
                tensor_params.tensor_parameter_name);
            TT_FATAL(
                runtime_spec.logical_shape().rank() == expected_spec->logical_shape().rank(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose logical_shape rank ({}) differs from the "
                "declared rank ({}). dynamic_tensor_shape lets the per-dim shape values vary, but the rank must "
                "remain constant.",
                tensor_params.tensor_parameter_name,
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
                tensor_params.tensor_parameter_name);
            TT_FATAL(
                runtime_spec.padded_shape() == expected_spec->padded_shape(),
                "TensorArgument for binding '{}' supplied a MeshTensor whose padded_shape does not match the binding's "
                "declared padded_shape. match_padded_shape_only requires padded_shape to be preserved across binds; "
                "use dynamic_tensor_shape if you need padded_shape to vary as well.",
                tensor_params.tensor_parameter_name);
        } else {
            TT_FATAL(
                runtime_spec == *expected_spec,
                "TensorArgument for binding '{}' supplied a MeshTensor whose TensorSpec does not match the binding's "
                "declared spec. The binding declaration in ProgramSpec is the single source of truth for layout; "
                "the supplied tensor must conform to it.",
                tensor_params.tensor_parameter_name);
        }
    }
    for (const std::string& declared : program_impl.get_registered_tensor_parameter_names()) {
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
        const KernelSpecName& kernel_name = kernel_params.kernel_spec_name;
        auto [it, inserted] = kernels_with_params.insert(kernel_name);
        TT_FATAL(
            inserted,
            "Duplicate kernel_spec_name '{}' in ProgramRunArgs.kernel_run_args. Each kernel must appear exactly "
            "once.",
            kernel_name);

        // Check that the kernel exists
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name);
        TT_FATAL(
            schema != nullptr,
            "Kernel '{}' has no RTA schema registered. Was the Program created from a ProgramSpec?",
            kernel_name);

        // Nodes the kernel runs on — the required domain for named-RTA coverage.
        const std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_name);
        const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();

        // Validate vararg RTA counts per node
        std::unordered_set<NodeCoord> nodes_with_vararg_params;
        for (const auto& [node_coord, args] : kernel_runtime_varargs(kernel_params)) {
            auto [it_node, node_inserted] = nodes_with_vararg_params.insert(node_coord);
            TT_FATAL(
                node_inserted,
                "Duplicate node_coord {} in runtime_varargs for kernel '{}'.",
                node_coord.str(),
                kernel_name);

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

        // Validate named RTAs: every declared name set per-node, no extras, no duplicate node entries.
        const auto& named_rta_names = schema->runtime_arg_names;
        const std::unordered_set<std::string> named_rta_name_set(named_rta_names.begin(), named_rta_names.end());

        std::unordered_set<NodeCoord> nodes_with_named_params;
        for (const auto& node_params : kernel_params.runtime_arg_values) {
            auto [it_node, inserted_node] = nodes_with_named_params.insert(node_params.node);
            TT_FATAL(
                inserted_node,
                "Duplicate node_coord {} in runtime_arg_values for kernel '{}'.",
                node_params.node.str(),
                kernel_name);
            TT_FATAL(
                kernel_nodes.contains(node_params.node),
                "Kernel '{}' is setting runtime_arg_values for node {}, but the kernel does not run on that node.",
                kernel_name,
                node_params.node.str());
            TT_FATAL(
                node_params.args.size() == named_rta_names.size(),
                "Kernel '{}' node {} expects {} named RTAs, but {} were provided",
                kernel_name,
                node_params.node.str(),
                named_rta_names.size(),
                node_params.args.size());
            for (const auto& [name, _value] : node_params.args) {
                (void)_value;
                TT_FATAL(
                    named_rta_name_set.contains(name),
                    "Kernel '{}' node {} sets named RTA '{}' which is not declared in the schema.",
                    kernel_name,
                    node_params.node.str(),
                    name);
            }
        }
        if (!named_rta_names.empty()) {
            for (const auto& node : kernel_nodes) {
                TT_FATAL(
                    nodes_with_named_params.contains(node),
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
                kernel_params.common_runtime_arg_values.contains(name),
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
    // Kernels whose schema declares no named RTAs, no named CRTAs, no vararg RTAs, no vararg
    // CRTAs, AND no tensor bindings have nothing to supply per enqueue and do not need a
    // kernel_run_args entry. Tensor bindings count as "something to supply" because their
    // base address (and any dynamic accessor fields) are written into the kernel's CRTA buffer
    // by SetProgramRunArgs from the corresponding TensorArgument — and SetProgramRunArgs
    // only walks kernels present in params.kernel_run_args, so a kernel with tensor bindings
    // omitted from kernel_run_args would have its binding CRTAs left uninitialized.
    std::vector<KernelSpecName> registered_names = program_impl.get_registered_kernel_names();
    for (const KernelSpecName& name : registered_names) {
        if (kernels_with_params.contains(name)) {
            continue;
        }
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(name);
        if (schema == nullptr) {
            continue;
        }
        auto kernel = program_impl.get_kernel_by_spec_name(name);
        const bool has_tensor_bindings = kernel != nullptr && !kernel->tensor_binding_handles().empty();
        const bool has_anything_to_supply = !schema->runtime_arg_names.empty() ||
                                            !schema->common_runtime_arg_names.empty() ||
                                            !schema->num_runtime_varargs_per_node.empty() ||
                                            schema->num_common_runtime_varargs > 0 || has_tensor_bindings;
        TT_FATAL(
            !has_anything_to_supply,
            "Kernel '{}' is registered in the Program with a non-empty RTA/CRTA schema but has no "
            "runtime parameters specified in ProgramRunArgs (the schema is non-empty, or the "
            "kernel binds tensor parameters whose addresses are filled from kernel_run_args)",
            name);
    }

    // Validate DFB runtime parameters
    std::unordered_set<DFBSpecName> dfbs_with_params;
    for (const auto& dfb_params : params.dfb_run_overrides) {
        auto [it, inserted] = dfbs_with_params.insert(dfb_params.dfb_spec_name);
        TT_FATAL(
            inserted,
            "Duplicate dfb_spec_name '{}' in ProgramRunArgs.dfb_run_overrides. Each DFB must appear at most once.",
            dfb_params.dfb_spec_name);
        TT_FATAL(
            !dfb_params.entry_size.has_value() && !dfb_params.num_entries.has_value(),
            "DFB size overrides are not yet implemented.");
    }

    // Unlike kernels, DFBs don't require DFBRunOverrides.
    // (Borrowed-memory DFBs don't need a DFBRunOverrides entry either — they identify their backing
    // MeshTensor by name via DataflowBufferSpec::borrowed_from, and the tensor flows through
    // tensor_args.)

    // Validate tensor runtime parameters (delegated to shared helper).
    ValidateTensorArgs(program, params.tensor_args);
}

// Compute the CRTA values for a single tensor binding:
//   [base_address_word, optional runtime_field_words...]
// Total length = 1 + handle.num_runtime_field_crta_words.
//
// The base address always lives in CRTAs (per-enqueue, since the bound MeshTensor's
// address can change between binds). Additional runtime field words appear immediately
// after, when the TensorParameter opts into a dynamic accessor field. Currently the
// only such field is tensor_shape_in_pages, for sharded TensorParameters with
// dynamic_tensor_shape=true; in that case there are `rank` shape words.
std::vector<uint32_t> ComputeBindingCrtaValues(const TensorBindingHandle& handle, const MeshTensor& tensor) {
    std::vector<uint32_t> values;
    values.reserve(1u + handle.num_runtime_field_crta_words);

    const auto address = tensor.address();
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "TensorParameter '{}' base address {} exceeds uint32_t max",
        handle.tensor_parameter_name,
        address);
    values.push_back(static_cast<uint32_t>(address));

    if (handle.num_runtime_field_crta_words > 0) {
        // Currently the only runtime accessor field that lives in CRTAs is tensor_shape_in_pages
        // (sharded TensorParameter with dynamic_tensor_shape=true). Read it from the bound
        // MeshTensor's buffer.
        const tt::tt_metal::Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
        TT_FATAL(
            buffer != nullptr,
            "Tensor binding '{}' has runtime accessor field CRTA words but no backing Buffer to "
            "source them from.",
            handle.tensor_parameter_name);
        const auto& bds_opt = buffer->buffer_distribution_spec();
        TT_FATAL(
            bds_opt.has_value(),
            "Tensor binding '{}' has runtime accessor field CRTA words but its bound MeshTensor's "
            "buffer has no BufferDistributionSpec. dynamic_tensor_shape currently requires a sharded "
            "TensorParameter.",
            handle.tensor_parameter_name);
        const auto& tensor_shape = bds_opt->tensor_shape_in_pages();
        TT_FATAL(
            tensor_shape.rank() == handle.num_runtime_field_crta_words,
            "Tensor binding '{}' supplied a MeshTensor whose shape rank ({}) differs from the rank "
            "({}) reserved at ProgramSpec resolution time. Rank must remain constant across binds; "
            "only the per-dim shape values may vary.",
            handle.tensor_parameter_name,
            tensor_shape.rank(),
            handle.num_runtime_field_crta_words);
        for (size_t i = 0; i < tensor_shape.rank(); ++i) {
            values.push_back(static_cast<uint32_t>(tensor_shape[i]));
        }
    }
    return values;
}

// Attach the actual L1 Buffer to every borrowed-memory DFB, by resolving each DFB's named
// TensorParameter to the corresponding MeshTensor passed in tensor_args and extracting its
// MeshBuffer's reference buffer. Under the lockstep mesh allocation invariant, any one of the
// per-device buffers is a valid representative — same convention used by dynamic CB via
// `MeshTensor::mesh_buffer().get_reference_buffer()`.
//
// How this works:
//   - The ProgramSpec declares that the DFB borrows from a named TensorParameter.
//   - At each runtime attach point (this function) we extract the Buffer from the bound
//     MeshTensor, validate it, and hand it to the device-side DFB.
//   - Re-entry (a subsequent run-params call with a different MeshTensor) just re-attaches.
//
// Pre-condition: ValidateTensorArgs has enforced that every declared TensorParameter
// has a corresponding TensorArgument, so the lookup below cannot miss for any registered binding.
void AttachBorrowedDFBBuffers(
    detail::ProgramImpl& program_impl, std::span<const ProgramRunArgs::TensorArgument> tensor_args) {
    const auto& borrowed_bindings = program_impl.get_dfb_borrowed_bindings();
    if (borrowed_bindings.empty()) {
        return;
    }

    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(tensor_args.size());
    for (const auto& tensor_params : tensor_args) {
        tensor_by_param.emplace(tensor_params.tensor_parameter_name, &tensor_params.tensor.get());
    }

    for (const auto& [dfb_id, tp_name] : borrowed_bindings) {
        auto it = tensor_by_param.find(tp_name);
        TT_FATAL(
            it != tensor_by_param.end(),
            "Internal error: DFB id {} borrows from TensorParameter '{}' but no TensorArgument supplied it (validation "
            "should have caught this).",
            dfb_id,
            tp_name);
        const MeshTensor& tensor = *it->second;
        const tt::tt_metal::Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();

        // Attach-time legality checks (analogous to dynamic CB's
        // CircularBufferConfig::set_globally_allocated_address_and_total_size validations).
        //   - L1-residency: only L1 buffers may back a DFB.
        //   - Sizing: the DFB's total bytes must fit in the buffer's per-bank allocation.
        // ProgramSpec-time validation already enforced the TensorSpec-level analogs; these
        // refine the check now that a concrete Buffer is in hand.
        TT_FATAL(
            buffer->is_l1(),
            "Borrowed-memory DFB id {} (from TensorParameter '{}') requires an L1-resident backing memory.",
            dfb_id,
            tp_name);
        auto dfb_impl = program_impl.get_dataflow_buffer(dfb_id);
        const uint32_t dfb_total_bytes = dfb_impl->config.entry_size * dfb_impl->config.num_entries;
        TT_FATAL(
            dfb_total_bytes <= buffer->aligned_size_per_bank(),
            "Borrowed-memory DFB id {} (from TensorParameter '{}') has total size {} B, which exceeds the borrowed "
            "Buffer's per-bank size of {} B.",
            dfb_id,
            tp_name,
            dfb_total_bytes,
            buffer->aligned_size_per_bank());

        // Attach the address to the device-side DFB. Per-enqueue update_program_dispatch_commands
        // reads from the cache this populates, so no further dispatch-command invalidation is
        // needed for either first-call or re-entry.
        const auto address = buffer->address();
        TT_FATAL(
            address <= std::numeric_limits<uint32_t>::max(),
            "Borrowed Buffer base address {} for DFB id {} (TensorParameter '{}') exceeds uint32_t max.",
            address,
            dfb_id,
            tp_name);
        dfb_impl->set_borrowed_memory_base_addr(static_cast<uint32_t>(address));
    }
}

// ============================================================================
// PUBLIC ENTRY POINTS: SetProgramRunArgs + UpdateTensorArgs + GetProgramRunArgsView
// ============================================================================

void SetProgramRunArgs(Program& program, const ProgramRunArgs& params) {
    log_debug(tt::LogMetal, "Setting ProgramRunArgs");

    // Validate parameters against the schema
    ValidateProgramRunArgs(program, params);

    detail::ProgramImpl& program_impl = program.impl();

    // Build a tensor_parameter_name -> MeshTensor lookup from the user's TensorArgument entries.
    // Used below to fill each kernel's TensorBinding CRTA section: at minimum, the binding's
    // base address (always present); additionally, any runtime accessor field words the
    // TensorParameter opts into via dynamic_tensor_shape.
    // NOTE: We assume lockstep mesh allocation, so a device-independent set of values per binding.
    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(params.tensor_args.size());
    for (const auto& tensor_params : params.tensor_args) {
        tensor_by_param.emplace(tensor_params.tensor_parameter_name, &tensor_params.tensor.get());
    }

    // Process kernel runtime arguments.
    // Named RTAs/CRTAs and vararg RTAs/CRTAs share a single dispatch buffer each (RTA + CRTA).
    //
    // Layout:
    //   RTA per-node:  [named_rta_0 ... named_rta_N-1, vararg_0 ... vararg_M-1]
    //   CRTA:          [named_crta_0 ... named_crta_K-1, ta_addr_0 ... ta_addr_B-1, vararg_0 ... vararg_L-1]
    //
    // RTA layout has two sections: named RTAs and RTA varargs.
    // CRTA layout has three sections: named CRTAs, TensorBinding addresses, and CRTA varargs.
    //
    // TensorBinding address section is used by headergen to emit the `ta::` namespace tokens.
    // The device-side get_vararg / get_common_vararg helpers invisibly add the combined named-arg + binding
    // offset, so kernel code indexes varargs from 0.
    for (const auto& kernel_params : params.kernel_run_args) {
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_params.kernel_spec_name);
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_params.kernel_spec_name);
        TT_FATAL(schema != nullptr, "Kernel '{}' has no RTA schema registered.", kernel_params.kernel_spec_name);

        // Build a node -> named-RTA-values-map lookup for serialization.
        std::unordered_map<NodeCoord, const std::unordered_map<std::string, uint32_t>*> named_rtas_by_node;
        for (const auto& node_params : kernel_params.runtime_arg_values) {
            named_rtas_by_node[node_params.node] = &node_params.args;
        }

        // Build a node -> vararg-values-span lookup.
        std::unordered_map<NodeCoord, const std::vector<uint32_t>*> varargs_by_node;
        for (const auto& [node, args] : kernel_runtime_varargs(kernel_params)) {
            varargs_by_node[node] = &args;
        }

        // Iterate over every node the kernel runs on. We need to set RTAs on any node that
        // has either named RTAs or vararg RTAs. (Validation has already confirmed coverage.)
        const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();
        const bool kernel_has_named_rtas = !schema->runtime_arg_names.empty();
        for (const auto& node : kernel_nodes) {
            auto named_it = named_rtas_by_node.find(node);
            auto vararg_it = varargs_by_node.find(node);
            const bool has_named = named_it != named_rtas_by_node.end();
            const bool has_varargs = vararg_it != varargs_by_node.end();
            if (!kernel_has_named_rtas && !has_varargs) {
                continue;  // Nothing to set on this node.
            }

            std::vector<uint32_t> combined;
            combined.reserve(schema->runtime_arg_names.size() + (has_varargs ? vararg_it->second->size() : 0));
            if (kernel_has_named_rtas) {
                TT_FATAL(
                    has_named,
                    "Internal error: validation passed but named RTAs missing for kernel '{}' node {}.",
                    kernel_params.kernel_spec_name,
                    node.str());
                for (const auto& name : schema->runtime_arg_names) {
                    auto v_it = named_it->second->find(name);
                    TT_FATAL(
                        v_it != named_it->second->end(),
                        "Internal error: named RTA '{}' missing for kernel '{}' node {}.",
                        name,
                        kernel_params.kernel_spec_name,
                        node.str());
                    combined.push_back(v_it->second);
                }
            }
            if (has_varargs) {
                combined.insert(combined.end(), vararg_it->second->begin(), vararg_it->second->end());
            }
            kernel->set_runtime_args(node, combined);
        }

        // Assemble the kernel's per-enqueue CRTA buffer in three structurally-separate sections:
        //   1. User-named CRTAs, in schema order, sourced from common_runtime_arg_values.
        //   2. TensorBinding section, in binding-handle order, sourced from TensorArgument via the
        //      tensor_by_param lookup. Each binding occupies (1 + num_runtime_field_crta_words)
        //      words: [address, optional shape...]. The handle's addr_crta_offset lines up with
        //      the address slot position chosen here.
        //   3. Common runtime varargs, in caller-supplied order.
        const auto& binding_handles = kernel->tensor_binding_handles();
        std::size_t binding_section_words = 0;
        for (const auto& handle : binding_handles) {
            binding_section_words += 1u + handle.num_runtime_field_crta_words;
        }
        std::vector<uint32_t> combined_crtas;
        combined_crtas.reserve(
            schema->common_runtime_arg_names.size() + binding_section_words +
            kernel_common_runtime_varargs(kernel_params).size());
        for (const auto& name : schema->common_runtime_arg_names) {
            auto v_it = kernel_params.common_runtime_arg_values.find(name);
            TT_FATAL(
                v_it != kernel_params.common_runtime_arg_values.end(),
                "Internal error: named CRTA '{}' missing for kernel '{}'.",
                name,
                kernel_params.kernel_spec_name);
            combined_crtas.push_back(v_it->second);
        }
        for (const auto& handle : binding_handles) {
            auto t_it = tensor_by_param.find(handle.tensor_parameter_name);
            TT_FATAL(
                t_it != tensor_by_param.end(),
                "Internal error: tensor binding '{}' has no resolved MeshTensor (validation should have caught "
                "this).",
                handle.tensor_parameter_name);
            const auto values = ComputeBindingCrtaValues(handle, *t_it->second);
            combined_crtas.insert(combined_crtas.end(), values.begin(), values.end());
        }
        combined_crtas.insert(
            combined_crtas.end(),
            kernel_common_runtime_varargs(kernel_params).begin(),
            kernel_common_runtime_varargs(kernel_params).end());

        // Set common runtime args
        // TODO: Why on earth does SetCommonRuntimeArgs() only work the first time??
        if (!combined_crtas.empty()) {
            if (kernel->common_runtime_args().empty()) {
                // First time: use the normal setter which allocates storage
                kernel->set_common_runtime_args(combined_crtas);
            } else {
                // Subsequent calls: update in-place
                // (set_common_runtime_args fatals if called twice, so use direct access)
                RuntimeArgsData& crta = kernel->common_runtime_args_data();
                TT_FATAL(
                    crta.size() == combined_crtas.size(),
                    "Kernel '{}' common runtime args count cannot change from {} to {}",
                    kernel_params.kernel_spec_name,
                    crta.size(),
                    combined_crtas.size());
                std::memcpy(crta.data(), combined_crtas.data(), combined_crtas.size() * sizeof(uint32_t));
            }
        }
    }

    // Process DFB runtime parameters:
    //   - Borrowed-memory DFB backing L1 Buffer*
    //   - Later, add DFB size overrides (not yet implemented)
    AttachBorrowedDFBBuffers(program_impl, params.tensor_args);
}

void UpdateTensorArgs(Program& program, std::span<const ProgramRunArgs::TensorArgument> tensor_args) {
    log_debug(tt::LogMetal, "Updating tensor args (partial fast-path)");

    // Validate the TensorArgument list (shared with the full-path validator).
    ValidateTensorArgs(program, tensor_args);

    detail::ProgramImpl& program_impl = program.impl();

    // Build a tensor_parameter_name -> MeshTensor lookup.
    // As in SetProgramRunArgs, this assumes lockstep mesh allocation:
    // a single device-independent value set per binding.
    std::unordered_map<std::string, const MeshTensor*> tensor_by_param;
    tensor_by_param.reserve(tensor_args.size());
    for (const auto& tensor_params : tensor_args) {
        tensor_by_param.emplace(tensor_params.tensor_parameter_name, &tensor_params.tensor.get());
    }

    // For every kernel with tensor bindings, patch the binding slots in its CRTA buffer in
    // place. Each binding occupies (1 + num_runtime_field_crta_words) words starting at
    // handle.addr_crta_offset: the always-present address word, plus any runtime accessor
    // field words (shape, when dynamic_tensor_shape is set). The rest of the buffer (named
    // CRTAs, vararg CRTAs) is left untouched and retains the values installed by the most
    // recent SetProgramRunArgs call. The kernel's RTA buffer is also left untouched
    // (tensor binding state lives in CRTAs only).
    for (const KernelSpecName& kernel_name : program_impl.get_registered_kernel_names()) {
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
            const auto values = ComputeBindingCrtaValues(handle, *t_it->second);
            // addr_crta_offset is a byte offset; data() is uint32_t*.
            const uint32_t base_word = handle.addr_crta_offset / sizeof(uint32_t);
            for (size_t i = 0; i < values.size(); ++i) {
                crta.data()[base_word + i] = values[i];
            }
        }
    }

    // Process DFB runtime parameters to update borrowed-memory DFB backing L1 Buffer*s.
    AttachBorrowedDFBBuffers(program_impl, tensor_args);
}

ProgramRunArgsView& GetProgramRunArgsView(Program& program) {
    (void)program;
    TT_FATAL(false, "GetProgramRunArgsView is not yet implemented.");

    // This is the fast path, power user API.
    // Return type was changed to a reference to avoid copying the view.
    // With this API, we will need to either:
    //   - Create the view object on the first call and stash it in the Program object.
    //   - Or, create the view object upon Program construction.
}

}  // namespace tt::tt_metal::experimental
