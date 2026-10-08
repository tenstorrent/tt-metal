// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

// Tensor bindings: accessor names, binding sequences
void ValidateTensorBindings(const KernelSpec& kernel) {
    // A tensor binding is legal on both DM and compute kernels:
    //   - a DM kernel can use the binding token to construct a TensorAccessor or LocalTensorAccessor
    //   - a compute kernel can only use LocalTensorAccessor (NOC-free, local-L1 only)

    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : kernel.tensor_bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted, "Kernel '{}' has duplicate tensor accessor_name '{}'", kernel.unique_id, binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' tensor accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "tensor", binding.accessor_name);
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

void ValidateTensorParametersUsed(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;

    // Every declared TensorParameter must be referenced by some kernel binding or a DFB
    // borrowed_from. (Same usage requirement as DFBs; an unused tensor parameter is a user error.)
    // A borrowed-memory DFB uses its backing TensorParameter via DataflowBufferSpec::borrowed_from
    // (resolved by name at runtime) even when no kernel binds it, so that counts as a use. Only local
    // DFBs are walked: borrowed memory is a local-L1 feature (cross-node DFBs are runtime-unsupported).
    // Existence of the borrowed_from referent is validated in ValidateDFBSpec.
    std::unordered_set<TensorParamName> used_tensor_parameters;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.tensor_bindings) {
            used_tensor_parameters.insert(binding.tensor_parameter_name);
        }
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.borrowed_from.has_value()) {
            used_tensor_parameters.insert(*dfb.borrowed_from);
        }
    }
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        TT_FATAL(
            used_tensor_parameters.contains(tensor_parameter.unique_id),
            "TensorParameter '{}' is defined but not bound by any kernel",
            tensor_parameter.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
