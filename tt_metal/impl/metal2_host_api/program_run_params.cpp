// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <unordered_set>

#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
// Helper Function Forward Declarations
// ============================================================================

void ValidateProgramRunParams(const Program& program, const ProgramRunParams& params);

// ============================================================================
// PUBLIC ENTRY POINTS: SetProgramRunParameters + GetProgramRunParamsView
// ============================================================================

void SetProgramRunParameters(Program& program, const ProgramRunParams& params) {
    // Validate parameters against the schema
    ValidateProgramRunParams(program, params);

    detail::ProgramImpl& program_impl = program.impl();

    // Process kernel runtime arguments
    for (const auto& kernel_params : params.kernel_run_params) {
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_params.kernel_spec_name);

        // Set per-node runtime args
        // set_runtime_args handles both first-time allocation and subsequent updates
        for (const auto& [node_coord, args] : kernel_params.runtime_args) {
            kernel->set_runtime_args(node_coord, args);
        }

        // Set common runtime args
        // TODO: Why on earth does SetCommonRuntimeArgs() only work the first time??
        if (!kernel_params.common_runtime_args.empty()) {
            if (kernel->common_runtime_args().empty()) {
                // First time: use the normal setter which allocates storage
                kernel->set_common_runtime_args(kernel_params.common_runtime_args);
            } else {
                // Subsequent calls: update in-place
                // (set_common_runtime_args fatals if called twice, so use direct access)
                RuntimeArgsData& crta = kernel->common_runtime_args_data();
                TT_FATAL(
                    crta.size() == kernel_params.common_runtime_args.size(),
                    "Kernel '{}' common runtime args count cannot change from {} to {}",
                    kernel_params.kernel_spec_name,
                    crta.size(),
                    kernel_params.common_runtime_args.size());
                std::memcpy(
                    crta.data(),
                    kernel_params.common_runtime_args.data(),
                    kernel_params.common_runtime_args.size() * sizeof(uint32_t));
            }
        }
    }

    // Process DFB runtime parameters
    // Currently, only validate that no unimplemented overrides are attempted
    for (const auto& dfb_params : params.dfb_run_params) {
        TT_FATAL(
            !dfb_params.entry_size.has_value() && !dfb_params.num_entries.has_value(),
            "DFB size overrides are not yet implemented.");
    }
}

ProgramRunParamsView GetProgramRunParamsView(Program& program) {
    (void)program;
    TT_FATAL(false, "GetProgramRunParamsView is not yet implemented.");
}

// ============================================================================
// IMPLEMENTATION: Validation
// ============================================================================

// Type alias for readability
using KernelRTASchema = detail::ProgramImpl::KernelRTASchema;

// Internal validation function - validates ProgramRunParams against the Program's schema
void ValidateProgramRunParams(const Program& program, const ProgramRunParams& params) {
    const detail::ProgramImpl& program_impl = program.impl();

    // Track which kernels we've seen parameters for
    std::unordered_set<KernelSpecName> kernels_with_params;

    for (const auto& kernel_params : params.kernel_run_params) {
        const KernelSpecName& kernel_name = kernel_params.kernel_spec_name;
        kernels_with_params.insert(kernel_name);

        // Check that the kernel exists
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_name);
        TT_FATAL(
            schema != nullptr,
            "Kernel '{}' has no RTA schema registered. Was the Program created from a ProgramSpec?",
            kernel_name);

        // Validate per-node runtime args counts
        // TODO: O(N^2) in nodes. Can optimize, or add a bypass validation option.
        //       Not too worried for now, as there's the power user API for speedy update.
        for (const auto& [node_coord, args] : kernel_params.runtime_args) {
            auto it = schema->num_runtime_args_per_node.find(node_coord);
            TT_FATAL(
                it != schema->num_runtime_args_per_node.end(),
                "Kernel {} is setting RTAs for node {}, but this is not a valid node for this kernel.",
                kernel_name,
                node_coord.str());
            TT_FATAL(
                args.size() == it->second,
                "Kernel '{}' node {} expects {} runtime args, but {} were provided",
                kernel_name,
                node_coord.str(),
                it->second,
                args.size());
        }

        // Validate common runtime args count
        TT_FATAL(
            kernel_params.common_runtime_args.size() == schema->num_common_runtime_args,
            "Kernel '{}' expects {} common runtime args, but {} were provided",
            kernel_name,
            schema->num_common_runtime_args,
            kernel_params.common_runtime_args.size());

        // Validate that all nodes in the schema have runtime args provided
        for (const auto& [node_coord, expected_count] : schema->num_runtime_args_per_node) {
            bool found = false;
            for (const auto& [provided_node, args] : kernel_params.runtime_args) {
                if (provided_node == node_coord) {
                    found = true;
                    break;
                }
            }
            TT_FATAL(
                found,
                "Kernel '{}' is missing runtime args for node {} (expected {} args)",
                kernel_name,
                node_coord.str(),
                expected_count);
        }
    }

    // Validate that all registered kernels have parameters
    std::vector<KernelSpecName> registered_names = program_impl.get_registered_kernel_names();
    for (const KernelSpecName& name : registered_names) {
        TT_FATAL(
            kernels_with_params.contains(name),
            "Kernel '{}' is registered in the Program but has no runtime parameters specified in ProgramRunParams",
            name);
    }
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
