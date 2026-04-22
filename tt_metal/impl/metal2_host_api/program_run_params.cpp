// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
// Validation Helper
// ============================================================================

// Type alias for readability
using KernelRTASchema = detail::ProgramImpl::KernelRTASchema;

// Internal validation function - validates ProgramRunParams against the Program's schema.
//
// Named RTAs/CRTAs and vararg RTAs/CRTAs are validated separately:
//   - Named args must have EVERY declared name set (and no extras). Named RTA values are
//     required for every node the kernel runs on.
//   - Vararg counts per-node must match the schema. A node with no vararg entry in the schema
//     expects zero varargs (but may still have named RTAs, if the schema declares them).
void ValidateProgramRunParams(const Program& program, const ProgramRunParams& params) {
    const detail::ProgramImpl& program_impl = program.impl();

    // Track which kernels we've seen parameters for
    std::unordered_set<KernelSpecName> kernels_with_params;

    // Validate kernel runtime parameters
    for (const auto& kernel_params : params.kernel_run_params) {
        const KernelSpecName& kernel_name = kernel_params.kernel_spec_name;
        auto [it, inserted] = kernels_with_params.insert(kernel_name);
        TT_FATAL(
            inserted,
            "Duplicate kernel_spec_name '{}' in ProgramRunParams.kernel_run_params. Each kernel must appear exactly "
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
        for (const auto& [node_coord, args] : kernel_params.runtime_varargs) {
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
        // Every node with a schema vararg entry must have values provided.
        for (const auto& [node_coord, expected_count] : schema->num_runtime_varargs_per_node) {
            TT_FATAL(
                nodes_with_vararg_params.contains(node_coord),
                "Kernel '{}' is missing vararg runtime args for node {} (expected {} args)",
                kernel_name,
                node_coord.str(),
                expected_count);
        }

        // Validate vararg CRTA count
        TT_FATAL(
            kernel_params.common_runtime_varargs.size() == schema->num_common_runtime_varargs,
            "Kernel '{}' expects {} vararg common runtime args, but {} were provided",
            kernel_name,
            schema->num_common_runtime_varargs,
            kernel_params.common_runtime_varargs.size());

        // Validate named RTAs: every declared name set per-node, no extras, no duplicate node entries.
        const auto& named_rta_names = schema->named_runtime_args;
        const std::unordered_set<std::string> named_rta_name_set(named_rta_names.begin(), named_rta_names.end());

        std::unordered_set<NodeCoord> nodes_with_named_params;
        for (const auto& node_params : kernel_params.named_runtime_args) {
            auto [it_node, inserted_node] = nodes_with_named_params.insert(node_params.node);
            TT_FATAL(
                inserted_node,
                "Duplicate node_coord {} in named_runtime_args for kernel '{}'.",
                node_params.node.str(),
                kernel_name);
            TT_FATAL(
                kernel_nodes.contains(node_params.node),
                "Kernel '{}' is setting named_runtime_args for node {}, but the kernel does not run on that node.",
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
                    "Kernel '{}' has named RTAs declared but no named_runtime_args provided for node {}.",
                    kernel_name,
                    node.str());
            }
        }

        // Validate named CRTAs: every declared name set, no extras.
        const auto& named_crta_names = schema->named_common_runtime_args;
        TT_FATAL(
            kernel_params.named_common_runtime_args.size() == named_crta_names.size(),
            "Kernel '{}' expects {} named CRTAs, but {} were provided",
            kernel_name,
            named_crta_names.size(),
            kernel_params.named_common_runtime_args.size());
        for (const auto& name : named_crta_names) {
            TT_FATAL(
                kernel_params.named_common_runtime_args.contains(name),
                "Kernel '{}' is missing named CRTA '{}'.",
                kernel_name,
                name);
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

    // Validate DFB runtime parameters
    std::unordered_set<DFBSpecName> dfbs_with_params;
    for (const auto& dfb_params : params.dfb_run_params) {
        auto [it, inserted] = dfbs_with_params.insert(dfb_params.dfb_spec_name);
        TT_FATAL(
            inserted,
            "Duplicate dfb_spec_name '{}' in ProgramRunParams.dfb_run_params. Each DFB must appear at most once.",
            dfb_params.dfb_spec_name);
        TT_FATAL(
            !dfb_params.entry_size.has_value() && !dfb_params.num_entries.has_value(),
            "DFB size overrides are not yet implemented.");
    }

    // Unlike kernels, DFBs don't require DFBRunParams.
    // It is only required for DFBs built on borrowed memory. (Which is not yet supported.)
}

// ============================================================================
// PUBLIC ENTRY POINTS: SetProgramRunParameters + GetProgramRunParamsView
// ============================================================================

void SetProgramRunParameters(Program& program, const ProgramRunParams& params) {
    log_debug(tt::LogMetal, "Setting ProgramRunParams");

    // Validate parameters against the schema
    ValidateProgramRunParams(program, params);

    detail::ProgramImpl& program_impl = program.impl();

    // Process kernel runtime arguments.
    // Named RTAs/CRTAs and vararg RTAs/CRTAs share a single dispatch buffer each (RTA + CRTA).
    // Layout:
    //   RTA per-node:  [named_rta_0 ... named_rta_N-1, vararg_0 ... vararg_M-1]
    //   CRTA:          [named_crta_0 ... named_crta_K-1, vararg_0 ... vararg_L-1]
    // Named args are placed first in schema declaration order (matches the byte-offset
    // layout emitted into kernel_args_generated.h). The device-side get_vararg / get_common_vararg
    // helpers add the named-arg offset transparently, so kernel code indexes varargs from 0.
    for (const auto& kernel_params : params.kernel_run_params) {
        std::shared_ptr<Kernel> kernel = program_impl.get_kernel_by_spec_name(kernel_params.kernel_spec_name);
        const KernelRTASchema* schema = program_impl.get_kernel_rta_schema(kernel_params.kernel_spec_name);
        TT_FATAL(schema != nullptr, "Kernel '{}' has no RTA schema registered.", kernel_params.kernel_spec_name);

        // Build a node -> named-RTA-values-map lookup for serialization.
        std::unordered_map<NodeCoord, const std::unordered_map<std::string, uint32_t>*> named_rtas_by_node;
        for (const auto& node_params : kernel_params.named_runtime_args) {
            named_rtas_by_node[node_params.node] = &node_params.args;
        }

        // Build a node -> vararg-values-span lookup.
        std::unordered_map<NodeCoord, const std::vector<uint32_t>*> varargs_by_node;
        for (const auto& [node, args] : kernel_params.runtime_varargs) {
            varargs_by_node[node] = &args;
        }

        // Iterate over every node the kernel runs on. We need to set RTAs on any node that
        // has either named RTAs or vararg RTAs. (Validation has already confirmed coverage.)
        const std::set<CoreCoord>& kernel_nodes = kernel->logical_cores();
        const bool kernel_has_named_rtas = !schema->named_runtime_args.empty();
        for (const auto& node : kernel_nodes) {
            auto named_it = named_rtas_by_node.find(node);
            auto vararg_it = varargs_by_node.find(node);
            const bool has_named = named_it != named_rtas_by_node.end();
            const bool has_varargs = vararg_it != varargs_by_node.end();
            if (!kernel_has_named_rtas && !has_varargs) {
                continue;  // Nothing to set on this node.
            }

            std::vector<uint32_t> combined;
            combined.reserve(schema->named_runtime_args.size() + (has_varargs ? vararg_it->second->size() : 0));
            if (kernel_has_named_rtas) {
                TT_FATAL(
                    has_named,
                    "Internal error: validation passed but named RTAs missing for kernel '{}' node {}.",
                    kernel_params.kernel_spec_name,
                    node.str());
                for (const auto& name : schema->named_runtime_args) {
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

        // Combine named CRTAs and vararg CRTAs into one common buffer.
        std::vector<uint32_t> combined_crtas;
        combined_crtas.reserve(schema->named_common_runtime_args.size() + kernel_params.common_runtime_varargs.size());
        for (const auto& name : schema->named_common_runtime_args) {
            auto v_it = kernel_params.named_common_runtime_args.find(name);
            TT_FATAL(
                v_it != kernel_params.named_common_runtime_args.end(),
                "Internal error: named CRTA '{}' missing for kernel '{}'.",
                name,
                kernel_params.kernel_spec_name);
            combined_crtas.push_back(v_it->second);
        }
        combined_crtas.insert(
            combined_crtas.end(),
            kernel_params.common_runtime_varargs.begin(),
            kernel_params.common_runtime_varargs.end());

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

    // Process DFB runtime parameters
    // (Not yet supported)
}

ProgramRunParamsView& GetProgramRunParamsView(Program& program) {
    (void)program;
    TT_FATAL(false, "GetProgramRunParamsView is not yet implemented.");

    // This is the fast path, power user API.
    // Return type was changed to a reference to avoid copying the view.
    // With this API, we will need to either:
    //   - Create the view object on the first call and stash it in the Program object.
    //   - Or, create the view object upon Program construction.
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
