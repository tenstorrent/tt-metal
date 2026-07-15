// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  ProgramRunArgs API
// ============================================================================
//
// A ProgramRunArgs object is a descriptor for the mutable properties of a
// Metalium Program, which can be specified anew with each Program execution
// (enqueue).
//
// ProgramRunArgs is the partner object to ProgramSpec, which describes the
// immutable properties of a Program. A ProgramRunArgs object can only be
// understood in the context of its corresponding ProgramSpec (program_spec.hpp).
// The ProgramSpec provides the schema for the mutable Program properties; the
// ProgramRunArgs provides the values.
//
// ============================================================================

//------------------------------------------------
// ProgramRunArgs
//------------------------------------------------

struct ProgramRunArgs {
    ////////////////////////////////////////////////////////////////////////
    // Kernel runtime arguments
    ////////////////////////////////////////////////////////////////////////
    struct KernelRunArgs {
        KernelSpecName kernel;

        // Per-node runtime argument values.
        // Arguments are keyed by argument name and then by node:
        //   runtime_arg_values[name][node] = value.
        //
        // When calling SetProgramRunArgs:
        //  Every argument in this kernel's RuntimeArgSchema::runtime_arg_names must be
        //  set, for every node the kernel runs on.
        //  Missing arguments or superfluous arguments will trigger validation errors.
        //
        // When calling UpdateProgramRunArgs:
        //  Only non-enqueue invariant runtime arguments must be set.
        //  Updated arguments must be updated for every node the kernel runs on.
        //
        // NOTE: If a kernel runtime argument always has the same value for all nodes,
        // passing a common runtime argument would provide better dispatch efficiency.
        //
        using RuntimeArgValues = Table<std::string, Table<NodeCoord, uint32_t>>;
        RuntimeArgValues runtime_arg_values;

        // Common runtime argument values (broadcast to every node).
        //
        // When calling SetProgramRunArgs:
        //  Every argument in this kernel's RuntimeArgSchema::common_runtime_arg_names
        //  must be set.
        //
        // When calling UpdateProgramRunArgs:
        //  Only non-enqueue invariant common runtime arguments must be set.
        //  (However, CRTAs should seldom-to-never be enqueue invariant!)
        //
        using CommonRuntimeArgValues = Table<std::string, uint32_t>;
        CommonRuntimeArgValues common_runtime_arg_values;

        // Advanced options (see advanced_options.hpp).
        // Companion to KernelAdvancedOptions on the schema side; holds
        // positional vararg values.
        AdvancedKernelRunArgs advanced_options;
    };
    // A KernelRunArgs must be specified for ALL kernels in the ProgramSpec.
    Group<KernelRunArgs> kernel_run_args;

    ////////////////////////////////////////////////////////////////////////
    // Tensor arguments
    ////////////////////////////////////////////////////////////////////////

    // The actual MeshTensor argument
    // (Non-owning reference. Will also permit MeshTensorView when it becomes available.)
    using TensorArgument = std::variant<std::reference_wrapper<const MeshTensor>>;

    // A TensorArgument must be specified for EVERY TensorParameter declared in the ProgramSpec.
    // The argument's TensorSpec must match the TensorParameter's TensorSpec (shape, layout, data type).
    Table<TensorParamName, TensorArgument> tensor_args;

    ////////////////////////////////////////////////////////////////////////
    // DFB parameters (optional, advanced use cases)
    ////////////////////////////////////////////////////////////////////////
    struct DFBRunOverrides {
        DFBSpecName dfb;

        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // These overrides are stateful across executions: if unset, the DFB keeps its current size
        // (initially the ProgramSpec value; a prior override persists until changed).
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;

        // Note: borrowed-memory DFBs update their backing L1 SRAM address from
        // the corresponding tensor_arg.
    };
    // DFBRunOverrides is optional. Provide entries only when overriding DFB sizes.
    Group<DFBRunOverrides> dfb_run_overrides;
};

//-----------------------------------------------------
// Convenience aliases
//-----------------------------------------------------

using KernelRunArgs = ProgramRunArgs::KernelRunArgs;
using DFBRunOverrides = ProgramRunArgs::DFBRunOverrides;
using TensorArgument = ProgramRunArgs::TensorArgument;

//-----------------------------------------------------
// Helper functions
//-----------------------------------------------------

// Resolve a TensorArgument to its MeshTensor.
// (Switch to std::visit once MeshTensorView is added as a second variant alternative.)
inline const MeshTensor& mesh_tensor_of(const TensorArgument& arg) {
    return std::get<std::reference_wrapper<const MeshTensor>>(arg).get();
}

// Helper function to merge two or more ProgramRunArgs objects into one.
// Validates that the provided ProgramRunArgs objects specify mutually disjoint arguments.
ProgramRunArgs MergeProgramRunArgs(
    ProgramRunArgs base, std::span<const ProgramRunArgs> rest, bool skip_validation = false);
// Invocation: auto full = MergeProgramRunArgs(std::move(base_run_args), {appended_run_args});

//-----------------------------------------------------
// Helper function for per-node runtime argument values
//-----------------------------------------------------

// Runtime argument (RTA) values must be provided for every node the kernel runs on.
// In ProgramRunArgs, runtime arguments (RTAs) are expressed first by name, then by
// node (name -> node -> value). However, legacy use sites usually produce RTA values
// in a node-first style (node -> name -> value).
//
// These helper functions bridge the gap between the legacy style and ProgramRunArgs.
// It is much better to refactor legacy code to express RTA values in name-first style.
// But, these helpers are provided for convenience and backward compatibility.

// Two shapes for the two call patterns:
//
//   Multi-node kernel — accumulate across the core loop:
//     KernelRunArgs kra{.kernel = ...};
//     for (const auto& core : cores) {
//         AddRuntimeArgsForNode(kra.runtime_arg_values, core, {{"num_tiles", num_tiles[core]}, ...});
//     }
//
//   Special case for a single-node kernel — build the table inline:
//     KernelRunArgs{
//         .kernel = ...,
//         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
//              node, {{"var", a}, {"num_tiles", n}}),
//     };

// Append RTAs for a single node to an existing RuntimeArgValues table.
inline void AddRuntimeArgsForNode(
    KernelRunArgs::RuntimeArgValues& runtime_arg_values,                  // existing RTA table
    const NodeCoord& node,                                                // node
    std::initializer_list<std::pair<std::string, uint32_t>> named_values  // name -> value list
) {
    for (const auto& [name, value] : named_values) {
        runtime_arg_values[name][node] = value;
    }
}

// Build a fresh table for a single node
inline KernelRunArgs::RuntimeArgValues MakeRuntimeArgsForSingleNode(
    const NodeCoord& node, std::initializer_list<std::pair<std::string, uint32_t>> named_values) {
    KernelRunArgs::RuntimeArgValues runtime_arg_values;
    AddRuntimeArgsForNode(runtime_arg_values, node, named_values);
    return runtime_arg_values;
}

}  // namespace tt::tt_metal::experimental
