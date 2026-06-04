// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
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
        // Runtime argument values: maps each node to its named-RTA values (a name -> value table).
        // Every argument in this kernel's RuntimeArgSchema::runtime_arg_names must be set,
        // for every node the kernel runs on.
        // Missing arguments or superfluous arguments will trigger validation errors.
        //
        // NOTE: If a kernel runtime argument always has the same value for all nodes, passing
        // a common runtime argument would provide better dispatch efficiency.
        using NamedRuntimeArgs = Table<std::string, uint32_t>;
        Table<NodeCoord, NamedRuntimeArgs> runtime_arg_values;

        // Common runtime argument values (broadcast to every node).
        // Every argument in this kernel's RuntimeArgSchema::common_runtime_arg_names must be set.
        NamedRuntimeArgs common_runtime_arg_values;

        // Advanced options (see advanced_options.hpp).
        // Companion to KernelAdvancedOptions on the schema side; holds
        // positional vararg values.
        AdvancedKernelRunArgs advanced_options;
    };
    // A KernelRunArgs must be specified for ALL kernels in the ProgramSpec, keyed by kernel name.
    Table<KernelSpecName, KernelRunArgs> kernel_run_args;

    ////////////////////////////////////////////////////////////////////////
    // Tensor arguments
    ////////////////////////////////////////////////////////////////////////
    struct TensorArgument {
        // The actual MeshTensor argument
        // (Non-owning reference. Will become MeshTensorView when available; existing callsites won't change.)
        std::reference_wrapper<const MeshTensor> tensor;
    };
    // A TensorArgument must be specified for EVERY TensorParameter declared in the ProgramSpec.
    // The argument's TensorSpec must match the TensorParameter's TensorSpec (shape, layout, data type).
    Table<TensorParameterName, TensorArgument> tensor_args;

    ////////////////////////////////////////////////////////////////////////
    // DFB parameters (optional, advanced use cases)
    ////////////////////////////////////////////////////////////////////////
    struct DFBRunOverrides {
        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // If unset, the ProgramSpec value is used.
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;

        // Note: borrowed-memory DFBs update their backing L1 SRAM address from
        // the corresponding tensor_arg.
    };
    // DFBRunOverrides is optional. Provide entries only when overriding DFB sizes.
    Table<DFBSpecName, DFBRunOverrides> dfb_run_overrides;
};

//------------------------------------------------
// ProgramRunArgsView (for advanced users)
//------------------------------------------------
//
// NOTE: ProgramRunArgsView is not yet supported! It is included here as a sketch only.
//
// Non-owning view into a Program's command buffers.
// Enables in-place modification of mutable Program parameters.
//
// STATEFULNESS: Program command buffers are stateful.
//   Parameters retain their previously specified value unless modified.
//
// LIFETIME: This view is valid for the lifetime of the Program.
//   Accessing the view after the Program is destroyed is undefined behavior.
//
// THREAD SAFETY: Modifications through this view are not synchronized;
//   the caller must ensure exclusive access when modifying.
//
struct ProgramRunArgsView {
    struct KernelRunArgsView {
        // Direct views into per-node vararg runtime args
        std::vector<std::pair<NodeCoord, std::span<uint32_t>>> runtime_varargs;

        // Direct view into common vararg runtime args
        std::span<uint32_t> common_runtime_varargs;
    };
    // TODO: Better to just expose the multi-dim dispatch vectors directly?
    //       Would eliminate the lookup indirection.
    //       ...But would mess up all the implicit RTAs....
    //       Look into this when implementing.
    std::unordered_map<KernelSpecName, KernelRunArgsView> kernel_run_args;

    struct DFBRunOverridesView {
        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // (This is seldom used in practice)
        uint32_t* entry_size;   // points to the value that will be used to allocate DFB ephemeral memory
        uint32_t* num_entries;  // always set to non-null location

        // Note: borrowed-memory DFBs update their backing L1 SRAM address from
        // the corresponding tensor_arg.
    };
    std::unordered_map<DFBSpecName, DFBRunOverridesView> dfb_run_overrides;
};

// TODO: Consider a const version of the view object, for debug/test use?

}  // namespace tt::tt_metal::experimental
