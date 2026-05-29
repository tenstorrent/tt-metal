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
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

//------------------------------------------------
// ProgramRunArgs
//------------------------------------------------
// Describes the mutable properties of a Program, specified anew for each Program execution.
//   (analogous to function arguments)
struct ProgramRunArgs {
    ////////////////////////////////////////////////////////////////////////
    // Kernel runtime arguments
    ////////////////////////////////////////////////////////////////////////
    struct KernelRunArgs {
        // Kernel identifier
        KernelSpecName kernel_spec_name;

        // Runtime argument values (per-node).
        // Every argument in this kernel's RuntimeArgSchema::runtime_arg_names must be set,
        // for every node the kernel runs on.
        // Missing arguments or superfluous arguments will trigger validation errors.
        //
        // NOTE: If a kernel runtime argument always has the same value for all nodes, passing
        // a common runtime argument would provide better dispatch efficiency.
        struct NodeRuntimeArgs {
            NodeCoord node;
            std::unordered_map<std::string, uint32_t> args;
        };
        std::vector<NodeRuntimeArgs> runtime_arg_values;

        // Common runtime argument values (broadcast to every node).
        // Every argument in this kernel's RuntimeArgSchema::common_runtime_arg_names must be set.
        std::unordered_map<std::string, uint32_t> common_runtime_arg_values;

        // Advanced options (see advanced_options.hpp).
        // Companion to KernelAdvancedOptions on the schema side; holds
        // positional vararg values.
        AdvancedKernelRunArgs advanced_options;
    };
    // KernelRunArgs must be specified for ALL kernels in the ProgramSpec.
    std::vector<KernelRunArgs> kernel_run_args;

    ////////////////////////////////////////////////////////////////////////
    // Tensor arguments
    ////////////////////////////////////////////////////////////////////////
    struct TensorArgument {
        // Tensor identifier (matches a TensorParameter::unique_id in the ProgramSpec)
        TensorParameterName tensor_parameter_name;

        // The actual MeshTensor argument
        // (Non-owning reference. Will become MeshTensorView when available; existing callsites won't change.)
        std::reference_wrapper<const MeshTensor> tensor;
    };
    // A TensorArgument must be specified for EVERY TensorParameter declared in the ProgramSpec.
    // The argument's TensorSpec must match the TensorParameter's TensorSpec (shape, layout, data type).
    std::vector<TensorArgument> tensor_args;

    ////////////////////////////////////////////////////////////////////////
    // DFB parameters (optional, advanced use cases)
    ////////////////////////////////////////////////////////////////////////
    struct DFBRunOverrides {
        // DFB identifier
        DFBSpecName dfb_spec_name;

        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // If unset, the ProgramSpec value is used.
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;

        // Note: borrowed-memory DFBs update their backing L1 SRAM address from
        // the corresponding tensor_arg.
    };
    // DFBRunOverrides is optional. Provide entries only when overriding DFB sizes.
    std::vector<DFBRunOverrides> dfb_run_overrides;
};

//------------------------------------------------
// ProgramRunArgsView (for advanced users)
//------------------------------------------------
// Non-owning view into a Program's command buffers.
// Enables in-place modification of mutable Program parameters.
// (WIP -- this is a sketch only)
//
// STATEFULNESS: Program command buffers are stateful. Parameters retain their previous value unless modified.
// (Q: How does this work for back-to-back enqueues?)
// (A: TBD. There is definitely foot-shooting potential if you overwrite parameters for a previous enqueue...
//     This API is "unsafe", intended for advanced users who will synchronize correctly.)
//
// LIFETIME: This view is valid for the lifetime of the Program.
// Accessing the view after the Program is destroyed is undefined behavior.
//
// THREAD SAFETY: Modifications through this view are not synchronized;
// the caller must ensure exclusive access when modifying.
//
// TODO: This will need rethinking for typed runtime arguments.
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

}  // namespace tt::tt_metal::experimental::metal2_host_api
