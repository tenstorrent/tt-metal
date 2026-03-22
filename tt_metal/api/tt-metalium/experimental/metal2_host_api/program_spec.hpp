// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

using WorkerSpecName = std::string;
using ProgramSpecName = std::string;

//------------------------------------------------
// ProgramSpec & WorkerSpec
//------------------------------------------------

// WorkerSpec describes the configuration of a worker node
struct WorkerSpec {
    // Worker type identifier
    WorkerSpecName unique_id;

    // Kernels, DFBs, and semaphores for this worker
    std::vector<KernelSpecName> kernels;
    std::vector<DFBSpecName> dataflow_buffers;
    std::vector<SemaphoreSpecName> semaphores;

    // The set of nodes configured by this WorkerSpec
    std::variant<NodeCoord, NodeRange, NodeRangeSet> target_nodes;
};

// ProgramSpec describes the immutable properties of a Program
//   (analogous to a function's signature and body)
struct ProgramSpec {
    // Program identifier (identifies a Program within a MeshWorkload)
    ProgramSpecName program_id;

    // Kernels, DFBs, and semaphores for this Program
    std::vector<KernelSpec> kernels;
    std::vector<DataflowBufferSpec> dataflow_buffers;
    std::vector<SemaphoreSpec> semaphores;

    // Worker specifications (optional on Gen1, required on Gen2+)
    // This info is redundant, but improves clarity and messaging.
    // (Done to simplify porting from ProgramDescriptor.)
    std::optional<std::vector<WorkerSpec>> workers = std::nullopt;
};

//------------------------------------------------
// ProgramRunParams
//------------------------------------------------
// Describes the mutable properties of a Program, specified anew for each Program execution.
//   (analogous to function arguments)
struct ProgramRunParams {
    ////////////////////////////////////////////////////////////////////////
    // Kernel runtime arguments
    ////////////////////////////////////////////////////////////////////////
    struct KernelRunParams {
        // Kernel identifier
        KernelSpecName kernel_spec_name;

        // Defined runtime arguments (named & typed)
        //   TODO

        // Defined common runtime arguments (named & typed)
        //   TODO

        // Unnamed runtime argument "varargs"
        // (these are specified per-node; length can vary per-node)
        using NodeRuntimeArgs = std::vector<uint32_t>;
        using RuntimeArgs = std::vector<std::pair<NodeCoord, NodeRuntimeArgs>>;
        RuntimeArgs runtime_args;

        // Unnamed common runtime argument "varargs"
        // (common to all nodes on which the kernel runs)
        using CommonRuntimeArgs = std::vector<uint32_t>;
        CommonRuntimeArgs common_runtime_args;
    };
    std::vector<KernelRunParams> kernel_run_params;

    ////////////////////////////////////////////////////////////////////////
    // DFB parameters (optional, advanced use cases)
    ////////////////////////////////////////////////////////////////////////
    struct DFBRunParams {
        // DFB identifier
        DFBSpecName dfb_spec_name;

        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // If unset, the ProgramSpec value is used.
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;

        // DFB borrowed memory
        // For DFBs built on borrowed memory, the underlying memory is passed as an argument.
        // using BorrowedMemory = std::variant<BufferView, MeshTensorView>; // non-owning view types, TBD
        // std::optional<BorrowedMemory> borrowed_memory = std::nullopt;
    };
    std::vector<DFBRunParams> dfb_run_params;
};

//------------------------------------------------
// ProgramRunParamsView (for advanced users)
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
struct ProgramRunParamsView {
    struct KernelRunParamsView {
        // Direct views into per-node runtime args
        std::vector<std::pair<NodeCoord, std::span<uint32_t>>> runtime_args;

        // Direct view into common runtime args
        std::span<uint32_t> common_runtime_args;
    };
    // TODO: Better to just expose the multi-dim dispatch vectors directly?
    //       Would eliminate the lookup indirection.
    //       ...But would mess up all the implicit RTAs....
    //       Look into this when implementing.
    std::unordered_map<KernelSpecName, KernelRunParamsView> kernel_run_params;

    struct DFBRunParamsView {
        // DFB size overrides
        // DFB sizes specified in the ProgramSpec may be overridden per Program execution.
        // (This is seldom used in practice)
        uint32_t* entry_size;   // points to the value that will be used to allocate DFB ephemeral memory
        uint32_t* num_entries;  // always set to non-null location

        // DFB borrowed memory
        // For DFBs built on borrowed memory, the underlying memory is passed as an argument.
        // (TODO)
    };
    std::unordered_map<DFBSpecName, DFBRunParamsView> dfb_run_params;
};

// TODO: Consider a const version of the view object, for debug/test use?

}  // namespace tt::tt_metal::experimental::metal2_host_api
