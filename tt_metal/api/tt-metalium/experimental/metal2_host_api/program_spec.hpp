// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

using WorkerSpecName = std::string;

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
    std::vector<KernelSpec> kernels;
    std::vector<DataflowBufferSpec> dataflow_buffers;
    std::vector<SemaphoreSpec> semaphores;

    // Worker specifications (optional on Gen1, required on Gen2+)
    // This info is redundant, but improves clarity and messaging.
    // (Done to simplify porting from ProgramDescriptor.)
    std::optional<std::vector<WorkerSpec>> workers = std::nullopt;
};

// ProgramRunParams describes the mutable properties of a Program,
// which are specified anew for each Program execution
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

        // Unnamed runtime arguments
        // (specified per-node, length can vary per-node)
        using NodeRuntimeArgs = std::vector<uint32_t>;
        using RuntimeArgs = std::vector<std::pair<NodeCoord, NodeRuntimeArgs>>;
        RuntimeArgs runtime_args;

        // Unnamed common runtime arguments (shared by all cores of a kernel)
        // (specified for all nodes)
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
        // DFB sizes specified in the ProgramSpec can (optionally) be overridden per Program execution.
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;

        // DFB borrowed memory
        // For DFBs built on borrowed memory, the underlying memory is passed as an argument.
        // using BorrowedMemory = std::variant<BufferView, MeshTensorView>; // non-owning view types, TBD
        // std::optional<BorrowedMemory> borrowed_memory = std::nullopt;
    };

};

}  // namespace tt::tt_metal::experimental::metal2_host_api
