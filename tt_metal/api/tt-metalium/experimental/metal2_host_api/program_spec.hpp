// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a ProgramSpec within a MeshWorkload.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* MATMUL_PROGRAM = "matmul";
//   ProgramSpec{.program_id = MATMUL_PROGRAM, ...};
// Reusing a single constant helps catch typos and errors at compile time.
using ProgramSpecName = std::string;

// A name identifying a WorkerSpec within a ProgramSpec.
// CONVENTION: define names as `constexpr const char*` constants (see above).
using WorkerSpecName = std::string;

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

    // (Optional) Worker specifications.
    // If std::nullopt, workers are inferred automatically from kernel/DFB/semaphore
    // target_nodes fields (nodes with identical coverage become one inferred WorkerSpec,
    // named "inferred_worker@<node_range_set>").
    // User-supplied workers let you assign readable names, and act as a redundant sanity
    // check on which nodes each entity covers.
    std::optional<std::vector<WorkerSpec>> workers = std::nullopt;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
