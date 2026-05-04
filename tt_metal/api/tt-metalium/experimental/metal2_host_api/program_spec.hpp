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

// A name identifying a WorkUnitSpec within a ProgramSpec.
// CONVENTION: define names as `constexpr const char*` constants (see above).
using WorkUnitSpecName = std::string;

//------------------------------------------------
// ProgramSpec & WorkUnitSpec
//------------------------------------------------

// A WorkUnitSpec describes a set of kernels that run together on a set of nodes.
// Each node in the WorkUnitSpec's target_nodes runs an identical set of kernel instances.
//
// Placement: The WorkUnitSpec defines the node placement of its kernels.
// (A kernel may be included in multiple WorkUnitSpecs.)
struct WorkUnitSpec {
    WorkUnitSpecName unique_id;

    // The kernels that run on this WorkUnitSpec's nodes.
    std::vector<KernelSpecName> kernels;

    // The set of nodes configured by this WorkUnitSpec.
    std::variant<NodeCoord, NodeRange, NodeRangeSet> target_nodes;
};

// A ProgramSpec describes the immutable properties of a Program:
// its kernels, DFBs, semaphores, and where they all run.
// Analogous to a function's signature and body — declared once, executed many times.
// (Each time with a new ProgramRunParams configuring the mutable execution parameters.)
struct ProgramSpec {
    // Program identifier (identifies a Program within a MeshWorkload)
    ProgramSpecName program_id;

    // Kernels, DFBs (local + remote), and semaphores that make up the Program
    std::vector<KernelSpec> kernels;
    std::vector<DataflowBufferSpec> dataflow_buffers;
    std::vector<RemoteDataflowBufferSpec> remote_dataflow_buffers;
    std::vector<SemaphoreSpec> semaphores;

    // WorkUnit specifications:
    // A valid ProgramSpec has at least one WorkUnitSpec.
    // Each kernel must be referenced by at least one WorkUnitSpec.
    std::vector<WorkUnitSpec> work_units;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
