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
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  ProgramSpec API
// ============================================================================
//
// A ProgramSpec is a descriptor object used to create a Metalium Program object.
// The ProgramSpec describes all the IMMUTABLE properties of a Program:
//  - compiled kernels
//  - program-scope resources
//      o dataflow buffers
//      o semaphores
//      o scratchpads
//  - user-managed resources (parameters)
//      o tensor parameters
//
// It also specifies the device nodes (physical location) where kernels will run,
// and where device resources will be allocated.
//
// The ProgramSpec is analogous to a function's signature and body —
// it is declared once, but can be executed many times.
//
// ProgramRunArgs (program_run_args.hpp) is the partner object to ProgramSpec.
// This descriptor is analogous to the function invocation's arguments.
// ProgramRunArgs describes the MUTABLE properties of a Program, which are specified
// anew with each execution (enqueue) of the Program.
//
// ============================================================================

//------------------------------------------------
// WorkUnitSpec
//------------------------------------------------

// A WorkUnitSpec describes a set of kernels that run together on a set of nodes.
// Each node in the WorkUnitSpec's target_nodes runs an identical set of kernel instances.
//
// Placement: The WorkUnitSpec defines the node placement of its kernels.
//   (A kernel may be included in multiple WorkUnitSpecs.)
//
struct WorkUnitSpec {
    // Human-readable name (debug/messaging only; no uniqueness invariant).
    std::string name;

    // The kernels that run on this WorkUnitSpec's nodes.
    Group<KernelSpecName> kernels;

    // The set of nodes configured by this WorkUnitSpec.
    Nodes target_nodes;
};

//------------------------------------------------
// ProgramSpec
//------------------------------------------------

// A ProgramSpec describes a complete Program (its immutable properties).
struct ProgramSpec {
    // Human-readable name (debug/messaging only; no uniqueness invariant).
    std::string name;

    // Kernels that make up the Program
    Group<KernelSpec> kernels;

    ////////////////////////////////////
    // Program-scope resources
    ////////////////////////////////////

    // Program-scope resources are runtime-managed memory resouces.
    // Their allocation is handled automatically, and they exist only during the
    // Program's execution lifetime.

    Group<DataflowBufferSpec> dataflow_buffers;
    Group<CrossNodeDataflowBufferSpec> cross_node_dataflow_buffers;
    Group<SemaphoreSpec> semaphores;
    Group<ScratchpadSpec> scratchpads;

    ////////////////////////////////////
    // Program parameters
    ////////////////////////////////////

    // Program parameters are user-managed memory resources. (User-controlled lifetime.)
    // The ProgramSpec declares these resources as parameters.
    // The actual resource objects are to supplied by the user as Program arguments at
    // runtime (via ProgramRunArgs), prior to enqueueing the Program for execution.

    // Tensor parameters
    // Defines the IDs and layout specs for MeshTensors the Program's kernels will operate on
    Group<TensorParameter> tensor_parameters;

    // GlobalSemaphore parameters
    Group<GlobalSemaphoreParameter> global_semaphore_parameters;

    // PrefetcherPipe parameters
    Group<PrefetcherPipeParameter> prefetcher_pipe_parameters;

    ////////////////////////////////////
    // Work Units
    ////////////////////////////////////

    // A valid ProgramSpec has at least one WorkUnitSpec.
    // Each kernel must be referenced by at least one WorkUnitSpec.
    Group<WorkUnitSpec> work_units;
};

}  // namespace tt::tt_metal::experimental
