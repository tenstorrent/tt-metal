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
#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
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
//      o prefetcher pipe parameters (advanced option)
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
    //
    // Invariant:
    // - Must be non-empty.
    // - Must not have repeated kernel names.
    Group<KernelSpecName> kernels;

    // The set of nodes configured by this WorkUnitSpec.
    //
    // Invariant:
    // - Must be non-empty.
    Nodes target_nodes;
};

//------------------------------------------------
// ProgramSpec
//------------------------------------------------

// A ProgramSpec describes a complete Program (its immutable properties).
struct ProgramSpec {
    // Object invariants (excluding advanced options):
    // For more information, please read the headings of associated headers.
    // Invariants listed here are structural invariants of ProgramSpec, individual fields of the ProgramSpec may
    // have additional invariants listed atop.
    //
    // Every binding within the kernels refers to a memory resource:
    // - Each KernelSpec::DFBBinding::dfb_spec_name names a spec in dataflow_buffers.
    // - Each KernelSpec::SemaphoreBinding::semaphore_spec_name names a spec in semaphores.
    // - Each KernelSpec::ScratchpadBinding::scratchpad_spec_name names a spec in scratchpads.
    // - Each KernelSpec::TensorBinding::tensor_parameter_name names a parameter in tensor_parameters.
    //
    // Every WorkUnitSpec refers to declared kernels:
    // - Each name in WorkUnitSpec::kernels names a spec in kernels.
    //
    // Every declaration is used:
    // - Each kernel is listed by at least one WorkUnitSpec.
    // - Each dataflow buffer is bound by at least one kernel.
    // - Each scratchpad is bound by at least one kernel.
    // - Each tensor parameter is bound by a kernel, or by a dataflow buffer's borrowed_from.
    // - (Semaphores are exempt here)
    //
    // Borrowed memory:
    // - Each DataflowBufferSpec::borrowed_from, when set, names a parameter in tensor_parameters.
    //   - That parameter's TensorSpec is L1-resident (L1 or L1_SMALL).
    //
    // Placement per WorkUnitSpec:
    // - If any kernel in a WorkUnitSpec binds a given DataflowBufferSpec, exactly one kernel in that
    //   WorkUnitSpec binds it as a producer and exactly one binds it as a consumer.
    //   (The producer and consumer may be the same kernel.)
    // - Within each WorkUnitSpec, at most one kernel binds a given ScratchpadSpec.
    // - Each WorkUnitSpec has at most one compute kernel.
    // - Summed num_threads over a WorkUnitSpec's kernels:
    //   - Gen1: at most 1 across compute kernels, at most 2 across data-movement kernels.
    //   - Gen2: at most 4 across compute kernels, at most 6 across data-movement kernels.
    // - Gen1: within each WorkUnitSpec, data-movement kernels use distinct config_1xx->processor values
    //   and the same config_1xx->noc_mode. Those in DM_DEDICATED_NOC mode also use distinct
    //   config_1xx->noc values.
    // - Gen2: each data-movement kernel can be given num_threads DM cores, the same cores on every node
    //   it runs on, such that kernels in the same WorkUnitSpec get disjoint cores, kernels on the same
    //   side of a DataflowBufferSpec get the same cores, and DM0 and DM1 are never used.
    //
    // DataflowBufferSpec endpoints (across all kernels that bind a given DataflowBufferSpec):
    // - All PRODUCER bindings share access_pattern, and their kernels share num_threads and kind
    //   (compute or data movement). The same holds for all CONSUMER bindings.
    // - Gen1: all data-movement PRODUCER kernels share config_1xx->processor. The same holds for
    //   data-movement CONSUMER kernels.
    // - If any kernel binds it as both PRODUCER and CONSUMER, its PRODUCER kernels and its CONSUMER
    //   kernels are the same set of kernels.
    // - If any binding kernel is a compute kernel, the DataflowBufferSpec sets data_format_metadata.
    // - Gen2: on each side, the data-movement kernels agree on whether implicit sync is disabled for it
    //   (via disable_dfb_implicit_sync_for or disable_dfb_implicit_sync_for_all).
    // - With P the PRODUCER kernels' num_threads and C the CONSUMER kernels' num_threads:
    //   - STRIDED consumers: num_entries % max(P, C) == 0 and num_entries / max(P, C) <= 65535.
    //   - ALL consumers: num_entries % P == 0 and num_entries / P <= 65535.
    //
    // DataflowBufferSpecs per node:
    // - A DataflowBufferSpec occupies every node its binding kernels run on. No node is occupied by
    //   more than 32 DataflowBufferSpecs on Wormhole, 64 on Blackhole, or 32 on Quasar. TODO: THIS IS LYING.
    //
    // Unpack modes, for each DataflowBufferSpec a compute kernel binds as CONSUMER:
    // - If its data_format_metadata is 32-bit (Float32, Int32, UInt32 or RawUInt32) and the kernel's
    //   unpack_modes maps it to UnpackToDest, the kernel must set enable_32_bit_dest.
    // - If its data_format_metadata is Float32 and the kernel sets enable_32_bit_dest, the kernel's
    //   unpack_modes must have an entry for it (either mode; no default is assumed).
    //
    // Semaphores:
    // - A SemaphoreSpec bound by a compute kernel is not also bound by a data-movement kernel.
    // - At most one SemaphoreSpec is bound by compute kernels.
    //
    // All subobjects uphold their own invariants.

    // Human-readable name (debug/messaging only; no uniqueness invariant).
    std::string name;

    // Kernels that make up the Program
    //
    // Invariant:
    // - Must not have repeated KernelSpec::unique_id.
    // - Must not be empty.
    Group<KernelSpec> kernels;

    // Program-scope resources (allocated for the Program's execution lifetime)
    // DFBs (local + cross-node), and semaphores

    // Invariant:
    // - Must not have repeated DataflowBufferSpec::unique_id.
    Group<DataflowBufferSpec> dataflow_buffers;

    // Invariant:
    // - Must be empty (Not yet implemented).
    Group<CrossNodeDataflowBufferSpec> cross_node_dataflow_buffers;

    // Invariant:
    // - Must not have repeated SemaphoreSpec::unique_id.
    // - Each node must have at most 16 SemaphoreSpecs associated with them.
    Group<SemaphoreSpec> semaphores;

    // Invariant:
    // - Must not have repeated ScratchpadSpec::unique_id.
    Group<ScratchpadSpec> scratchpads;

    // Tensor parameter declarations
    // Provides ids and layout specs for tensors the Program's kernels will operate on
    // (The actual MeshTensors are supplied via ProgramRunArgs.)
    //
    // Invariant:
    // - Must not have repeated TensorParameter::unique_id.
    Group<TensorParameter> tensor_parameters;

    // WorkUnit specifications
    //
    // Invariant:
    // - Must have at least one WorkUnitSpec.
    // - All work_units must be on distinct nodes.
    Group<WorkUnitSpec> work_units;

    // Advanced options (see advanced_options.hpp)
    // Experimental PrefetcherPipe parameter declarations live here.
    ProgramAdvancedOptions advanced_options;
};

}  // namespace tt::tt_metal::experimental
