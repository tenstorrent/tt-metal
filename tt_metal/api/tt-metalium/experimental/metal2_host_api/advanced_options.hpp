// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
// Advanced options for Metal 2.0 specs
// ============================================================================
//
// Each Metal 2.0 Spec (KernelSpec, DataflowBufferSpec, TensorParameter, ...) may
// carry a *AdvancedOptions field at the end of its struct.
// Features in "advanced options" are one (or more) of:
//
//   - Not safe by construction (requires extreme caution to use correctly)
//   - Extremely niche (only relevant to a tiny fraction of use cases)
//   - Unstable / experimental
//
// NOTE: Features that are "advanced" but mainstream and core to the API
//       belong on the primary Spec. *AdvancedOptions features are limited
//       to those that are truly niche, unsafe, or unstable.
//
// Use the advanced options with caution!
// The header comments for each field describe special considerations for use.
//
// ============================================================================

// Name identifying a DataflowBufferSpec within a ProgramSpec.
using DFBSpecName = ttsl::StrongType<std::string, struct DFBSpecNameTag>;
// NOTE: DFBSpecName is also declared at the top of dataflow_buffer_spec.hpp, but is
//       re-declared here for use in AdvancedOptions to avoid circular dependency.
//       This is legal so long as the declarations are identical, which is compiler-enforced.

struct KernelAdvancedOptions {
    ////////////////////////////////////////////////////////////////////////////////
    // Enqueue-loop invariant kernel arguments
    ////////////////////////////////////////////////////////////////////////////////

    // Designate certain runtime arguments and common runtime arguments as enqueue-loop
    // invariant. This permits the same argument value to be reused across multiple Program
    // enqueues via UpdateProgramRunArgs, which can improve performance in enqueue loops.
    // By default, every runtime argument and common runtime argument is expected to be
    // re-specified (via SetProgramRunArgs) on every enqueue.
    //
    // CAUTION: This feature is unsafe if used incorrectly! The onus is on the programmer
    // to ensure that the designated arguments remain valid across enqueues.
    Group<std::string> enqueue_invariant_runtime_args;
    Group<std::string> enqueue_invariant_common_runtime_args;

    ////////////////////////////////////////////////////////////////////////////////
    // Varargs
    ////////////////////////////////////////////////////////////////////////////////

    // In Metal 2.0, kernel arguments are NAMED parameters declared in the KernelSpec.
    // However, until typed kernel argument support is available, certain advanced use
    // cases require a VARIABLE number of arguments. e.g.:
    //   - N runtime arguments, representing the size of an N-dimensional tensor
    //   - a kernel that accepts a variadic number of tensor arguments
    //
    // Varargs must be accessed POSITIONALLY in the kernel code.
    //
    // The vararg schema below is a temporary mechanism to support these use cases.
    // It will later be deprecated and replaced by std::array typed arguments.

    //--------------------------------
    // Compile time varargs
    //--------------------------------
    // TODO: This is currently unimplemented.
    //       However, certain variadic kernels require this workaround.
    //       (#45388 tracks the implementation of this feature.)

    //--------------------------------
    // Runtime varargs
    //--------------------------------
    // Number of runtime varargs for the kernel.
    // Set the vararg values (per node) via ProgramRunArgs.
    uint32_t num_runtime_varargs = 0;

    // Number of common runtime varargs for the kernel.
    // Set the vararg values via ProgramRunArgs.
    // (The same argument values are broadcast to every node the kernel runs on.)
    uint32_t num_common_runtime_varargs = 0;

    // Per-node runtime vararg-count override.
    // In very rare cases a kernel needs a DIFFERENT number of runtime varargs on
    // different nodes. Each entry pairs a node set with its vararg count; nodes
    // not listed default to num_runtime_varargs.
    // TODO: This feature is truly bizarre. It will be removed from the API once
    //       existing uses are refactored to avoid it.
    [[deprecated("Per-node-vararg-count feature is deprecated and will be removed.")]]
    Table<Nodes, /* num_varargs */ uint32_t> num_runtime_varargs_per_node;
};

struct DFBAdvancedOptions {
    ////////////////////////////////////////////////////////////////////////////////
    // Aliased DFBs
    ////////////////////////////////////////////////////////////////////////////////

    // Alias two or more DFBs.
    // Aliased DFBs are logically distinct, but physically share the same backing memory.
    // This is an advanced feature for memory use optimization in niche use cases.
    //
    // CAUTION:
    // Aliased DFBs offer NO guarantees against data clobbering!
    // This feature is unsafe in most circumstances; kernel logic must ensure safety.
    //
    // Rules for aliased DFBs:
    //   - Every DFB in the alias group must list every other member as an alias
    //   - Aliased DFBs must have the same total size (num_entries * entry_size).
    //   - All members must target the same node set
    //     (derived from their bound kernels' WorkUnitSpecs).
    Group<DFBSpecName> alias_with;
};

struct AdvancedKernelRunArgs {
    ////////////////////////////////////////////////////////////////////////////////
    // Varargs
    ////////////////////////////////////////////////////////////////////////////////

    using Varargs = std::vector<uint32_t>;

    // Unnamed runtime argument "varargs"
    // (Companion to the vararg schema declared on KernelAdvancedOptions).
    // Specified per-node; length can vary per-node (as declared in schema).
    Table<NodeCoord, Varargs> runtime_varargs;

    // Unnamed common runtime argument "varargs"
    // (Companion to num_common_runtime_varargs in the schema.)
    // Broadcast to every node the kernel runs on.
    Varargs common_runtime_varargs;
};

struct SemaphoreAdvancedOptions {
    ////////////////////////////////////////////////////////////////////////////////
    // Non-zero initial value
    ////////////////////////////////////////////////////////////////////////////////

    // NOTE: Setting a non-zero initial value is not supported on Gen2 architectures.
    // NOTE: Runtime wants to deprecate this feature for ALL architectures.
    //       When cross-node DFB becomes available, non-zero initial values will be removed.
    [[deprecated("Non-zero semaphore initialization is deprecated and will be removed.")]]
    uint32_t initial_value = 0;
};

struct TensorParameterAdvancedOptions {
    ////////////////////////////////////////////////////////////////////////////////
    // Enqueue-loop invariance
    ////////////////////////////////////////////////////////////////////////////////

    // Designate this TensorParameter as enqueue-loop invariant.
    // Permits the same MeshTensor argument to be reused across multiple Program
    // enqueues via UpdateProgramRunArgs. By default, a TensorParameter is expected to
    // be re-specified on every enqueue.
    //
    // CAUTION:
    // The user is responsible for managing the MeshTensor argument's lifetime and
    // ensuring that it remains valid across enqueues. Undefined behavior will result
    // if the MeshTensor goes out of scope (and its device memory is deallocated),
    // and you try to re-enqueue the Program with the now-stale MeshTensor argument.
    bool enqueue_invariant = false;

    ////////////////////////////////////////////////////////////////////////////////
    // TensorSpec match relaxation options
    ////////////////////////////////////////////////////////////////////////////////

    // By default, the MeshTensor argument provided at execution time must
    // EXACTLY match the TensorParameter's declared TensorSpec.
    // The options here relax this match requirement in particular ways.
    //
    // CAUTION:
    // These options are UNSAFE if set to true; most kernels will not function
    // correctly if the tensor argument's spec deviates from the declared spec.
    // Use with caution. You must ensure that your kernel logic outside of the
    // TensorAccessor itself is compatible with the chosen relaxation option(s)!

    // Permit tensor arguments whose logical_shape differs from the declared shape.
    // The argument's padded_shape must still match exactly.
    //
    // Effects:
    //  - Validation checks are relaxed
    //  - TensorAccessor configuration is completely unchanged
    bool match_padded_shape_only = false;

    // Permit tensor arguments with dynamic logical shape.
    // The argument's logical_shape AND padded_shape may differ from the declared shape.
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - For a sharded tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's actual shape.
    //    Shape, expressed in pages-per-dim, becomes implicit common runtime arguments.
    //  - For an interleaved TILED tensor:
    //    TensorAccessor configuration is unchanged
    //    (The page size is fixed by dtype/tile dims, so it cannot vary with shape).
    //  - For an interleaved ROW-MAJOR tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's page size.
    //    NOTE: page_size = last_dim_width * element_size is part of the varying shape!
    //    The aligned_page_size becomes an implicit common runtime argument.
    //    (Your kernel can access this value via TensorAccessor::get_aligned_page_size().)
    bool dynamic_tensor_shape = false;
};

}  // namespace tt::tt_metal::experimental
