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
// Each Metal 2.0 Spec (KernelSpec, DataflowBufferSpec, ...) may
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
    // Runtime varargs
    //--------------------------------
    // Number of runtime varargs for the kernel.
    // Set the vararg values (per node) via ProgramRunArgs.
    //
    // To retrieve these values in kernel code, use:
    //   get_vararg(uint32_t idx); // index in [0, num_runtime_varargs - 1]
    //
    // CAUTION: This feature exists to address niche uses cases only.
    //          Prefer regular, named runtime arguments unless varargs are strictly necessary.
    uint32_t num_runtime_varargs = 0;

    // Number of common runtime varargs for the kernel.
    // Set the vararg values via ProgramRunArgs.
    // (The same argument values are broadcast to every node the kernel runs on.)
    //
    // To retrieve these values in kernel code, use:
    //    get_common_vararg(uint32_t idx); // index in [0, num_common_runtime_varargs - 1]
    //
    // CAUTION: This feature exists to address niche uses cases only.
    //          Prefer named common runtime arguments unless varargs are strictly necessary.
    uint32_t num_common_runtime_varargs = 0;

    // Per-node runtime vararg-count override.
    // In very rare cases a kernel needs a DIFFERENT number of runtime varargs on
    // different nodes. Each entry pairs a node set with its vararg count; nodes
    // not listed default to num_runtime_varargs.
    // TODO: This feature is truly bizarre. It will be removed from the API once
    //       existing uses are refactored to avoid it.
    [[deprecated("Per-node-vararg-count feature is deprecated and will be removed.")]]
    Table<Nodes, /* num_varargs */ uint32_t> num_runtime_varargs_per_node;

    //--------------------------------
    // Compile time varargs
    //--------------------------------
    // Compile-time vararg VALUES for the kernel.
    // (Unlike the runtime varargs fields above, these values are baked into the Program
    // at kernel compile time.)
    //
    // To retrieve these values in kernel code, use:
    //   - get_compile_time_vararg(idx)   // for a computed index
    //   - get_compile_time_vararg<idx>() // for a compile-time constant index
    //   - get_num_compile_time_varargs() // for the count
    //
    // CAUTION: This is a temporary API that will removed in favor of compile-time array arguments.
    //          It exists to solve a niche, isolated use case.
    //          Always prefer regular, named compile-time arguments.
    [[deprecated("Compile-time varargs is a temporary feature that will be removed in the future.")]]
    std::vector<uint32_t> compile_time_varargs;

    ////////////////////////////////////////////////////////////////////////////////
    // Tensor binding sequences
    ////////////////////////////////////////////////////////////////////////////////

    // A tensor binding sequence gives a kernel an additional way to retrieve its tensor
    // bindings: positionally, by index, rather than by name.
    //
    // In kernel code, a tensor binding is normally retrieved by name (e.g. `tensor::in0`).
    // Declaring a TensorBindingSequence with a list of tensor binding names also emits the
    // following into the generated header's `tensor` namespace, in the order specified:
    //
    //    constexpr auto my_binding_sequence = std::make_tuple(in0, in1, /* ... */ inN);
    //
    // To make use of this, the kernel then calls:
    //   /* create a tuple of TensorAccessor from the binding token tuple */
    //   auto accessor_tuple =  make_tensor_accessors(my_binding_sequence);
    //   /* create an array non-owning, type-erased TensorAccessor handles */
    //   auto accessor_array = make_abstract_tensor_accessor_wrappers(accessor_tuple);
    //
    // Usage: A niche mechanism for a kernel that wishes to express a compile-time-variadic number of
    //        tensor bindings, and therefore needs to access them positionally. Prefer the default
    //        named TensorBindingToken access whenever possible.
    //
    // Notes:
    //   - The named tokens are still emitted, so a sequence adds positional access rather than
    //      replacing named access.
    //   - A separate count argument is not required, as the sequence is available kernel-side via
    //      `std::tuple_size_v<decltype(tensor::my_binding_sequence)>`
    //
    // Constraints:
    //   - Every `members` entry is a TensorBinding::accessor_name on this kernel
    //   - No duplicate members within a sequence (one binding may appear in several sequences)
    //   - `accessor_name` is a valid C++ identifier, unique in this kernel's `tensor::` namespace
    //   - An empty members list is legal; this produces an empty std::tuple<>.

    struct TensorBindingSequence {
        std::string sequence_name;         // device: tensor::<sequence_name>
        std::vector<std::string> members;  // TensorBinding::accessor_name; order is the device tuple order
    };
    Group<TensorBindingSequence> tensor_binding_sequences;
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

    ////////////////////////////////////////////////////////////////////////////////
    // DFB multi-bindings
    ////////////////////////////////////////////////////////////////////////////////

    // A DFB is a software FIFO. By default, a DFB instance (i.e., a DFB on a
    // particular node) must have exactly one producer kernel instance and exactly
    // one consumer kernel instance.
    //
    // This invariant holds at the INSTANCE level; a *DFBSpec* (spanning multiple
    // nodes) can have more than one KernelSpec producer or more consumer bindings,
    // as long as every node's DFB instance has one producer and one consumer.
    // (This enables, for example, a grid-spanning compute kernel to be fed data by
    // different types of producer kernels on different nodes.)
    //
    // "Multi-binding" refers to a DFB instance that has more than one producer
    // and/or more than one consumer kernel instance. Gen1 hardware (Wormhole and
    // Blackhole) is technically capable of supporting multi-binding: a DFB lowers
    // to a plain circular buffer there, so the FIFO pointers are shared L1 state
    // that any number of producer/consumer RISCs can drive.
    // However, this configuration is unsafe and its use is discouraged.
    //
    // CAUTION:
    // Multi-binding a DFB instance is UNSAFE in most circumstances.
    // The kernel logic must explicitly ensure access safety and synchronization.
    // Multi-binding forfeits the protections of the FIFO synchronization mechanics.
    // There is a high likelihood of race conditions and non-deterministic behavior.
    //
    // NOTE:
    // This feature is included for backwards compatibility with legacy APIs.
    // It is NOT supported on Gen2 architectures: setting this flag on a Gen2
    // target is a hard error, whether or not any instance is actually multi-bound.
    bool allow_instance_multi_binding = false;
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

}  // namespace tt::tt_metal::experimental
