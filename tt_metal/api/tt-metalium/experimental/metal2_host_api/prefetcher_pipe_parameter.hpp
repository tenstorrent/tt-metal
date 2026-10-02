// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  PrefetcherPipeParameter API
// ============================================================================
//
// A PrefetcherPipeParameter declares that a Program participates in a
// PrefetcherPipe: a durable, user-managed 1:N data ring that carries entries
// from one sender node to a set of receiver nodes over the NoC, with credit
// flow control at both ends.
//
// Like a TensorParameter, the pipe is a user-managed resource whose lifetime is
// not bound to the Program. The ProgramSpec names the pipe's GEOMETRY (receivers,
// ring size, entry size) in ProgramAdvancedOptions::prefetcher_pipe_parameters; the
// actual PrefetcherPipe object is supplied at execution time via
// AdvancedProgramRunArgs::prefetcher_pipe_args and must match this geometry.
//
// PrefetcherPipe is a separate experimental effort from the core Metal 2.0 spec, so
// every piece of its spec surface lives in the *AdvancedOptions structs
// (advanced_options.hpp); the core KernelSpec / DataflowBufferSpec / ProgramSpec /
// ProgramRunArgs do not depend on it.
//
// The sender is NOT part of the parameter. It is a property of the pipe object (and
// of its config page): a consumer Program never needs to know where the data comes
// from, and for a DRAM-resident sender there is no worker node to name. A Program
// that runs the sender kernel states the sender node once, as that kernel's
// WorkUnitSpec target; the pipe supplied at run time must have its sender there.
//
// KERNEL ACCESS: A data-movement kernel binds pipes via
//   KernelAdvancedOptions::prefetcher_pipe_bindings and constructs the device object from the
//   emitted token:
//     experimental::PrefetcherPipe pipe(pipe::<accessor_name>);
//   One binding (accessor) may name SEVERAL parameters: on every node the kernel runs
//   on exactly one of them is present, and the host resolves which per node, so one
//   compiled kernel drives e.g. N sender cores that each own a 1:N pipe, or a receiver
//   grid fed by several senders. Pipes sharing an accessor must agree on ring_size and
//   entry_size.
//   Compute kernels never bind a pipe directly. A receiver-side data-movement kernel
//   forwards pipe entries to compute through a RELAY DFB: a DataflowBufferSpec whose
//   DFBAdvancedOptions::prefetcher_pipe_relays names the same pipe group as that kernel's accessor,
//   aliasing the ring as its backing storage. Compute binds that DFB like any other.
//
// ROLE: A kernel's role (sender or receiver) is DERIVED from the node coverage of
//   the WorkUnitSpecs that place it, compared against the receiver sets of the pipes
//   in the accessor group. The kernel's node set must equal EITHER the disjoint union
//   of the group's receiver sets (it is a receiver of every pipe in the group) OR be
//   disjoint from all of them with exactly one node per pipe (it is the sender of
//   every pipe in the group; which node hosts which pipe is settled when the pipes
//   are supplied, from each pipe's sender). Partial coverage of the receiver side,
//   or a mix of the two, is rejected. Per pipe, at most one kernel plays each role, so
//   exactly one kernel instance owns the credit counters on each node. Roles may be
//   split across Programs (e.g. a prefetcher op and a consumer op) that share one pipe
//   through their run args.
//
// MULTI-THREADED RECEIVERS (Gen2/Quasar): the receiver-side kernel's num_threads
//   selects how many credit lanes the pipe uses on the receivers. It must divide the
//   ring's entry count (ring_size / entry_size), and must not exceed the pipe's lane
//   capacity. When a relay DFB is present, its PRODUCER kernel is that receiver kernel.
//
// ============================================================================

// A name identifying a PrefetcherPipeParameter within a ProgramSpec.
using PrefetcherPipeParamName = ttsl::StrongType<std::string, struct PrefetcherPipeParamNameTag>;

struct PrefetcherPipeParameter {
    // Pipe identifier: used to reference this pipe within the ProgramSpec
    PrefetcherPipeParamName unique_id;

    // Geometry: the non-empty set of receiver nodes. The sender is the pipe object's, not the
    // Program's, to declare (see the ROLE note above).
    Nodes receivers;

    // Size in bytes of the data ring on every participating node.
    uint32_t ring_size = 0;

    // Dense entry size in bytes for this Program's use of the ring. Must be a multiple of the
    // L1 alignment and at most ring_size. It must divide ring_size when the receiver kernel is
    // multi-threaded or a relay DFB names this pipe (the relay then has ring_size / entry_size
    // entries of this size).
    uint32_t entry_size = 0;
};

}  // namespace tt::tt_metal::experimental
