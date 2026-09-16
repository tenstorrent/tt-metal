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
// not bound to the Program. The ProgramSpec names the pipe's GEOMETRY (sender,
// receivers, ring size, entry size); the actual PrefetcherPipe object is supplied
// at execution time via ProgramRunArgs::prefetcher_pipe_args and must match this
// geometry.
//
// ROLE: A kernel's role (sender or receiver) is DERIVED from the node coverage of
//   the WorkUnitSpecs that place it, compared against this parameter's geometry.
//   A kernel placed on the sender node acts as the pipe's sender; a kernel placed
//   on the receiver nodes acts as a receiver. Roles are complete: a kernel that
//   binds the pipe must cover either none or all of the receiver nodes, and either
//   not the sender node or exactly it. This lets the sender and the receivers live
//   in different Programs (e.g. a prefetcher op and a consumer op) that share one
//   pipe through their run args.
//
// KERNEL ACCESS: A data-movement kernel binds the pipe via
//   KernelSpec::prefetcher_pipe_bindings and constructs the device object from the
//   emitted token:
//     experimental::PrefetcherPipe pipe(pipe::<accessor_name>);
//   Compute kernels never bind a pipe directly. A receiver-side data-movement kernel
//   forwards pipe entries to compute through a RELAY DFB: a DataflowBufferSpec that
//   lists this parameter in its `prefetcher_pipe_relays`, aliasing the pipe's ring
//   as its backing storage. Compute binds that DFB like any other.
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

    // Geometry: one sender node feeding a set of receiver nodes. The sender must not be one of
    // the receivers, and the receiver set must be non-empty.
    NodeCoord sender;
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
