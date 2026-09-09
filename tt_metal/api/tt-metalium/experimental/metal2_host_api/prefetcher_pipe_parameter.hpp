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
// A PrefetcherPipe is a user-managed resource: it owns durable L1 (a data ring plus config pages)
// that outlives any one Program, and a sender outside the Program fills it. A ProgramSpec
// therefore does not hold one; it declares a PrefetcherPipeParameter, and the caller supplies the
// pipe object per execution through ProgramRunArgs::prefetcher_pipe_args, the same way a
// TensorParameter is filled by a MeshTensor.
//
// A parameter is consumed by naming it in DataflowBufferSpec::prefetcher_pipe_relays: that lays
// the DFB over the pipe's ring, so the DFB's consumer reads delivered entries in place.
//
// The declared properties below are what the runtime checks the supplied pipe against, so a spec
// can be validated, laid out and compiled before any pipe exists.

// A name identifying a PrefetcherPipeParameter within a ProgramSpec.
using PrefetcherPipeParamName = ttsl::StrongType<std::string, struct PrefetcherPipeParamNameTag>;

struct PrefetcherPipeParameter {
    // PrefetcherPipe identifier: used to reference this PrefetcherPipe within the ProgramSpec.
    PrefetcherPipeParamName unique_id;

    // The nodes the supplied pipe delivers to. These are the DFB nodes the relay covers, and they
    // must be exactly the pipe's receivers: a pipe's receiver role cannot be split across
    // Programs, so a Program that attaches a pipe attaches all of it.
    NodeRangeSet receiver_nodes;

    // Bytes of ring the supplied pipe holds per receiver. A relay DFB spans the whole ring, so
    // this must be a whole number of the DFB's entries — checked against the DFB at spec
    // validation, and against the pipe when the argument is bound.
    uint32_t ring_size = 0;
};

}  // namespace tt::tt_metal::experimental
