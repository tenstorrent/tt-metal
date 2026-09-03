// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Impl-internal plumbing for DRAM-sender PrefetcherPipes: the placement one pipe needs, the
// friend shim that reaches PrefetcherPipe's private DRAM-sender constructor, and the DRISC L1
// accessor the Tensor prefetcher manager reads back. All of it is consumed only inside
// tt_metal/, so it lives here rather than on the public experimental surface in
// tt-metalium/experimental/prefetcher_pipe.hpp, which keeps only what ttnn consumes.

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

#include "impl/dataflow_buffer/prefetcher_pipe.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

namespace prefetcher_pipe_dram_sender {

// Where a DRAM-sender pipe's sender lives and what its DRISC L1 block must say. The block address
// comes from a DriscL1Allocation held by the TensorPrefetcherPipes aggregate that owns the pipe,
// so the range outlives the pipe by construction.
struct DramSenderPlacement {
    // DRAM-logical coord of the sender core (x == bank id).
    CoreCoord sender_logical;
    // Offset of the sender block in DRISC L1. Uniform across banks: every sender plants its
    // prefix + config page at the same L1 offset.
    DeviceAddr drisc_block_addr = 0;
    // Bank-local slab index of this sender's first receiver, so a bank's two senders can split
    // its receiver set.
    uint32_t recv_index_base = 0;
    // Largest receiver count across the sibling pipes sharing the DRISC block, not this pipe's
    // own receiver count: the prefetcher request page uses it as the uniform per-tensor
    // layout-slot stride.
    uint32_t max_num_receivers = 0;
};

struct PrefetcherPipeDramSenderInternals {
    // Construct a PrefetcherPipe whose sender is the programmable DRAM core named by `placement`.
    static std::shared_ptr<PrefetcherPipe> make_dram_sender(
        distributed::MeshDevice* mesh_device,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        uint32_t fixed_entry_size,
        const DramSenderPlacement& placement,
        BufferType buffer_type);

    // DRISC L1 base of this pipe's sender block: a PrefetcherPipeDramSenderState prefix followed
    // by the sender's config page (9-word header, receiver NOC XY table, and the per-receiver
    // entries_sent/entries_acked pairs). Pre-written by the constructor on every
    // (device, sender_core) at a uniform offset. The DRISC kernel reads the prefix for its slab
    // base, then hands the config-page address to setup_prefetcher_pipe_interface.
    //
    // Layout: tt_metal/impl/buffers/prefetcher_pipe_dram_sender_state.hpp. Zero for
    // worker-sender pipes.
    static DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe);
};

}  // namespace prefetcher_pipe_dram_sender

class TensorPrefetcherPipes;

// DRISC L1 base of the one sender block every pipe in `pipes` writes its prefix + config page
// into, at a uniform offset across banks. The Tensor prefetcher stamps it into each request's
// header so the DRISC kernel can find the target's state.
DeviceAddr sender_state_drisc_l1_base(const TensorPrefetcherPipes& pipes);

}  // namespace experimental
}  // namespace tt::tt_metal
