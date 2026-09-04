// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Impl-internal plumbing for DRAM-sender PrefetcherPipes: the friend shim that reaches
// PrefetcherPipe's private DRAM-sender constructor, and the DRISC L1 accessor the Tensor
// prefetcher manager stamps into its requests. Both are consumed only inside tt_metal/, so they
// live here rather than on the public experimental surface in
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

struct PrefetcherPipeDramSenderInternals {
    // Construct a PrefetcherPipe whose sender is the programmable DRAM core
    // `dram_sender_logical` (a DRAM-logical coord, x == bank id).
    static std::shared_ptr<PrefetcherPipe> make_dram_sender(
        distributed::MeshDevice* mesh_device,
        CoreCoord dram_sender_logical,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        uint32_t initial_entry_size,
        BufferType buffer_type);

    // DRISC L1 address of this pipe's sender config page (9-word header, receiver NOC XY table,
    // and the per-receiver entries_sent/entries_acked pairs). Pre-written by the constructor on
    // every device, at an offset reserved on this pipe's sender core alone -- so sibling pipes on
    // other banks may report the same address. The DRISC kernel hands it straight to
    // setup_prefetcher_pipe_interface.
    //
    // Zero for worker-sender pipes.
    static DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe);
};

}  // namespace prefetcher_pipe_dram_sender

// DRISC L1 address of `pipe`'s sender config page. The Tensor prefetcher stamps it into the header
// of every request routed to that pipe's sender, so the DRISC kernel can find its endpoint state.
DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe);

}  // namespace experimental
}  // namespace tt::tt_metal
