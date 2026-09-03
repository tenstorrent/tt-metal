// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental DRAM-sender extension to PrefetcherPipe: durable remote dataflow buffers whose
// sender is a programmable DRAM core (a Blackhole DRISC) rather than a worker core. This is the
// delivery target the Tensor prefetcher streams into as an alternative to a DRAM-sender
// GlobalCircularBuffer.
//
// A PrefetcherPipe has exactly one sender, so one prefetcher target is a list of them -- one per
// DRISC sender core, built from the mapping BuildTensorPrefetcherSenderMapping returns. The list,
// and whatever the caller wants to bundle with it, is the caller's own concept; this header
// supplies the one-pipe pieces and the accessors a caller needs through the forward declaration
// (ttnn includes no impl/ headers).
//
// Consumers are unchanged from an ordinary PrefetcherPipe: the consumer program calls
// AttachPrefetcherPipe on each pipe's receiver cores and its kernels use the device-side
// experimental::PrefetcherPipe (wait_front / get_read_ptr / pop_front). Only the producer side
// differs, and it is owned by the prefetcher.
//
// Experimental: no API-stability guarantee. Everything here may change or be removed.

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipe;

// Place the Tensor prefetcher's DRAM sender cores for a (bank id -> receivers) request, returning
// one (DRAM-logical sender core, its receivers) pair per pipe to create.
//
// Placement, the receiver split, and slab numbering are the same ones
// CreateGlobalCircularBufferForTensorPrefetcher uses, so a tensor laid out for one transport is
// laid out for the other. With `dual_senders_per_bank`, a bank whose receiver set has more than
// one core is driven by two DRISC senders that split it ceil/floor -- which requires a
// receiver-contiguous layout, where no shard feeds more than one receiver.
//
// The returned order is semantic and must be preserved: a bank's senders are adjacent, and the
// first of them owns the bank's leading receivers (bank-local slab index 0).
std::vector<std::pair<CoreCoord, CoreRangeSet>> BuildTensorPrefetcherSenderMapping(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank);

// Create one PrefetcherPipe driven by the programmable DRAM core `dram_sender_logical`, holding
// `num_entries` entries of `entry_size` bytes per receiver. Pass a (sender, receivers) pair from
// BuildTensorPrefetcherSenderMapping.
//
// `entry_size` is the per-receiver push granularity and must equal the streamed tensor's
// per-receiver page size: a DRAM sender is never dispatched to and so cannot answer a
// receiver-side resize. The pipe is stamped with it, and both AttachPrefetcherPipe and
// QueueTensorPrefetcherRequest reject any other size (with the offending values in the message).
//
// The ring comes from the persistent L1 arena, which refuses a core a live Program has sealed with
// its own local circular buffers. Create the pipes before running any op on the receiver cores --
// under ttnn's program cache a cached op keeps its Program, and its seal, alive.
//
// Keep the returned pipe alive for as long as any program has Attached it or the prefetcher may
// still deliver into it: an attached Program holds a non-owning pointer to it, and destroying it
// frees the ring and config pages.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config page lives on MeshDeviceImpl.
std::shared_ptr<PrefetcherPipe> CreatePrefetcherPipeForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1);

// Accessors usable through the forward declaration above, for callers that hold pipes but do not
// include the impl header. Each returns the same value as the member of the same name.
CoreCoord prefetcher_pipe_sender_core(const PrefetcherPipe& pipe);
const CoreRangeSet& prefetcher_pipe_receiver_cores(const PrefetcherPipe& pipe);
uint32_t prefetcher_pipe_ring_size(const PrefetcherPipe& pipe);
uint32_t prefetcher_pipe_fixed_entry_size(const PrefetcherPipe& pipe);
SenderCoreType prefetcher_pipe_sender_core_type(const PrefetcherPipe& pipe);

// Attach one PrefetcherPipe to `program` on `cores`, returning the program-local slot id kernels
// name. Re-declared here (against the forward declaration above) so a consumer op can Attach
// without including the impl header. Same function as the one in
// impl/dataflow_buffer/prefetcher_pipe.hpp.
uint8_t AttachPrefetcherPipe(
    Program& program, PrefetcherPipe& prefetcher_pipe, const CoreRangeSet& cores, uint32_t entry_size);

}  // namespace experimental
}  // namespace tt::tt_metal
