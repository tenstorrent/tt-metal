// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental DRAM-sender extension to PrefetcherPipe: durable remote dataflow buffers whose
// sender is a programmable DRAM core (a Blackhole DRISC) rather than a worker core. This is the
// delivery target the Tensor prefetcher streams into as an alternative to a DRAM-sender
// GlobalCircularBuffer.
//
// A PrefetcherPipe has exactly one sender, and a bank may be driven by two of them, so one
// prefetcher target is a per-bank group of pipes. CreatePrefetcherPipesForTensorPrefetcher places
// the senders and creates every pipe; the caller only says which receivers each bank feeds.
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
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipe;

// The PrefetcherPipes driving one DRAM bank's receivers.
struct TensorPrefetcherBankPipes {
    uint32_t bank_id = 0;
    // One pipe, or two when the bank is split across both of its DRISC sender cores. The split is
    // ceil/floor over the bank's ordered receivers: pipes[0] owns the leading ceil(n/2) receivers
    // (bank-local slab index 0), pipes[1] the rest. That order is what assigns each sender its
    // bank-local slab base, so keep the pipes in it.
    std::vector<std::shared_ptr<PrefetcherPipe>> pipes;
};

// Create the PrefetcherPipes that deliver one Tensor-prefetcher request, one group per
// `bank_to_receivers` entry, in input order. Sender placement, the receiver split, and slab
// numbering are the ones CreateGlobalCircularBufferForTensorPrefetcher uses, so a tensor laid out
// for one transport is laid out for the other.
//
// With `support_multi_receiver_shards` a bank is driven by a single sender, which is what the
// legacy interleaved layout (a shard feeding more than one receiver) requires. Without it — the
// default, matching the receiver-contiguous layout — a bank with more than one receiver gets two
// senders, each pushing roughly half of them.
//
// `entry_size` is the per-receiver push granularity a pipe starts life at, and `num_entries` is how
// many of them a receiver's ring holds; together they fix the ring size, which never changes. A
// later Attach and a later queued tensor may use a different entry size as long as it divides the
// ring: the DRAM sender snaps its write cursor onto the new grid and publishes the skipped bytes as
// pad credits, which is the same resize handshake a worker sender runs.
//
// The rings come from the persistent L1 arena, which refuses a core a live Program has sealed with
// its own local circular buffers. Create the pipes before running any op on the receiver cores --
// under ttnn's program cache a cached op keeps its Program, and its seal, alive.
//
// Keep the returned pipes alive for as long as any program has Attached them or the prefetcher may
// still deliver into them: an attached Program holds a non-owning pointer to each pipe, and
// destroying one frees its ring and config pages.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config pages lives on MeshDeviceImpl.
std::vector<TensorPrefetcherBankPipes> CreatePrefetcherPipesForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = false);

// Accessors usable through the forward declaration above, for callers that hold pipes but do not
// include the impl header. Each returns the same value as the member of the same name.
CoreCoord prefetcher_pipe_sender_core(const PrefetcherPipe& pipe);
const CoreRangeSet& prefetcher_pipe_receiver_cores(const PrefetcherPipe& pipe);
SenderCoreType prefetcher_pipe_sender_core_type(const PrefetcherPipe& pipe);
// Address of this pipe's receiver-side config page. Two live pipes over the same receivers never
// share one, so a consumer op can use it (with the pipe's core mapping) as a cache-key identity
// that distinguishes pipes of identical geometry.
uint32_t prefetcher_pipe_config_address(const PrefetcherPipe& pipe);

// Attach one PrefetcherPipe to `program` on `cores`, returning the program-local slot id kernels
// name. Re-declared here (against the forward declaration above) so a consumer op can Attach
// without including the impl header. Same function as the one in
// impl/dataflow_buffer/prefetcher_pipe.hpp.
uint8_t AttachPrefetcherPipe(
    Program& program, PrefetcherPipe& prefetcher_pipe, const CoreRangeSet& cores, uint32_t entry_size);

// Create a local circular buffer over `prefetcher_pipe`'s ring and register it as that pipe's
// relay, so the pages the sender delivers are read in place by compute through the ordinary CB
// API. `prefetcher_pipe_id` must be the id AttachPrefetcherPipe returned for these cores.
//
// The bridge a compute kernel needs that a DataflowBuffer relay cannot give it: a real
// circular buffer index, which is what a legacy program factory's kernels name.
//
// `config` declares exactly one local buffer index and no remote index, its total size is the
// pipe's ring size, and its page size is the Attached entry size, so one delivered entry is one CB
// page. `core_spec` must be a subset of the pipe's receiver cores.
// The CB is globally allocated at the ring address, so it neither consumes program-local L1 nor
// accepts UpdateDynamicCircularBufferAddress.
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const PrefetcherPipe& prefetcher_pipe,
    uint8_t prefetcher_pipe_id);

}  // namespace experimental
}  // namespace tt::tt_metal
