// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental DRAM-sender extension to PrefetcherPipe: durable remote dataflow buffers whose
// sender is a programmable DRAM core (a Blackhole DRISC) rather than a worker core. This is the
// delivery target the Tensor prefetcher streams into as an alternative to a DRAM-sender
// GlobalCircularBuffer.
//
// A PrefetcherPipe has exactly one sender, so one prefetcher target is a set of pipes -- one per
// DRISC sender core -- held together by TensorPrefetcherPipes, which is what the prefetcher and
// the consumer op both name.
//
// Consumers are unchanged from an ordinary PrefetcherPipe: the consumer program calls
// AttachTensorPrefetcherPipes on its receiver cores and its kernels use the device-side
// experimental::PrefetcherPipe (wait_front / get_read_ptr / pop_front). Only the producer side
// differs, and it is owned by the prefetcher.
//
// Experimental: no API-stability guarantee. Everything here may change or be removed.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class DriscL1Allocation;
class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipe;

// One Tensor-prefetcher delivery target: a PrefetcherPipe per DRAM sender core, all sharing one
// entry size, one ring geometry, and one block in the per-mesh DRISC L1 arena.
//
// Keep it alive for as long as any program has Attached it or the prefetcher may still deliver
// into it: an attached Program holds a non-owning pointer to each pipe, and destroying this frees
// the rings and config pages.
class TensorPrefetcherPipes {
public:
    TensorPrefetcherPipes(const TensorPrefetcherPipes&) = delete;
    TensorPrefetcherPipes& operator=(const TensorPrefetcherPipes&) = delete;
    TensorPrefetcherPipes(TensorPrefetcherPipes&&) = delete;
    TensorPrefetcherPipes& operator=(TensorPrefetcherPipes&&) = delete;
    ~TensorPrefetcherPipes();

    // Per-receiver push granularity, shared by every pipe. Every Attach and every queued tensor
    // must match it: a DRAM sender is never dispatched to and so cannot answer a resize.
    uint32_t entry_size() const { return entry_size_; }
    // Entries a receiver's ring holds.
    uint32_t num_entries() const { return num_entries_; }
    // Bytes of ring per receiver (entry_size * num_entries).
    uint32_t ring_size() const { return ring_size_; }
    // Largest receiver count over the senders; the prefetcher request page's layout-slot stride.
    uint32_t max_num_receivers() const { return max_num_receivers_; }

    // Every receiver across every pipe. This is the core set a consumer program attaches.
    const CoreRangeSet& receiver_cores() const { return receiver_cores_; }
    // Sender core (DRAM-logical, x == bank id) -> its receivers, in the order that fixes bank-local
    // slab numbering. One entry per pipe, in pipe-id order.
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const { return mapping_; }

    size_t num_pipes() const { return pipes_.size(); }
    // The pipe driving mapping entry `index`. Kernels name it by the id AttachTensorPrefetcherPipes
    // returns, not by this index.
    PrefetcherPipe& pipe(size_t index);

private:
    friend std::shared_ptr<TensorPrefetcherPipes> CreatePrefetcherPipesForTensorPrefetcher(
        distributed::MeshDevice& mesh_device,
        const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
        uint32_t entry_size,
        uint32_t num_entries,
        BufferType buffer_type,
        bool support_multi_receiver_shards);
    friend DeviceAddr sender_state_drisc_l1_base(const TensorPrefetcherPipes& pipes);

    TensorPrefetcherPipes(
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping,
        CoreRangeSet receiver_cores,
        std::shared_ptr<DriscL1Allocation> drisc_sender_state_alloc,
        std::vector<std::shared_ptr<PrefetcherPipe>> pipes,
        uint32_t entry_size,
        uint32_t num_entries,
        uint32_t max_num_receivers);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping_;
    CoreRangeSet receiver_cores_;
    // Declared before pipes_ so it is destroyed after them: the pipes' DRISC blocks live inside
    // this range. One allocation for the whole set -- a per-pipe allocation would burn the small
    // DRISC arena zone once per sender.
    std::shared_ptr<DriscL1Allocation> drisc_sender_state_alloc_;
    std::vector<std::shared_ptr<PrefetcherPipe>> pipes_;
    uint32_t entry_size_ = 0;
    uint32_t num_entries_ = 0;
    uint32_t ring_size_ = 0;
    uint32_t max_num_receivers_ = 0;
};

// Build the PrefetcherPipes for a Tensor-prefetcher target: one pipe per programmable DRAM sender
// core, sized to hold `num_entries` entries of `entry_size` bytes per receiver.
//
// Sender placement, the dual-sender receiver split, and slab numbering are identical to
// CreateGlobalCircularBufferForTensorPrefetcher -- the two share build_dram_sender_mapping -- so a
// tensor laid out for one transport is laid out for the other. See that function for what
// `support_multi_receiver_shards` promises about the source layout.
//
// `entry_size` is the per-receiver push granularity and must equal the streamed tensor's
// per-receiver page size: this transport does not resize mid-flight, so every queued tensor and
// every Attach must agree with it (both enforced, with the offending values in the message).
//
// The rings come from the persistent L1 arena, which refuses a core a live Program has sealed with
// its own local circular buffers. Create the pipes before running any op on the receiver cores --
// under ttnn's program cache a cached op keeps its Program, and its seal, alive.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config pages lives on MeshDeviceImpl.
std::shared_ptr<TensorPrefetcherPipes> CreatePrefetcherPipesForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = false);

// Attach every pipe to `program` on its own receiver cores, at the shared entry size. Returns the
// program-local pipe id per mapping entry, in mapping order; a receiver core's kernel takes the id
// of the one pipe it belongs to (as a runtime argument, since one kernel serves receivers of
// different pipes).
std::vector<uint8_t> AttachTensorPrefetcherPipes(Program& program, TensorPrefetcherPipes& pipes);

// Attach one PrefetcherPipe to `program` on `cores`, returning the program-local slot id kernels
// name. Re-declared here (against the forward declaration above) so a consumer op can Attach
// without including the impl header. Same function as the one in
// impl/dataflow_buffer/prefetcher_pipe.hpp.
uint8_t AttachPrefetcherPipe(
    Program& program, PrefetcherPipe& prefetcher_pipe, const CoreRangeSet& cores, uint32_t entry_size);

}  // namespace experimental
}  // namespace tt::tt_metal
