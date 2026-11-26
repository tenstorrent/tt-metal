// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device side of the request FIFO shared by the Tensor prefetcher kernels that fast dispatch runs on a
// dispatch engine. Layout: tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp.
//
// Compile-time defines, which fast dispatch gives every one of those kernels (TensorPrefetcherKernel):
//   TENSOR_PREFETCHER_REGION_BASE        L1 base of the region
//   TENSOR_PREFETCHER_LANES              lanes per request slot (fifo::RegionLayout)
//   TENSOR_PREFETCHER_KERNEL_STATE_BYTES all the kernels' own state (fifo::RegionLayout)
//   TENSOR_PREFETCHER_STAGE_RING_SIZE    bytes of the region's stage ring
//   TENSOR_PREFETCHER_KERNEL_INDEX       this kernel's index in tensor_prefetcher_kernels(): its FIFO consumer
//   TENSOR_PREFETCHER_STATE_BASE         L1 base of this kernel's own state, zeroed before launch
//   TENSOR_PREFETCHER_STATE_SIZE         bytes of this kernel's own state
//   TENSOR_PREFETCHER_TRID_OFFSET        where this kernel's NoC transaction ids start in the Tensor prefetcher's
//                                        range (dispatch_noc_trids::kTensorPrefetcher)
//   TENSOR_PREFETCHER_TRIDS              how many NoC transaction ids this kernel has (at least 1)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "internal/prefetcher_pipe_init.h"
#include "internal/risc_attribs.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"
#include "tt_metal/impl/dispatch/kernels/dispatch_noc_trids.hpp"

namespace tensor_prefetcher {

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;

using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherLayoutMode;
using tt::tt_metal::TensorPrefetcherRequestHeader;
using tt::tt_metal::TensorPrefetcherTensorLayout;

// This kernel's NoC transaction ids, its range in dispatch_noc_trids.hpp. Each of its reads and writes carries one,
// so it waits on its own transactions instead of every untagged transaction on the dispatch engine, which includes
// the command queues' traffic, and it can wait on some of its transactions while others stay in flight.
inline constexpr dispatch_noc_trids::TridRange kTrids =
    dispatch_noc_trids::tensor_prefetcher_trids(TENSOR_PREFETCHER_TRID_OFFSET, TENSOR_PREFETCHER_TRIDS);
static_assert(kTrids.size() >= 1 && kTrids.end() <= dispatch_noc_trids::kTensorPrefetcher.end());

// Groups of this kernel's transactions in flight, one per id, retired in the order they were issued: slot k's
// transactions carry kTrids[k], and the oldest group is in slot `oldest`.
struct TridQueue {
    static constexpr uint32_t kSlots = kTrids.size();
    uint32_t oldest = 0;
    uint32_t in_flight = 0;

    FORCE_INLINE bool full() const { return in_flight == kSlots; }
    // The slot the next group goes in.
    FORCE_INLINE uint32_t next() const {
        const uint32_t k = oldest + in_flight;
        return k < kSlots ? k : k - kSlots;
    }
    FORCE_INLINE void push() { ++in_flight; }
    FORCE_INLINE void pop() {
        oldest = oldest + 1 == kSlots ? 0 : oldest + 1;
        --in_flight;
    }
};

// Every word in the region is shared with the host, the other kernels or the NoC, none of which snoop
// this core's data cache, so it is always accessed through the uncached L1 alias.
template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* region_ptr(uint32_t region_base, uint32_t offset) {
#ifdef ARCH_QUASAR
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset + MEM_L1_UNCACHED_BASE);
#else
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset);
#endif
}

// One dispatch engine's region as a kernel sees it: its L1 base and its layout (both compile-time).
struct Region {
    uint32_t base = 0;
    fifo::RegionLayout layout;

    // Bytes of receiver `i`'s share of the stage ring the reader has staged, and the pusher has pushed (so the reader
    // may stage into them again), each counting the gaps that blocks skip at the end of the share.
    FORCE_INLINE volatile tt_l1_ptr uint32_t* staged(uint32_t i) const {
        return region_ptr<uint32_t>(base, layout.staged_offset() + i * fifo::kLineBytes);
    }
    FORCE_INLINE volatile tt_l1_ptr uint32_t* pushed(uint32_t i) const {
        return region_ptr<uint32_t>(base, layout.pushed_offset() + i * fifo::kLineBytes);
    }
    FORCE_INLINE volatile tt_l1_ptr fifo::SlotHeader* slot(uint32_t slot_count) const {
        return region_ptr<fifo::SlotHeader>(base, layout.slot_offset(slot_count));
    }
    constexpr uint32_t stage_ring_base() const { return base + layout.stage_ring_offset(); }
};

// This kernel's dispatch engine region and its parts.
inline constexpr Region region{
    .base = TENSOR_PREFETCHER_REGION_BASE,
    .layout = fifo::RegionLayout{
        .lanes = TENSOR_PREFETCHER_LANES, .kernel_state_bytes = TENSOR_PREFETCHER_KERNEL_STATE_BYTES}};
inline constexpr uint32_t stage_ring_base = region.stage_ring_base();
inline constexpr uint32_t stage_ring_size = TENSOR_PREFETCHER_STAGE_RING_SIZE;
inline constexpr uint32_t kernel_state_base = TENSOR_PREFETCHER_STATE_BASE;
inline constexpr uint32_t kernel_state_size = TENSOR_PREFETCHER_STATE_SIZE;

// This kernel's view of the FIFO: it retires slots in order and publishes how many it has retired.
class FifoConsumer {
public:
    FORCE_INLINE FifoConsumer() :
        terminate_(region_ptr<uint32_t>(region.base, fifo::kTerminateOffset)),
        wr_(region_ptr<uint32_t>(region.base, fifo::kFifoWrOffset)),
        rd_(region_ptr<uint32_t>(region.base, fifo::fifo_rd_offset(TENSOR_PREFETCHER_KERNEL_INDEX))),
        rd_count_(*rd_) {}

    FORCE_INLINE bool terminate_requested() const { return *terminate_ != 0; }

    // Waits for the next unretired slot. Returns nullptr if termination was requested first.
    FORCE_INLINE volatile tt_l1_ptr fifo::SlotHeader* wait_for_slot() const {
        WAYPOINT("TFW");
        while (*wr_ == rd_count_) {
            if (terminate_requested()) {
                return nullptr;
            }
        }
        WAYPOINT("TFD");
        return region.slot(rd_count_);
    }

    FORCE_INLINE void retire_slot() { *rd_ = ++rd_count_; }

    // The slot after the one being served, if the host has written it already; nullptr otherwise.
    FORCE_INLINE volatile tt_l1_ptr fifo::SlotHeader* next_slot() const {
        return *wr_ - rd_count_ > 1 ? region.slot(rd_count_ + 1) : nullptr;
    }

private:
    volatile tt_l1_ptr uint32_t* terminate_;
    volatile tt_l1_ptr uint32_t* wr_;
    volatile tt_l1_ptr uint32_t* rd_;
    uint32_t rd_count_;
};

// One lane's request page, read in place: each field is loaded through the slot's uncached alias when
// used, so a kernel's stack does not grow with the number of lanes a slot can carry.
class LanePage {
public:
    FORCE_INLINE LanePage(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t lane) :
        bytes_(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot) + fifo::kSlotPagesOffset +
            lane * fifo::kLanePageBytes) {}

    FORCE_INLINE volatile tt_l1_ptr TensorPrefetcherRequestHeader* header() const {
        return reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherRequestHeader*>(bytes_);
    }
    // The words after the header: a PREFETCH page's entries, or a DRAIN page's target config pages.
    FORCE_INLINE volatile tt_l1_ptr uint32_t* body_words() const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bytes_ + sizeof(TensorPrefetcherRequestHeader));
    }
    FORCE_INLINE volatile tt_l1_ptr TensorPrefetcherEntry* entry(uint32_t e) const {
        return reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherEntry*>(body_words()) + e;
    }
    FORCE_INLINE volatile tt_l1_ptr TensorPrefetcherTensorLayout* layout_of(
        volatile tt_l1_ptr TensorPrefetcherEntry* entry) const {
        return reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherTensorLayout*>(bytes_ + entry->layout_offset);
    }

private:
    volatile tt_l1_ptr uint8_t* bytes_;
};

// The first kernel in tensor_prefetcher_kernels() plans each tensor of a PREFETCH slot: it reads the tensor's
// source and decides how many blocks each receiver gets, which it publishes in the slot's plan. Every later
// kernel follows that plan (wait_for_tensor_blocks) rather than derive it, so a group selector is read once.
inline constexpr bool kPlansBlocks = TENSOR_PREFETCHER_KERNEL_INDEX == 0;

// The word of `slot`'s plan that says how many blocks each receiver gets of tensor `e`.
FORCE_INLINE volatile tt_l1_ptr uint32_t* tensor_blocks(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e) {
    ASSERT(e < fifo::kMaxSlotTensors);
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
               reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot) + fifo::kSlotHeaderBytes) +
           e;
}

// Waits for the kernel that reads tensor `e` of `slot` from its source to plan how many blocks each receiver
// gets of it, and returns that count in `blocks`. `idle()` is the caller's own work while it waits, and returns
// whether it made progress; termination is checked only when it made none. False if termination was requested first.
template <typename Idle>
FORCE_INLINE bool wait_for_tensor_blocks(
    const FifoConsumer& requests,
    volatile tt_l1_ptr fifo::SlotHeader* slot,
    uint32_t e,
    uint32_t& blocks,
    Idle&& idle) {
    volatile tt_l1_ptr uint32_t* planned = tensor_blocks(slot, e);
    WAYPOINT("TBW");
    while ((blocks = *planned) == fifo::kTensorBlocksUnplanned) {
        if (!idle() && requests.terminate_requested()) {
            return false;
        }
    }
    WAYPOINT("TBD");
    return true;
}
FORCE_INLINE bool wait_for_tensor_blocks(
    const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, uint32_t& blocks) {
    return wait_for_tensor_blocks(requests, slot, e, blocks, [] { return false; });
}

// Waits until the signal slot a WAIT_SIGNAL page names reaches its count. The slots only grow and may wrap, so
// their difference is compared. False if termination was requested first.
FORCE_INLINE bool wait_for_signal(
    const FifoConsumer& requests, volatile tt_l1_ptr TensorPrefetcherRequestHeader* header) {
    volatile tt_l1_ptr uint32_t* signal =
        region_ptr<uint32_t>(region.base, fifo::signal_slot_offset(header->wait_signal.slot_index));
    const uint32_t count = header->wait_signal.wait_value;
    WAYPOINT("TSW");
    while (static_cast<int32_t>(*signal - count) < 0) {
        if (requests.terminate_requested()) {
            return false;
        }
    }
    WAYPOINT("TSD");
    return true;
}

// Puts this kernel's NoC command buffers back on id 0, as a command buffer's id must be 0 at kernel exit. Each
// keeps the last id given to it: the read command buffer the one the reader last read with, the write command
// buffer the one the pusher's last tagged write carried.
FORCE_INLINE void clear_noc_trids() {
    noc_async_read_set_trid(0);
    noc_async_write_set_trid(0);
}

// Retires every slot the host writes, as this kernel's FIFO consumer, until termination is requested.
// `serve_request(requests, slot)` handles a PREFETCH or DRAIN request slot and returns false if termination
// was requested before it finished. Every kernel waits out a WAIT_SIGNAL slot here, so no kernel reads or
// sends anything the requests after it name until the signal arrives.
template <typename ServeRequest>
FORCE_INLINE void serve_requests(ServeRequest&& serve_request) {
    // Every read this kernel issues carries one of its ids, which the read command buffer keeps until the next is set.
    noc_async_read_set_trid(kTrids[0]);
    FifoConsumer requests;
    while (volatile tt_l1_ptr fifo::SlotHeader* slot = requests.wait_for_slot()) {
        ASSERT(slot->num_lanes <= region.layout.lanes);
        switch (slot->command) {
            case fifo::kSlotNop: break;
            case fifo::kSlotRequest: {
                ASSERT(slot->num_lanes > 0);
                volatile tt_l1_ptr TensorPrefetcherRequestHeader* header = LanePage(slot, 0).header();
                const bool served = header->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_WAIT_SIGNAL
                                        ? wait_for_signal(requests, header)
                                        : serve_request(requests, slot);
                if (!served) {
                    clear_noc_trids();
                    return;
                }
                break;
            }
            default: ASSERT(false); break;
        }
        requests.retire_slot();
    }
    clear_noc_trids();
}

// How a tensor's blocks lie in a receiver's share of the stage ring.
struct StageGeometry {
    uint32_t block_bytes = 0;
    // Bytes of each receiver's share of the stage ring, set by the receiver count alone (fifo::stage_share_bytes).
    uint32_t share_bytes = 0;
    // Bytes each block takes in its receiver's share (fifo::stage_block_footprint).
    uint32_t block_footprint = 0;
};

// How one tensor of a PREFETCH slot streams: every receiver of every lane, numbered in lane order. The
// kernels derive the same plan from the same slot, so a receiver number names the same stage share for all.
struct TensorStream : StageGeometry {
    uint32_t num_receivers = 0;
    // Blocks of each of the tensor's groups per receiver; a group selector decides how many groups arrive.
    uint32_t block_count = 0;
    uint8_t lane[fifo::kMaxReceivers];
    uint8_t receiver_in_lane[fifo::kMaxReceivers];

    // The L1 address `offset` bytes into receiver `i`'s share of the stage ring.
    uint32_t stage_addr(uint32_t i, uint32_t offset) const { return stage_ring_base + i * share_bytes + offset; }

    // The most blocks the kernels move in one NoC transfer: what fits in kMaxRunBytes, and at least one. Moving
    // small blocks together pays a transfer's fixed costs once for all of them; the cap keeps a run from holding
    // back its first block for long.
    uint32_t max_run_blocks() const { return max_run_blocks(block_bytes); }
    static uint32_t max_run_blocks(uint32_t block_bytes) {
        return block_bytes >= kMaxRunBytes ? 1 : kMaxRunBytes / block_bytes;
    }
    static constexpr uint32_t kMaxRunBytes = 256 * 1024;
};

// How many blocks of `footprint` bytes fit in `bytes`, at most `max_blocks`. Divides only when fewer than
// `max_blocks` fit: these sizes change from run to run, and a division costs tens of cycles on a data-movement core.
FORCE_INLINE uint32_t blocks_in(uint32_t bytes, uint32_t footprint, uint32_t max_blocks) {
    return bytes >= max_blocks * footprint ? max_blocks : bytes / footprint;
}

// Where one receiver's blocks go in its share of the stage ring. The reader and the pusher each keep one per receiver
// and place the same blocks in the same order, so they agree on every block's offset. Each starts its shares over
// when the receiver count changes, which happens only while the stage ring is empty.
class StageCursor {
public:
    // The bytes the next block takes: its footprint, and before it the end of the share if it does not fit there.
    FORCE_INLINE uint32_t bytes_taken(const StageGeometry& geometry) const {
        return (wraps(geometry) ? geometry.share_bytes - offset_ : 0) + geometry.block_footprint;
    }
    FORCE_INLINE void reset() { offset_ = 0; }

    // A run is blocks that lie back to back in the share, so one NoC transfer moves them. The most blocks, up to
    // `max_blocks`, a run from the next block can hold: the first may start the share over, and the run stops at
    // the share's end. Blocks lie back to back only when their footprint is their size.
    FORCE_INLINE uint32_t run_blocks(const StageGeometry& geometry, uint32_t max_blocks) const {
        if (geometry.block_footprint != geometry.block_bytes) {
            return 1;
        }
        return blocks_in(geometry.share_bytes - (wraps(geometry) ? 0 : offset_), geometry.block_footprint, max_blocks);
    }
    // The bytes a run of `num_blocks` from the next block takes (bytes_taken for its first block).
    FORCE_INLINE uint32_t run_bytes_taken(const StageGeometry& geometry, uint32_t num_blocks) const {
        return bytes_taken(geometry) + (num_blocks - 1) * geometry.block_footprint;
    }
    // Places a run of `num_blocks` (at most run_blocks) and returns its offset in the share.
    FORCE_INLINE uint32_t place_run(const StageGeometry& geometry, uint32_t num_blocks) {
        const uint32_t at = wraps(geometry) ? 0 : offset_;
        offset_ = at + num_blocks * geometry.block_footprint;
        return at;
    }

private:
    FORCE_INLINE bool wraps(const StageGeometry& geometry) const {
        return offset_ + geometry.block_footprint > geometry.share_bytes;
    }

    uint32_t offset_ = 0;
};

// Plans tensor `e` of a PREFETCH slot. The host serializes one page per lane for the same tensors, so every
// lane agrees on the geometry and differs only in its bank, its receivers and their slabs.
FORCE_INLINE void plan_tensor(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, TensorStream& stream) {
    const uint32_t num_lanes = slot->num_lanes;
    stream.num_receivers = 0;
    for (uint32_t l = 0; l < num_lanes; ++l) {
        const LanePage page(slot, l);
        volatile tt_l1_ptr TensorPrefetcherEntry* entry = page.entry(e);
        volatile tt_l1_ptr TensorPrefetcherTensorLayout* layout = page.layout_of(entry);
        // Receiver-contiguous and batched: the host rejects the rest for this placement.
        ASSERT(
            layout->layout_mode == static_cast<uint32_t>(TensorPrefetcherLayoutMode::ReceiverContiguous) &&
            layout->streaming == 0);
        const uint32_t block_bytes = layout->page_bytes_per_recv;
        const uint32_t block_count = layout->block_count;
        if (l == 0) {
            stream.block_bytes = block_bytes;
            stream.block_count = block_count;
        }
        ASSERT(stream.block_bytes == block_bytes && stream.block_count == block_count);
        // The lane's pipe receivers, from the sender config page the host stamped into this core's L1.
        const uint32_t lane_receivers = experimental::load_prefetcher_pipe_config_word(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page.header()->prefetch.target_config_page_addr),
            REMOTE_DFB_CFG_NUM_RECEIVERS);
        for (uint32_t r = 0; r < lane_receivers; ++r) {
            const uint32_t i = stream.num_receivers++;
            ASSERT(i < fifo::kMaxReceivers);
            stream.lane[i] = l;
            stream.receiver_in_lane[i] = r;
        }
    }
    stream.share_bytes = fifo::stage_share_bytes(stage_ring_size, stream.num_receivers);
    stream.block_footprint = fifo::stage_block_footprint(stream.block_bytes);
    // The host checks the fit at queue time.
    ASSERT(stream.block_bytes <= fifo::max_stage_block_bytes(stream.share_bytes));
}

}  // namespace tensor_prefetcher
