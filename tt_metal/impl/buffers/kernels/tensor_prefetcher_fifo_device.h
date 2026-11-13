// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device side of the request FIFO shared by the Tensor prefetcher kernels that fast dispatch runs on a
// dispatch engine. Layout: tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp.

#include <stdint.h>

#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "internal/prefetcher_pipe_init.h"
#include "internal/risc_attribs.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

namespace tensor_prefetcher {

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherLayoutMode;
using tt::tt_metal::TensorPrefetcherRequestHeader;
using tt::tt_metal::TensorPrefetcherTensorLayout;

// Every word in the region is shared with the host, the other kernel or the NoC, none of which snoop
// this core's data cache, so it is always accessed through the uncached L1 alias.
template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* region_ptr(uint32_t region_base, uint32_t offset) {
#ifdef ARCH_QUASAR
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset + MEM_L1_UNCACHED_BASE);
#else
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset);
#endif
}

// Blocks of receiver `i` the reader has staged, and the pusher has pushed (freeing their stage slots).
FORCE_INLINE volatile tt_l1_ptr uint32_t* staged_ptr(uint32_t region_base, uint32_t i) {
    return region_ptr<uint32_t>(region_base, fifo::kStagedOffset + i * fifo::kLineBytes);
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* pushed_ptr(uint32_t region_base, uint32_t i) {
    return region_ptr<uint32_t>(region_base, fifo::kPushedOffset + i * fifo::kLineBytes);
}

// One kernel's view of the FIFO: it retires slots in order and publishes how many it has retired.
class FifoConsumer {
public:
    FORCE_INLINE FifoConsumer(uint32_t region_base, uint32_t rd_offset) :
        region_base_(region_base),
        terminate_(region_ptr<uint32_t>(region_base, fifo::kTerminateOffset)),
        wr_(region_ptr<uint32_t>(region_base, fifo::kFifoWrOffset)),
        rd_(region_ptr<uint32_t>(region_base, rd_offset)),
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
        return region_ptr<fifo::SlotHeader>(region_base_, fifo::slot_offset(rd_count_));
    }

    FORCE_INLINE void retire_slot() { *rd_ = ++rd_count_; }

private:
    uint32_t region_base_;
    volatile tt_l1_ptr uint32_t* terminate_;
    volatile tt_l1_ptr uint32_t* wr_;
    volatile tt_l1_ptr uint32_t* rd_;
    uint32_t rd_count_;
};

// Retires every slot the host writes until termination is requested. `serve_request(requests, slot)`
// handles a request slot and returns false if termination was requested before it finished.
template <typename ServeRequest>
FORCE_INLINE void serve_requests(uint32_t region_base, uint32_t rd_offset, ServeRequest&& serve_request) {
    FifoConsumer requests(region_base, rd_offset);
    while (volatile tt_l1_ptr fifo::SlotHeader* slot = requests.wait_for_slot()) {
        switch (slot->command) {
            case fifo::kSlotNop: break;
            case fifo::kSlotRequest:
                if (!serve_request(requests, slot)) {
                    return;
                }
                break;
            default: ASSERT(false); break;
        }
        requests.retire_slot();
    }
}

// One lane's request page, read in place: each field is loaded through the slot's uncached alias when
// used, so a kernel's stack does not grow with the number of lanes a slot can carry.
class LanePage {
public:
    FORCE_INLINE LanePage(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t lane) :
        bytes_(
            reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot) + fifo::kSlotHeaderBytes +
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

// How one tensor of a PREFETCH slot streams: every receiver of every lane, numbered in lane order,
// with the DRAM bank and bank-local address its slab starts at. The reader and the pusher derive
// the same plan from the same slot, so a receiver number names the same stage slots for both.
struct TensorStream {
    uint32_t num_receivers = 0;
    uint32_t block_bytes = 0;
    uint32_t block_count = 0;
    // Bytes of each stage slot, set by the receiver count alone (fifo::stage_slot_bytes).
    uint32_t slot_bytes = 0;
    uint8_t lane[fifo::kMaxReceivers];
    uint8_t receiver_in_lane[fifo::kMaxReceivers];
    uint32_t bank[fifo::kMaxReceivers];
    uint32_t slab_base[fifo::kMaxReceivers];

    // Stage slot `k` (0 .. kStageSlotsPerReceiver) of receiver `i`.
    uint32_t stage_slot_addr(uint32_t stage_ring_base, uint32_t i, uint32_t k) const {
        return stage_ring_base + (i * fifo::kStageSlotsPerReceiver + k) * slot_bytes;
    }
};

// Plans tensor `e` of a PREFETCH slot against a stage ring of `stage_ring_size` bytes. The host serializes
// one page per lane for the same tensors, so every lane agrees on the geometry and differs only in its
// bank, its receivers and their slabs.
FORCE_INLINE void plan_tensor(
    volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, uint32_t stage_ring_size, TensorStream& stream) {
    const uint32_t num_lanes = slot->num_lanes;
    stream.num_receivers = 0;
    for (uint32_t l = 0; l < num_lanes; ++l) {
        const LanePage page(slot, l);
        volatile tt_l1_ptr TensorPrefetcherEntry* entry = page.entry(e);
        volatile tt_l1_ptr TensorPrefetcherTensorLayout* layout = page.layout_of(entry);
        // Receiver-contiguous, batched, ungrouped and unselected: the host rejects the rest for
        // this placement.
        ASSERT(
            layout->layout_mode == static_cast<uint32_t>(TensorPrefetcherLayoutMode::ReceiverContiguous) &&
            layout->streaming == 0 && entry->num_groups == 1);
        const uint32_t block_bytes = layout->page_bytes_per_recv;
        const uint32_t block_count = layout->block_count;
        if (l == 0) {
            stream.block_bytes = block_bytes;
            stream.block_count = block_count;
        }
        ASSERT(stream.block_bytes == block_bytes && stream.block_count == block_count);
        const uint32_t recv_stride = layout->recv_stride_bytes;
        const uint32_t slab_base = entry->bank_local_base + layout->recv_index_base * recv_stride;
        // The lane's pipe receivers, from the sender config page the host stamped into this core's L1.
        const uint32_t lane_receivers = experimental::load_prefetcher_pipe_config_word(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page.header()->prefetch.target_config_page_addr),
            REMOTE_DFB_CFG_NUM_RECEIVERS);
        for (uint32_t r = 0; r < lane_receivers; ++r) {
            const uint32_t i = stream.num_receivers++;
            ASSERT(i < fifo::kMaxReceivers);
            stream.lane[i] = l;
            stream.receiver_in_lane[i] = r;
            stream.bank[i] = slot->lane_bank_id[l];
            stream.slab_base[i] = slab_base + r * recv_stride;
        }
    }
    stream.slot_bytes = fifo::stage_slot_bytes(stage_ring_size, stream.num_receivers);
    // The host checks the fit at queue time.
    ASSERT(stream.block_bytes <= stream.slot_bytes);
}

}  // namespace tensor_prefetcher
