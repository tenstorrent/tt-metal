// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device side of the request FIFO shared by the Tensor prefetcher kernels that fast dispatch runs on a
// dispatch engine. Layout: tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp.

#include <stdint.h>

#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "hostdev/remote_dfb_config_layout.h"
#include "internal/risc_attribs.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

namespace tensor_prefetcher {

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tt::tt_metal::kRequestPageBytes;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherRequestHeader;
using tt::tt_metal::TensorPrefetcherTensorLayout;

static_assert(fifo::kLanePageBytes == kRequestPageBytes, "A FIFO lane holds exactly one request page");

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

    uint32_t retired() const { return rd_count_; }

private:
    uint32_t region_base_;
    volatile tt_l1_ptr uint32_t* terminate_;
    volatile tt_l1_ptr uint32_t* wr_;
    volatile tt_l1_ptr uint32_t* rd_;
    uint32_t rd_count_;
};

// One lane's request page, copied out of its slot so it can be parsed as ordinary memory.
struct LanePage {
    uint32_t words[kRequestPageBytes / sizeof(uint32_t)];

    const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words); }
    const TensorPrefetcherRequestHeader& header() const {
        return *reinterpret_cast<const TensorPrefetcherRequestHeader*>(words);
    }
    const TensorPrefetcherEntry& entry(uint32_t e) const {
        return reinterpret_cast<const TensorPrefetcherEntry*>(bytes() + sizeof(TensorPrefetcherRequestHeader))[e];
    }
    const TensorPrefetcherTensorLayout& layout_of(const TensorPrefetcherEntry& entry) const {
        return *reinterpret_cast<const TensorPrefetcherTensorLayout*>(bytes() + entry.layout_offset);
    }
};

FORCE_INLINE void copy_lane_page(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t lane, LanePage& page) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot) + fifo::kSlotHeaderBytes + lane * fifo::kLanePageBytes);
    for (uint32_t w = 0; w < kRequestPageBytes / sizeof(uint32_t); ++w) {
        page.words[w] = src[w];
    }
}

// How one tensor of a PREFETCH slot streams: every receiver of every lane, numbered in lane order,
// with the DRAM bank and bank-local address its slab starts at. The reader and the pusher derive
// the same plan from the same slot, so a receiver number names the same stage slots for both.
struct TensorStream {
    uint32_t num_receivers = 0;
    uint32_t block_bytes = 0;
    uint32_t block_count = 0;
    uint8_t lane[fifo::kMaxReceivers];
    uint8_t receiver_in_lane[fifo::kMaxReceivers];
    uint32_t bank[fifo::kMaxReceivers];
    uint32_t slab_base[fifo::kMaxReceivers];

    // Stage slot `k` (0 .. kStageSlotsPerReceiver) of receiver `i`.
    uint32_t stage_slot_addr(uint32_t stage_ring_base, uint32_t i, uint32_t k) const {
        return stage_ring_base + (i * fifo::kStageSlotsPerReceiver + k) * block_bytes;
    }
};

// Receivers a lane's pipe has, from the sender config page the host stamped into this core's L1.
FORCE_INLINE uint32_t pipe_num_receivers(uint32_t config_page_addr) {
    return *region_ptr<uint32_t>(config_page_addr, REMOTE_DFB_CFG_NUM_RECEIVERS * sizeof(uint32_t));
}

// Plans tensor `e` of a PREFETCH slot. The host serializes one page per lane for the same tensors,
// so every lane agrees on the geometry and differs only in its bank, its receivers and their slabs.
FORCE_INLINE void plan_tensor(
    volatile tt_l1_ptr fifo::SlotHeader* slot, const LanePage* pages, uint32_t e, TensorStream& stream) {
    const uint32_t num_lanes = slot->num_lanes;
    stream.num_receivers = 0;
    for (uint32_t l = 0; l < num_lanes; ++l) {
        const LanePage& page = pages[l];
        const TensorPrefetcherEntry& entry = page.entry(e);
        const TensorPrefetcherTensorLayout& layout = page.layout_of(entry);
        // Receiver-contiguous, batched, ungrouped and unselected: the host rejects the rest for
        // this placement.
        ASSERT(layout.layout_mode == fifo::kReceiverContiguousLayout && layout.streaming == 0 && entry.num_groups == 1);
        if (l == 0) {
            stream.block_bytes = layout.page_bytes_per_recv;
            stream.block_count = layout.block_count;
        }
        ASSERT(stream.block_bytes == layout.page_bytes_per_recv && stream.block_count == layout.block_count);
        const uint32_t lane_receivers = pipe_num_receivers(page.header().prefetch.target_config_page_addr);
        for (uint32_t r = 0; r < lane_receivers; ++r) {
            const uint32_t i = stream.num_receivers++;
            ASSERT(i < fifo::kMaxReceivers);
            stream.lane[i] = l;
            stream.receiver_in_lane[i] = r;
            stream.bank[i] = slot->lane_bank_id[l];
            stream.slab_base[i] = entry.bank_local_base + (layout.recv_index_base + r) * layout.recv_stride_bytes;
        }
    }
}

}  // namespace tensor_prefetcher
