// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher reader. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_pusher.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Both retire every request slot the host writes.
//
// For each tensor of a PREFETCH slot the reader makes passes over every receiver, reading the next block
// of each one that has a free stage slot and skipping one whose slots are all waiting on the pusher. A
// pass's reads are in flight together and published together after one read barrier. Reading for whichever receiver
// has room, rather than draining one receiver's slab before starting the next, is what keeps a consumer
// that needs all of its receivers to progress together from deadlocking against a sender that is
// waiting on one of them.
//
// A receiver's stage slots stay put from one tensor and request to the next as long as the receiver
// count does (tensor_prefetcher_fifo.hpp), so the reader runs on into the next tensor while the pusher
// is still sending the last one's blocks.
//
// Compile-time defines (set by TensorPrefetcherKernel):
//   TENSOR_PREFETCHER_REGION_BASE      L1 base of the region described in tensor_prefetcher_fifo.hpp
//   TENSOR_PREFETCHER_LANES            lanes per request slot (fifo::RegionLayout)
//   TENSOR_PREFETCHER_STAGE_RING_SIZE  bytes of that region's stage ring

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tensor_prefetcher::FifoConsumer;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::TensorStream;

constexpr tensor_prefetcher::Region region{
    .base = TENSOR_PREFETCHER_REGION_BASE, .layout = fifo::RegionLayout{TENSOR_PREFETCHER_LANES}};
constexpr uint32_t stage_ring_base = region.stage_ring_base();
constexpr uint32_t stage_ring_size = TENSOR_PREFETCHER_STAGE_RING_SIZE;

static_assert(fifo::kMaxReceivers <= 32, "A pass tracks its receivers in one 32-bit mask");

namespace {

// Waits until the pusher has sent everything staged for receivers [0, num_receivers), so the stage ring
// may be laid out for a different receiver count. False if termination was requested first.
bool wait_for_pusher(const FifoConsumer& requests, uint32_t num_receivers) {
    WAYPOINT("TRPW");
    for (uint32_t i = 0; i < num_receivers; ++i) {
        while (*region.pushed(i) != *region.staged(i)) {
            if (requests.terminate_requested()) {
                return false;
            }
        }
    }
    WAYPOINT("TRPD");
    return true;
}

// Stages every block of one tensor. False if termination was requested first.
bool stage_tensor(const FifoConsumer& requests, const TensorStream& stream) {
    uint32_t staged[fifo::kMaxReceivers];
    uint32_t next_block[fifo::kMaxReceivers];
    for (uint32_t i = 0; i < stream.num_receivers; ++i) {
        staged[i] = *region.staged(i);
        next_block[i] = 0;
    }
    uint32_t remaining = stream.num_receivers * stream.block_count;
    WAYPOINT("TRSW");
    while (remaining > 0) {
        uint32_t reading = 0;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (next_block[i] == stream.block_count || staged[i] - *region.pushed(i) == fifo::kStageSlotsPerReceiver) {
                continue;
            }
            const uint32_t dst = stream.stage_slot_addr(stage_ring_base, i, staged[i] % fifo::kStageSlotsPerReceiver);
            const uint64_t src = get_noc_addr_from_bank_id<true>(
                stream.bank[i], stream.slab_base[i] + next_block[i] * stream.block_bytes);
            noc_async_read(src, dst, stream.block_bytes);
            reading |= 1u << i;
        }
        if (reading == 0) {
            if (requests.terminate_requested()) {
                return false;
            }
            continue;
        }
        // A staged count tells the pusher its block has landed.
        noc_async_read_barrier();
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (reading & (1u << i)) {
                *region.staged(i) = ++staged[i];
                ++next_block[i];
                --remaining;
            }
        }
    }
    WAYPOINT("TRSD");
    return true;
}

// Stages every tensor of a PREFETCH slot, one tensor at a time. `layout_receivers` is the receiver count
// the stage ring is laid out for (0 before the first request); a request for a different count first waits
// for the pusher to empty the ring. False if termination was requested.
bool stage_request(
    const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t& layout_receivers) {
    ASSERT(slot->num_lanes > 0);
    volatile tt_l1_ptr tt::tt_metal::TensorPrefetcherRequestHeader* header = LanePage(slot, 0).header();
    if (header->base.cmd_id != tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH) {
        return true;  // DRAIN is the pusher's alone.
    }
    const uint32_t num_tensors = header->prefetch.num_entries;
    TensorStream stream;
    for (uint32_t e = 0; e < num_tensors; ++e) {
        tensor_prefetcher::plan_tensor(slot, e, stage_ring_size, stream);
        if (stream.num_receivers != layout_receivers) {
            // Every tensor of a request has the same receivers, so this is its first tensor.
            ASSERT(e == 0);
            if (!wait_for_pusher(requests, layout_receivers)) {
                return false;
            }
            layout_receivers = stream.num_receivers;
        }
        if (!stage_tensor(requests, stream)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void kernel_main() {
    uint32_t layout_receivers = 0;
    tensor_prefetcher::serve_requests(
        region, fifo::kReaderRdOffset, [&](const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
            return stage_request(requests, slot, layout_receivers);
        });
}
