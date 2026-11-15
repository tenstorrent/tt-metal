// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The Tensor prefetcher reader's loop: it stages weight blocks into the region's stage ring for the pusher
// (tensor_prefetcher_pusher.cpp) to send on to the receivers. Where a block comes from is a BlockSource, so a
// reader can stage from any source the NoC reads; tensor_prefetcher_reader.cpp reads DRAM banks.
//
// For each tensor of a PREFETCH slot the reader makes passes over every receiver, reading the next block
// of each one that has a free stage slot and a ready block, and skipping the rest. A pass's reads are in
// flight together and published together after one read barrier. Reading for whichever receiver has room,
// rather than draining one receiver's slab before starting the next, is what keeps a consumer that needs
// all of its receivers to progress together from deadlocking against a sender that is waiting on one of
// them.
//
// A receiver's stage slots stay put from one tensor and request to the next as long as the receiver
// count does (tensor_prefetcher_fifo.hpp), so the reader runs on into the next tensor while the pusher
// is still sending the last one's blocks.
//
// A BlockSource has these members, `i` naming a receiver of the tensor's TensorStream:
//   uint32_t begin_tensor(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream)
//       Before the reader stages tensor `e` of `slot`, planned as `stream`. Returns how many blocks each receiver
//       gets of it, which the reader publishes in the slot's plan for the kernels after it.
//   bool ready(uint32_t i, uint32_t block)
//       Whether receiver `i`'s block `block` can be read now. A block that is not ready waits for a later
//       pass, as one whose stage slot is still full does.
//   void read(uint32_t i, uint32_t block, uint32_t dst)
//       Issues the NoC reads that bring the block into L1 at `dst`. The pass's noc_async_read_barrier()
//       waits for them.
//   void release(uint32_t i, uint32_t block)
//       The block has landed in the stage ring.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace tensor_prefetcher {

static_assert(fifo::kMaxReceivers <= 32, "A pass tracks its receivers in one 32-bit mask");

// Waits until the pusher has sent everything staged for receivers [0, num_receivers), so the stage ring
// may be laid out for a different receiver count. False if termination was requested first.
inline bool wait_for_pusher(const FifoConsumer& requests, uint32_t num_receivers) {
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

// Stages `blocks` blocks of one tensor for each receiver. False if termination was requested first.
template <typename BlockSource>
bool stage_tensor(const FifoConsumer& requests, const TensorStream& stream, uint32_t blocks, BlockSource& source) {
    uint32_t staged[fifo::kMaxReceivers];
    uint32_t next_block[fifo::kMaxReceivers];
    for (uint32_t i = 0; i < stream.num_receivers; ++i) {
        staged[i] = *region.staged(i);
        next_block[i] = 0;
    }
    uint32_t remaining = stream.num_receivers * blocks;
    WAYPOINT("TRSW");
    while (remaining > 0) {
        uint32_t reading = 0;
        bool waiting_on_pusher = false;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (next_block[i] == blocks) {
                continue;
            }
            if (staged[i] - *region.pushed(i) == fifo::kStageSlotsPerReceiver) {
                waiting_on_pusher = true;
                continue;
            }
            if (!source.ready(i, next_block[i])) {
                continue;
            }
            source.read(i, next_block[i], stream.stage_slot_addr(i, staged[i] % fifo::kStageSlotsPerReceiver));
            reading |= 1u << i;
        }
        if (reading == 0) {
            // Nothing to read. The waypoint says whether a receiver's stage slots wait for the pusher to send
            // them (TRSP) or every unfinished receiver waits for the block source (TRSB).
            if (waiting_on_pusher) {
                WAYPOINT("TRSP");
            } else {
                WAYPOINT("TRSB");
            }
            if (requests.terminate_requested()) {
                return false;
            }
            continue;
        }
        WAYPOINT("TRSW");
        // A staged count tells the pusher its block has landed.
        noc_async_read_barrier();
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (reading & (1u << i)) {
                *region.staged(i) = ++staged[i];
                source.release(i, next_block[i]);
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
template <typename BlockSource>
bool stage_request(
    const FifoConsumer& requests,
    volatile tt_l1_ptr fifo::SlotHeader* slot,
    uint32_t& layout_receivers,
    BlockSource& source) {
    ASSERT(slot->num_lanes > 0);
    volatile tt_l1_ptr TensorPrefetcherRequestHeader* header = LanePage(slot, 0).header();
    if (header->base.cmd_id != tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH) {
        return true;  // DRAIN is the pusher's alone.
    }
    const uint32_t num_tensors = header->prefetch.num_entries;
    TensorStream stream;
    for (uint32_t e = 0; e < num_tensors; ++e) {
        plan_tensor(slot, e, stream);
        if (stream.num_receivers != layout_receivers) {
            // Every tensor of a request has the same receivers, so this is its first tensor.
            ASSERT(e == 0);
            if (!wait_for_pusher(requests, layout_receivers)) {
                return false;
            }
            layout_receivers = stream.num_receivers;
        }
        const uint32_t blocks = source.begin_tensor(slot, e, stream);
        *tensor_blocks(slot, e) = blocks;
        if (!stage_tensor(requests, stream, blocks, source)) {
            return false;
        }
    }
    return true;
}

// Serves the request FIFO, staging every PREFETCH slot's blocks from `source`, until termination.
template <typename BlockSource>
void run_reader(BlockSource& source) {
    uint32_t layout_receivers = 0;
    serve_requests([&](const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
        return stage_request(requests, slot, layout_receivers, source);
    });
}

}  // namespace tensor_prefetcher
