// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The Tensor prefetcher reader's loop: it stages weight blocks into the region's stage ring for the pusher
// (tensor_prefetcher_pusher.cpp) to send on to the receivers. Where a block comes from is a BlockSource, so a
// reader can stage from any source the NoC reads; tensor_prefetcher_reader.cpp reads DRAM banks.
//
// For each tensor of a PREFETCH slot the reader makes passes over every receiver, reading the next block
// of each one that has room in its share of the stage ring and a ready block, and skipping the rest. Each pass's
// reads carry one of the kernel's transaction ids, so up to one pass per id is in flight, across the tensors of a
// request; passes land and are published in the order they were issued. Reading for whichever receiver has room, rather
// than draining one receiver's slab before starting the next, is what keeps a consumer that needs all of its receivers
// to progress together from deadlocking against a sender that is waiting on one of them.
//
// A receiver's share of the stage ring stays put from one tensor and request to the next as long as the receiver
// count does (tensor_prefetcher_fifo.hpp), so the reader runs on into the next tensor while the pusher
// is still sending the last one's blocks.
//
// A reader that is the first kernel in tensor_prefetcher_kernels() plans each tensor (kPlansBlocks): its source
// decides how many blocks each receiver gets, and the reader publishes that in the slot's plan. A reader further
// down follows the plan the first kernel published.
//
// A BlockSource has these members, `i` naming a receiver of the tensor's TensorStream:
//   uint32_t plan_blocks(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream,
//                        uint32_t read_trid)
//       Only in a planning reader: how many blocks each receiver gets of tensor `e` of `slot`, planned as
//       `stream`. GroupSelection (tensor_prefetcher_groups.h) picks a grouped tensor's groups. Any reads it
//       issues carry `read_trid` and it waits for them itself; of the reads in flight, at most the oldest pass's
//       carry that id too.
//   void begin_tensor(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream,
//                     uint32_t blocks)
//       Before the reader stages `blocks` blocks of tensor `e` for each receiver.
//   bool ready(uint32_t i, uint32_t block)
//       Whether receiver `i`'s block `block` can be read now. A block that is not ready waits for a later
//       pass, as one whose receiver's share is still full does.
//   void read(uint32_t i, uint32_t block, uint32_t dst)
//       Issues the NoC reads that bring the block into L1 at `dst`, with the transaction id the reader set on the
//       read command buffer (noc_async_read does). The reader waits on that id before publishing the block.
// and optionally:
//   void release(uint32_t i, uint32_t bytes)
//       `bytes` more of receiver `i`'s source bytes, the oldest first, have landed in the stage ring, so the source
//       may reuse what they were read from. Called as each pass lands, which may be after the reader has begun a
//       later tensor of the request.
// and, to let the reader bring in several blocks of a receiver with one read, both of:
//   uint32_t contiguous_blocks(uint32_t i, uint32_t block, uint32_t max_blocks) const
//       How many blocks from `block`, at least 1 and at most `max_blocks`, one read() can bring in back to back,
//       every one of them ready. Only asked once ready(i, block) holds.
//   void read(uint32_t i, uint32_t block, uint32_t num_blocks, uint32_t dst)
//       read() for `num_blocks` blocks from `block`, which land back to back at `dst`.

#include <stdint.h>

#include <type_traits>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace tensor_prefetcher {

static_assert(fifo::kMaxReceivers <= 32, "A pass tracks its receivers in one 32-bit mask");

// Whether a BlockSource reads runs of blocks (contiguous_blocks and the four-argument read).
template <typename BlockSource, typename = void>
struct reads_runs : std::false_type {};
template <typename BlockSource>
struct reads_runs<BlockSource, std::void_t<decltype(std::declval<const BlockSource&>().contiguous_blocks(0u, 0u, 0u))>>
    : std::true_type {};

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

// The reader's state, kept from one tensor and request to the next.
struct ReaderState {
    static constexpr uint32_t kMaxPasses = TridQueue::kSlots;

    // Where each receiver's next read goes in its share of the stage ring, and where the next read to land went;
    // `land` trails `issue` by the reads in flight.
    StageCursor issue[fifo::kMaxReceivers];
    StageCursor land[fifo::kMaxReceivers];
    // Bytes of each receiver's share staged: published (`staged`), and counting the reads in flight (`issued`). And
    // the pusher's count for it as last read: it only grows, so the reader reads it again only when the count it has
    // does not leave room for the longest run it could read. The host zeroes the shared counts before launch, and
    // this kernel is the only writer of `staged`, so these start at 0 with them.
    uint32_t staged[fifo::kMaxReceivers] = {};
    uint32_t issued[fifo::kMaxReceivers] = {};
    uint32_t pushed_seen[fifo::kMaxReceivers] = {};
    // The passes in flight, possibly from earlier tensors of the request: the receivers each one reads for, how many
    // blocks it reads for each, and how its tensor's blocks lie. Slot k's reads carry kTrids[k].
    uint32_t pass_receivers[kMaxPasses] = {};
    uint16_t pass_blocks[kMaxPasses][fifo::kMaxReceivers] = {};
    StageGeometry pass_geometry[kMaxPasses] = {};
    TridQueue queue;

    // Starts every share over, for a new receiver count. Nothing is in flight.
    FORCE_INLINE void reset_cursors() {
        for (uint32_t i = 0; i < fifo::kMaxReceivers; ++i) {
            issue[i].reset();
            land[i].reset();
        }
    }
};
// The reader lives as long as fast dispatch does, and so does this.
inline ReaderState reader_state;
static_assert(TensorStream::kMaxRunBytes / fifo::kLineBytes <= 0xFFFF, "A pass counts a run's blocks in 16 bits");

// Whether a BlockSource has release(), which the reader calls with each receiver's bytes as a pass lands.
template <typename BlockSource, typename = void>
struct releases_blocks : std::false_type {};
template <typename BlockSource>
struct releases_blocks<BlockSource, std::void_t<decltype(std::declval<BlockSource&>().release(0u, 0u))>>
    : std::true_type {};

// Publishes every pass that has landed, oldest first; a staged count tells the pusher its blocks have landed.
// Returns whether any pass landed.
template <typename BlockSource>
FORCE_INLINE bool land_passes(ReaderState& state, BlockSource& source) {
    bool landed = false;
    Noc noc;
    while (state.queue.in_flight > 0 && noc.is_read_trid_flushed(kTrids[state.queue.oldest])) {
        const uint32_t slot = state.queue.oldest;
        const StageGeometry& geometry = state.pass_geometry[slot];
        for (uint32_t receivers = state.pass_receivers[slot]; receivers != 0; receivers &= receivers - 1) {
            const uint32_t i = __builtin_ctz(receivers);
            const uint32_t num_blocks = state.pass_blocks[slot][i];
            state.staged[i] += state.land[i].run_bytes_taken(geometry, num_blocks);
            state.land[i].place_run(geometry, num_blocks);
            *region.staged(i) = state.staged[i];
            if constexpr (releases_blocks<BlockSource>::value) {
                source.release(i, num_blocks * geometry.block_bytes);
            }
        }
        state.queue.pop();
        landed = true;
    }
    return landed;
}

// Waits for every read in flight to land and publishes it. False, with nothing in flight, if termination was
// requested first.
template <typename BlockSource>
bool land_all_passes(const FifoConsumer& requests, ReaderState& state, BlockSource& source) {
    WAYPOINT("TRLW");
    while (state.queue.in_flight > 0) {
        if (!land_passes(state, source) && requests.terminate_requested()) {
            for (uint32_t k = 0; k < ReaderState::kMaxPasses; ++k) {
                noc_async_read_barrier_with_trid(kTrids[k]);
            }
            state.queue.in_flight = 0;
            return false;
        }
    }
    WAYPOINT("TRLD");
    return true;
}

// Issues the reads of `blocks` blocks of one tensor for each receiver, publishing them as they land. Returns once
// every read is issued: the last ones may still be in flight, to land while the next tensor's reads go out. False if
// termination was requested first.
template <typename BlockSource>
bool stage_tensor(
    const FifoConsumer& requests,
    const TensorStream& stream,
    uint32_t blocks,
    BlockSource& source,
    ReaderState& state) {
    uint32_t next_block[fifo::kMaxReceivers];
    for (uint32_t i = 0; i < stream.num_receivers; ++i) {
        next_block[i] = 0;
    }
    const uint32_t max_run_blocks = reads_runs<BlockSource>::value ? stream.max_run_blocks() : 1;
    uint32_t to_issue = stream.num_receivers * blocks;
    WAYPOINT("TRSW");
    while (to_issue > 0) {
        bool progressed = land_passes(state, source);
        bool waiting_on_pusher = false;
        if (!state.queue.full()) {
            const uint32_t slot = state.queue.next();
            noc_async_read_set_trid(kTrids[slot]);
            uint32_t reading = 0;
            for (uint32_t i = 0; i < stream.num_receivers; ++i) {
                if (next_block[i] == blocks) {
                    continue;
                }
                StageCursor& cursor = state.issue[i];
                const uint32_t first_taken = cursor.bytes_taken(stream);
                const uint32_t blocks_left = blocks - next_block[i];
                const uint32_t longest =
                    cursor.run_blocks(stream, blocks_left < max_run_blocks ? blocks_left : max_run_blocks);
                uint32_t free_bytes = stream.share_bytes - (state.issued[i] - state.pushed_seen[i]);
                if (free_bytes < cursor.run_bytes_taken(stream, longest)) {
                    state.pushed_seen[i] = *region.pushed(i);
                    free_bytes = stream.share_bytes - (state.issued[i] - state.pushed_seen[i]);
                    if (free_bytes < first_taken) {
                        waiting_on_pusher = true;
                        continue;
                    }
                }
                if (!source.ready(i, next_block[i])) {
                    continue;
                }
                // A run: as many of the receiver's next blocks as fit in its free share bytes back to back.
                uint32_t num_blocks = 1;
                if constexpr (reads_runs<BlockSource>::value) {
                    num_blocks = 1 + blocks_in(free_bytes - first_taken, stream.block_footprint, longest - 1);
                    num_blocks = source.contiguous_blocks(i, next_block[i], num_blocks);
                }
                state.issued[i] += cursor.run_bytes_taken(stream, num_blocks);
                const uint32_t dst = stream.stage_addr(i, cursor.place_run(stream, num_blocks));
                if constexpr (reads_runs<BlockSource>::value) {
                    source.read(i, next_block[i], num_blocks, dst);
                } else {
                    source.read(i, next_block[i], dst);
                }
                state.pass_blocks[slot][i] = static_cast<uint16_t>(num_blocks);
                next_block[i] += num_blocks;
                to_issue -= num_blocks;
                reading |= 1u << i;
            }
            if (reading != 0) {
                state.pass_receivers[slot] = reading;
                state.pass_geometry[slot] = stream;
                state.queue.push();
                progressed = true;
            }
        }
        if (!progressed) {
            // Nothing landed and nothing to read. The waypoint says whether every id has a pass in flight (TRSL),
            // a receiver's share waits for the pusher to send what it holds (TRSP), or every unfinished receiver
            // waits for the block source (TRSB).
            if (state.queue.full()) {
                WAYPOINT("TRSL");
            } else if (waiting_on_pusher) {
                WAYPOINT("TRSP");
            } else {
                WAYPOINT("TRSB");
            }
            if (requests.terminate_requested()) {
                land_all_passes(requests, state, source);
                return false;
            }
            continue;
        }
        WAYPOINT("TRSW");
    }
    WAYPOINT("TRSD");
    return true;
}

// Stages every tensor of a PREFETCH slot, one tensor at a time, and returns once every block has landed.
// `layout_receivers` is the receiver count the stage ring is laid out for (0 before the first request); a request
// for a different count first waits for the pusher to empty the ring. False if termination was requested.
template <typename BlockSource>
bool stage_request(
    const FifoConsumer& requests,
    volatile tt_l1_ptr fifo::SlotHeader* slot,
    uint32_t& layout_receivers,
    ReaderState& state,
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
            state.reset_cursors();
        }
        uint32_t blocks = 0;
        if constexpr (kPlansBlocks) {
            // The next pass's id: free, or the oldest pass's, which lands first anyway.
            blocks = source.plan_blocks(slot, e, stream, kTrids[state.queue.next()]);
            *tensor_blocks(slot, e) = blocks;
        } else if (!wait_for_tensor_blocks(requests, slot, e, blocks)) {
            land_all_passes(requests, state, source);
            return false;
        }
        source.begin_tensor(slot, e, stream, blocks);
        if (!stage_tensor(requests, stream, blocks, source, state)) {
            return false;
        }
    }
    // Nothing stays in flight past the request: the next one may lay the ring out for other receivers, or not come.
    return land_all_passes(requests, state, source);
}

// Serves the request FIFO, staging every PREFETCH slot's blocks from `source`, until termination.
template <typename BlockSource>
void run_reader(BlockSource& source) {
    uint32_t layout_receivers = 0;
    serve_requests([&](const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
        return stage_request(requests, slot, layout_receivers, reader_state, source);
    });
}

}  // namespace tensor_prefetcher
