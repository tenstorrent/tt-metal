// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher pusher. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_reader.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Both retire every request slot the host writes.
//
// The pusher serves whichever receiver has a block staged and room in its PrefetcherPipe ring, not the
// receivers in a fixed order: a consumer that needs all of its receivers to progress together would
// otherwise deadlock with a pusher stalled on one of them. Each lane of a request is one DRAM-bank
// sender endpoint, whose PrefetcherPipe sender state the host keeps in this core's L1.
//
// Compile-time defines (set by TensorPrefetcherKernel):
//   TENSOR_PREFETCHER_REGION_BASE      L1 base of the region described in tensor_prefetcher_fifo.hpp
//   TENSOR_PREFETCHER_STAGE_RING_SIZE  bytes of that region's stage ring

#include <stdint.h>

#include <new>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using experimental::DramSenderConfigPage;
using experimental::PrefetcherPipe;
using tensor_prefetcher::FifoConsumer;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::region_ptr;
using tensor_prefetcher::TensorStream;

constexpr uint32_t region_base = TENSOR_PREFETCHER_REGION_BASE;
constexpr uint32_t stage_ring_base = region_base + fifo::kStageRingOffset;

namespace {

FORCE_INLINE volatile tt_l1_ptr uint32_t* staged_ptr(uint32_t i) {
    return region_ptr<uint32_t>(region_base, fifo::kStagedOffset + i * fifo::kLineBytes);
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* pushed_ptr(uint32_t i) {
    return region_ptr<uint32_t>(region_base, fifo::kPushedOffset + i * fifo::kLineBytes);
}

// The PrefetcherPipe of each lane, alive for one tensor. Constructing one applies that tensor's block
// size to the pipe; destroying it persists the pipe's cursors for the next request.
class LanePipes {
public:
    LanePipes(const LanePage* pages, uint32_t num_lanes, uint32_t entry_size) : num_lanes_(num_lanes) {
        for (uint32_t l = 0; l < num_lanes_; ++l) {
            new (&storage_[l])
                PrefetcherPipe(DramSenderConfigPage{pages[l].header().prefetch.target_config_page_addr}, entry_size);
        }
    }
    LanePipes(const LanePipes&) = delete;
    LanePipes& operator=(const LanePipes&) = delete;
    ~LanePipes() {
        for (uint32_t l = 0; l < num_lanes_; ++l) {
            (*this)[l].~PrefetcherPipe();
        }
    }
    PrefetcherPipe& operator[](uint32_t l) { return *std::launder(reinterpret_cast<PrefetcherPipe*>(&storage_[l])); }

private:
    struct alignas(PrefetcherPipe) Storage {
        uint8_t bytes[sizeof(PrefetcherPipe)];
    };
    Storage storage_[fifo::kMaxLanes];
    uint32_t num_lanes_;
};

// Pushes every block of one tensor as the reader stages it. False if termination was requested first.
bool push_tensor(const FifoConsumer& requests, const LanePage* pages, uint32_t num_lanes, const TensorStream& stream) {
    LanePipes pipes(pages, num_lanes, stream.block_bytes);
    Noc noc;
    uint32_t pushed[fifo::kMaxReceivers];
    uint32_t blocks_pushed[fifo::kMaxReceivers];
    for (uint32_t i = 0; i < stream.num_receivers; ++i) {
        pushed[i] = *pushed_ptr(i);
        blocks_pushed[i] = 0;
    }
    uint32_t remaining = stream.num_receivers * stream.block_count;
    WAYPOINT("TPPW");
    while (remaining > 0) {
        bool progressed = false;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (blocks_pushed[i] == stream.block_count || *staged_ptr(i) == pushed[i]) {
                continue;
            }
            PrefetcherPipe& pipe = pipes[stream.lane[i]];
            const uint32_t r = stream.receiver_in_lane[i];
            if (!pipe.receiver_has_space(r, 1)) {
                continue;
            }
            const CoreLocalMem<uint8_t> block(
                stream.stage_slot_addr(stage_ring_base, i, pushed[i] % fifo::kStageSlotsPerReceiver));
            pipe.write_to_receiver(noc, r, block, 1);
            // The block must land before the credit that advertises it, and before the reader may
            // reuse its stage slot.
            pipe.flush_writes(noc);
            pipe.push_back_to_receiver(r, 1, noc);
            *pushed_ptr(i) = ++pushed[i];
            ++blocks_pushed[i];
            --remaining;
            progressed = true;
        }
        if (!progressed && requests.terminate_requested()) {
            return false;
        }
    }
    WAYPOINT("TPPD");
    return true;
}

// Waits for every receiver of each pipe a DRAIN page names to ack all it was sent.
void drain(const LanePage& page) {
    const uint32_t num_targets = page.header().drain.num_targets;
    const uint32_t* targets =
        reinterpret_cast<const uint32_t*>(page.bytes() + sizeof(tt::tt_metal::TensorPrefetcherRequestHeader));
    for (uint32_t t = 0; t < num_targets; ++t) {
        // Keep the entry size the pipe last applied, so draining never re-grids it.
        const uint32_t applied_entry_size =
            *region_ptr<uint32_t>(targets[t], PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE * sizeof(uint32_t));
        PrefetcherPipe pipe(DramSenderConfigPage{targets[t]}, applied_entry_size);
        WAYPOINT("TPDW");
        pipe.barrier();
        WAYPOINT("TPDD");
    }
}

// Serves one request slot. False if termination was requested first.
bool push_request(const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
    const uint32_t num_lanes = slot->num_lanes;
    ASSERT(num_lanes > 0 && num_lanes <= fifo::kMaxLanes);
    LanePage pages[fifo::kMaxLanes];
    for (uint32_t l = 0; l < num_lanes; ++l) {
        tensor_prefetcher::copy_lane_page(slot, l, pages[l]);
    }
    if (pages[0].header().base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_DRAIN) {
        for (uint32_t l = 0; l < num_lanes; ++l) {
            drain(pages[l]);
        }
        return true;
    }
    ASSERT(pages[0].header().base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH);
    const uint32_t num_tensors = pages[0].header().prefetch.num_entries;
    TensorStream stream;
    for (uint32_t e = 0; e < num_tensors; ++e) {
        tensor_prefetcher::plan_tensor(slot, pages, e, stream);
        if (!push_tensor(requests, pages, num_lanes, stream)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void kernel_main() {
    FifoConsumer requests(region_base, fifo::kPusherRdOffset);
    while (true) {
        volatile tt_l1_ptr fifo::SlotHeader* slot = requests.wait_for_slot();
        if (slot == nullptr) {
            break;
        }
        switch (slot->command) {
            case fifo::kSlotNop: break;
            case fifo::kSlotRequest:
                if (!push_request(requests, slot)) {
                    return;
                }
                break;
            default: ASSERT(false); break;
        }
        requests.retire_slot();
    }
}
