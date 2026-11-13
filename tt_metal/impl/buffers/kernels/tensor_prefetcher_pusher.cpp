// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher pusher. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_reader.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Both retire every request slot the host writes.
//
// The pusher serves whichever receivers have a block staged and room in their PrefetcherPipe rings, not
// the receivers in a fixed order: a consumer that needs all of its receivers to progress together would
// otherwise deadlock with a pusher stalled on one of them. Each pass writes one block to every such
// receiver, waits once for all of those writes to land, then credits them. Each lane of a request is one DRAM-bank
// sender endpoint, whose PrefetcherPipe sender state the host keeps in this core's L1.
//
// Compile-time defines (set by TensorPrefetcherKernel):
//   TENSOR_PREFETCHER_REGION_BASE      L1 base of the region described in tensor_prefetcher_fifo.hpp
//   TENSOR_PREFETCHER_LANES            lanes per request slot (fifo::RegionLayout)
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
using tensor_prefetcher::TensorStream;

constexpr tensor_prefetcher::Region region{
    .base = TENSOR_PREFETCHER_REGION_BASE, .layout = fifo::RegionLayout{TENSOR_PREFETCHER_LANES}};
constexpr uint32_t stage_ring_base = region.stage_ring_base();
constexpr uint32_t stage_ring_size = TENSOR_PREFETCHER_STAGE_RING_SIZE;

static_assert(fifo::kMaxReceivers <= 32, "A pass tracks its receivers in one 32-bit mask");

namespace {

// The PrefetcherPipe of each lane, alive for one tensor. Constructing one applies that tensor's block
// size to the pipe; destroying it persists the pipe's cursors for the next request.
class LanePipes {
public:
    LanePipes(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t num_lanes, uint32_t entry_size) :
        num_lanes_(num_lanes) {
        for (uint32_t l = 0; l < num_lanes_; ++l) {
            const uint32_t config_page_addr = LanePage(slot, l).header()->prefetch.target_config_page_addr;
            new (&storage_[l]) PrefetcherPipe(DramSenderConfigPage{config_page_addr}, entry_size);
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
    Storage storage_[TENSOR_PREFETCHER_LANES];
    uint32_t num_lanes_;
};

// Pushes every block of one tensor as the reader stages it. False if termination was requested first.
bool push_tensor(
    const FifoConsumer& requests,
    volatile tt_l1_ptr fifo::SlotHeader* slot,
    uint32_t num_lanes,
    const TensorStream& stream) {
    LanePipes pipes(slot, num_lanes, stream.block_bytes);
    Noc noc;
    uint32_t pushed[fifo::kMaxReceivers];
    uint32_t blocks_pushed[fifo::kMaxReceivers];
    for (uint32_t i = 0; i < stream.num_receivers; ++i) {
        pushed[i] = *region.pushed(i);
        blocks_pushed[i] = 0;
    }
    uint32_t remaining = stream.num_receivers * stream.block_count;
    WAYPOINT("TPPW");
    while (remaining > 0) {
        uint32_t writing = 0;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (blocks_pushed[i] == stream.block_count || *region.staged(i) == pushed[i]) {
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
            writing |= 1u << i;
        }
        if (writing == 0) {
            if (requests.terminate_requested()) {
                return false;
            }
            continue;
        }
        // The blocks must land before the credits that advertise them, and before the reader may reuse
        // their stage slots. The flush covers every pipe's writes on this NoC.
        pipes[0].flush_writes(noc);
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            if (writing & (1u << i)) {
                pipes[stream.lane[i]].push_back_to_receiver(stream.receiver_in_lane[i], 1, noc);
                *region.pushed(i) = ++pushed[i];
                ++blocks_pushed[i];
                --remaining;
            }
        }
    }
    WAYPOINT("TPPD");
    return true;
}

// Waits for every receiver of each pipe a DRAIN page names to ack all it was sent. False if termination
// was requested first.
bool drain(const FifoConsumer& requests, const LanePage& page) {
    const uint32_t num_targets = page.header()->drain.num_targets;
    volatile tt_l1_ptr uint32_t* targets = page.body_words();
    for (uint32_t t = 0; t < num_targets; ++t) {
        const uint32_t config_page_addr = targets[t];
        volatile tt_l1_ptr uint32_t* config_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
        // Keep the entry size the pipe last applied, so draining never re-grids it.
        PrefetcherPipe pipe(
            DramSenderConfigPage{config_page_addr},
            experimental::load_prefetcher_pipe_config_word(config_page, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE));
        const uint32_t num_receivers =
            experimental::load_prefetcher_pipe_config_word(config_page, REMOTE_DFB_CFG_NUM_RECEIVERS);
        WAYPOINT("TPDW");
        for (uint32_t r = 0; r < num_receivers; ++r) {
            while (!pipe.receiver_drained(r)) {
                if (requests.terminate_requested()) {
                    return false;
                }
            }
        }
        WAYPOINT("TPDD");
    }
    return true;
}

// Serves one request slot. False if termination was requested first.
bool push_request(const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
    const uint32_t num_lanes = slot->num_lanes;
    ASSERT(num_lanes > 0);
    volatile tt_l1_ptr tt::tt_metal::TensorPrefetcherRequestHeader* header = LanePage(slot, 0).header();
    if (header->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_DRAIN) {
        for (uint32_t l = 0; l < num_lanes; ++l) {
            if (!drain(requests, LanePage(slot, l))) {
                return false;
            }
        }
        return true;
    }
    ASSERT(header->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH);
    const uint32_t num_tensors = header->prefetch.num_entries;
    TensorStream stream;
    for (uint32_t e = 0; e < num_tensors; ++e) {
        tensor_prefetcher::plan_tensor(slot, e, stage_ring_size, stream);
        if (!push_tensor(requests, slot, num_lanes, stream)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void kernel_main() { tensor_prefetcher::serve_requests(region, fifo::kPusherRdOffset, push_request); }
