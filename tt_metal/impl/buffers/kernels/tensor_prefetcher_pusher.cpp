// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher pusher. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_reader.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Every Tensor prefetcher kernel retires every request slot
// the host writes.
//
// The pusher serves whichever receivers have a block staged and room in their PrefetcherPipe rings, not
// the receivers in a fixed order: a consumer that needs all of its receivers to progress together would
// otherwise deadlock with a pusher stalled on one of them. Each write carries a run: as many of a receiver's staged
// blocks as lie back to back in the stage ring and fit in its pipe ring, up to half the ring, credited together. Each
// run carries one of the kernel's transaction ids, so up to one run per id is in flight, several to one receiver
// (each written after that receiver's runs not credited yet). Runs are credited in the order they were written, each
// as soon as its writes have left this core's L1, while later runs keep the NoC busy. Runs stay in flight from one
// tensor to the next, and from one request to the next when that request is queued already through the same pipes.
// Each lane of a request is one DRAM-bank sender endpoint, whose PrefetcherPipe sender state the host keeps in this
// core's L1.
//
// Defines: tensor_prefetcher_fifo_device.h.

#include <stdint.h>

#include <new>
#include <optional>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "dev_mem_map.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using experimental::DramSenderConfigPage;
using experimental::PrefetcherPipe;
using tensor_prefetcher::blocks_in;
using tensor_prefetcher::FifoConsumer;
using tensor_prefetcher::kTrids;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::region;
using tensor_prefetcher::StageCursor;
using tensor_prefetcher::TensorStream;
using tensor_prefetcher::TridQueue;

static_assert(!tensor_prefetcher::kPlansBlocks, "The pusher sends what a kernel before it planned");

namespace {

// The PrefetcherPipe of each lane, alive for a run of tensors with the same block size. Constructing one applies
// that block size to the pipe; destroying it persists the pipe's cursors for the next request.
class LanePipes {
public:
    LanePipes(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t num_lanes, uint32_t entry_size) :
        entry_size_(entry_size), num_lanes_(num_lanes) {
        const uint32_t transfer_blocks = TensorStream::max_run_blocks(entry_size);
        for (uint32_t l = 0; l < num_lanes_; ++l) {
            const uint32_t config_page_addr = LanePage(slot, l).header()->prefetch.target_config_page_addr;
            config_page_addr_[l] = config_page_addr;
            new (&storage_[l]) PrefetcherPipe(DramSenderConfigPage{config_page_addr}, entry_size);
            const uint32_t ring_entries = (*this)[l].ring_entries();
            const uint32_t half_ring = ring_entries < 2 ? 1 : ring_entries / 2;
            max_run_blocks_[l] = half_ring < transfer_blocks ? half_ring : transfer_blocks;
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
    // The most blocks one run to a receiver of lane `l` carries: what one transfer moves
    // (TensorStream::max_run_blocks), and at most half its ring. A run is credited only once all of it has left this
    // core, so a run that filled a ring would leave its receiver with nothing to take while the run was written; with
    // half, the receiver takes one half while the other is written.
    uint32_t max_run_blocks(uint32_t l) const { return max_run_blocks_[l]; }
    uint32_t entry_size() const { return entry_size_; }
    // Whether PREFETCH slot `slot` goes through these pipes: the same pipe for each of its lanes.
    bool carries(volatile tt_l1_ptr fifo::SlotHeader* slot) const {
        if (slot->num_lanes != num_lanes_) {
            return false;
        }
        for (uint32_t l = 0; l < num_lanes_; ++l) {
            if (LanePage(slot, l).header()->prefetch.target_config_page_addr != config_page_addr_[l]) {
                return false;
            }
        }
        return true;
    }

private:
    struct alignas(PrefetcherPipe) Storage {
        uint8_t bytes[sizeof(PrefetcherPipe)];
    };
    Storage storage_[TENSOR_PREFETCHER_LANES];
    uint32_t max_run_blocks_[TENSOR_PREFETCHER_LANES];
    uint32_t config_page_addr_[TENSOR_PREFETCHER_LANES];
    uint32_t entry_size_ = 0;
    uint32_t num_lanes_ = 0;
};

constexpr uint32_t kMaxRuns = TridQueue::kSlots;

// A run in flight: written to its receiver and not credited yet.
struct Run {
    uint32_t receiver = 0;
    uint32_t lane = 0;
    uint32_t receiver_in_lane = 0;
    uint32_t blocks = 0;
    // The bytes it takes in the receiver's share of the stage ring.
    uint32_t taken = 0;
};

// The pusher, which lives as long as the kernel does. Its runs stay in flight from one tensor to the next, and from
// one request to the next when that request is already queued through the same pipes, so a tensor's last runs leave
// while the next tensor's first ones go out; it waits for them all (finish) only before anything that needs the
// pipes idle or rebuilt.
class Pusher {
public:
    // Serves one request slot. False if termination was requested first.
    bool serve(const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
        const uint32_t num_lanes = slot->num_lanes;
        ASSERT(num_lanes > 0);
        volatile tt_l1_ptr tt::tt_metal::TensorPrefetcherRequestHeader* header = LanePage(slot, 0).header();
        if (header->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_DRAIN) {
            // A drain opens its own view of each pipe, so this one's must be closed first.
            if (!finish(requests)) {
                return false;
            }
            for (uint32_t l = 0; l < num_lanes; ++l) {
                if (!drain(requests, LanePage(slot, l))) {
                    return false;
                }
            }
            return true;
        }
        ASSERT(header->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH);
        if (pipes_ && !pipes_->carries(slot) && !finish(requests)) {
            return false;
        }
        const uint32_t num_tensors = header->prefetch.num_entries;
        TensorStream stream;
        for (uint32_t e = 0; e < num_tensors; ++e) {
            tensor_prefetcher::plan_tensor(slot, e, stream);
            if (stream.num_receivers != layout_receivers_) {
                // Every tensor of a request has the same receivers, so this is its first tensor. The reader lays the
                // stage ring out for them only once it is empty, and starts each share over, as the pusher does here.
                ASSERT(e == 0);
                if (!finish(requests)) {
                    return false;
                }
                layout_receivers_ = stream.num_receivers;
                for (uint32_t i = 0; i < fifo::kMaxReceivers; ++i) {
                    stage_[i].reset();
                }
            }
            // As many blocks as the first kernel planned, which a group selector may make fewer than the tensor holds.
            uint32_t blocks = 0;
            Noc noc;
            issue_tail(noc);
            if (!tensor_prefetcher::wait_for_tensor_blocks(
                    requests, slot, e, blocks, [&] { return credit_sent(noc); })) {
                abandon();
                return false;
            }
            // A selector that picks no group leaves nothing to send, and the pipes keep the entry size they had.
            if (blocks == 0) {
                continue;
            }
            if (!pipes_ || pipes_->entry_size() != stream.block_bytes) {
                if (!finish(requests)) {
                    return false;
                }
                pipes_.emplace(slot, num_lanes, stream.block_bytes);
            }
            if (!push_tensor(requests, stream, blocks)) {
                return false;
            }
        }
        // The runs stay in flight only if the next request is queued already and goes through these pipes; anything
        // else (a signal to wait for, a drain, nothing queued yet) may depend on the receivers having them all.
        // carries() also checks the slot has lanes, so its first page's header is there to read.
        volatile tt_l1_ptr fifo::SlotHeader* next = requests.next_slot();
        const bool next_continues =
            next != nullptr && next->command == fifo::kSlotRequest && pipes_ && pipes_->carries(next) &&
            LanePage(next, 0).header()->base.cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH;
        return next_continues || finish(requests);
    }

    // Credits every run in flight and closes the pipes, which persists their cursors. False if termination was
    // requested first.
    bool finish(const FifoConsumer& requests) {
        Noc noc;
        issue_tail(noc);
        WAYPOINT("TPFW");
        while (queue_.in_flight > 0) {
            if (!credit_sent(noc) && requests.terminate_requested()) {
                abandon();
                return false;
            }
        }
        WAYPOINT("TPFD");
        pipes_.reset();
        return true;
    }

    // At termination: waits for every write to leave and closes the pipes without crediting what is in flight.
    void abandon() {
        Noc noc;
        issue_tail(noc);
        for (uint32_t k = 0; k < kMaxRuns; ++k) {
            noc_async_write_flushed_with_trid(kTrids[k]);
        }
        queue_.in_flight = 0;
        pipes_.reset();
    }

private:
    // Credits the runs whose writes have left this core's L1, in the order they were written, which is the order
    // their receivers' cursors take them in. Each run's last packet carries the NoC flush tag, so its receiver commits
    // the run before the credit. Leaving L1 is also what frees the run's stage bytes. Returns whether any was credited.
    bool credit_sent(const Noc& noc) {
        bool credited = false;
        // The newest run waits for its held-back last packet (issue_tail) before it can leave.
        const uint32_t held_back = tail_pipe_ != nullptr ? 1 : 0;
        while (queue_.in_flight > held_back &&
               ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, kTrids[queue_.oldest])) {
            const Run& run = runs_[queue_.oldest];
            const uint32_t i = run.receiver;
            (*pipes_)[run.lane].push_back_to_receiver(run.receiver_in_lane, run.blocks, noc);
            pushed_[i] += run.taken;
            *region.pushed(i) = pushed_[i];
            --runs_to_[i];
            pending_blocks_[i] -= run.blocks;
            queue_.pop();
            credited = true;
        }
        return credited;
    }

    // Issues the newest run's flush-tagged last packet, if it is still held back.
    void issue_tail(const Noc& noc) {
        if (tail_pipe_ != nullptr) {
            tail_pipe_->write_to_receiver_tail<NocOptions::TXN_ID>(noc, tail_, {.trid = tail_trid_});
            tail_pipe_ = nullptr;
        }
    }

    // Writes `blocks` blocks of one tensor to each receiver as the reader stages them, and returns once every one is
    // written, with its last runs possibly still in flight. False if termination was requested first.
    bool push_tensor(const FifoConsumer& requests, const TensorStream& stream, uint32_t blocks) {
        Noc noc;
        LanePipes& pipes = *pipes_;
        uint32_t blocks_sent[fifo::kMaxReceivers];
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            blocks_sent[i] = 0;
        }
        // Each receiver gets an equal share of the run slots, rounded up: a receiver that always has blocks staged
        // does not take them all, and no slot stays idle while a receiver could use it.
        const uint32_t max_runs_per_receiver = (kMaxRuns + stream.num_receivers - 1) / stream.num_receivers;
        uint32_t to_write = stream.num_receivers * blocks;
        // The receiver a scan for new runs starts from, which moves on every scan.
        uint32_t first_receiver = 0;
        WAYPOINT("TPPW");
        while (to_write > 0) {
            bool progressed = credit_sent(noc);
            // Start runs while a slot is free: to each receiver with blocks staged and room in its ring, as many of
            // its staged blocks as lie back to back in its share and fit in its ring before the ring's end, in one
            // write placed after its runs in flight.
            bool waiting_on_receiver = false;
            uint32_t i = first_receiver;
            for (uint32_t n = 0; n < stream.num_receivers && !queue_.full();
                 ++n, i = i + 1 == stream.num_receivers ? 0 : i + 1) {
                if (blocks_sent[i] == blocks || runs_to_[i] == max_runs_per_receiver) {
                    continue;
                }
                StageCursor& cursor = stage_[i];
                const uint32_t first_taken = cursor.bytes_taken(stream);
                const uint32_t lane = stream.lane[i];
                const uint32_t blocks_left = blocks - blocks_sent[i];
                const uint32_t longest = cursor.run_blocks(
                    stream, blocks_left < pipes.max_run_blocks(lane) ? blocks_left : pipes.max_run_blocks(lane));
                uint32_t staged_bytes = staged_seen_[i] - written_[i];
                if (staged_bytes < cursor.run_bytes_taken(stream, longest)) {
                    staged_seen_[i] = *region.staged(i);
                    staged_bytes = staged_seen_[i] - written_[i];
                    if (staged_bytes < first_taken) {
                        continue;
                    }
                }
                // The reader stages whole runs, and a run never crosses the share's end, so every staged byte past
                // the first block's is another block's footprint up to the end of the share.
                uint32_t num_blocks = 1 + blocks_in(staged_bytes - first_taken, stream.block_footprint, longest - 1);
                PrefetcherPipe& pipe = pipes[lane];
                const uint32_t r = stream.receiver_in_lane[i];
                num_blocks = pipe.receiver_contiguous_space(r, num_blocks, pending_blocks_[i]);
                if (num_blocks == 0) {
                    waiting_on_receiver = true;
                    continue;
                }
                const uint32_t taken = cursor.run_bytes_taken(stream, num_blocks);
                const CoreLocalMem<uint8_t> src(stream.stage_addr(i, cursor.place_run(stream, num_blocks)));
                const uint32_t slot = queue_.next();
                // The previous run's last packet goes out now, so its payload streamed while this run was planned.
                issue_tail(noc);
                tail_ = pipe.write_to_receiver_head<NocOptions::TXN_ID>(
                    noc, r, src, num_blocks, {}, {.trid = kTrids[slot]}, pending_blocks_[i]);
                tail_pipe_ = &pipe;
                tail_trid_ = kTrids[slot];
                runs_[slot] = {
                    .receiver = i, .lane = lane, .receiver_in_lane = r, .blocks = num_blocks, .taken = taken};
                queue_.push();
                ++runs_to_[i];
                pending_blocks_[i] += num_blocks;
                written_[i] += taken;
                blocks_sent[i] += num_blocks;
                to_write -= num_blocks;
                progressed = true;
            }
            first_receiver = first_receiver + 1 == stream.num_receivers ? 0 : first_receiver + 1;
            if (!progressed) {
                if (tail_pipe_ != nullptr) {
                    // Nothing else to do: finish the newest run, so it can be credited.
                    issue_tail(noc);
                    continue;
                }
                // Nothing to credit or write. The waypoint says whether writes in flight are what is left (TPPF), a
                // staged block waits for its receiver to free ring space (TPPC), or every unfinished receiver with a
                // free run slot waits for the reader to stage its next block (TPPR).
                if (queue_.in_flight != 0) {
                    WAYPOINT("TPPF");
                } else if (waiting_on_receiver) {
                    WAYPOINT("TPPC");
                } else {
                    WAYPOINT("TPPR");
                }
                if (requests.terminate_requested()) {
                    abandon();
                    return false;
                }
                continue;
            }
            WAYPOINT("TPPW");
        }
        // The last run's last packet goes out now: the next tensor may be a while coming.
        issue_tail(noc);
        WAYPOINT("TPPD");
        return true;
    }

    // Waits for every receiver of each pipe a DRAIN page names to ack all it was sent. False if termination
    // was requested first.
    static bool drain(const FifoConsumer& requests, const LanePage& page) {
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

    // The pusher's place in each receiver's share of the stage ring, and the receiver count the shares are laid out
    // for.
    StageCursor stage_[fifo::kMaxReceivers];
    uint32_t layout_receivers_ = 0;
    // The pipes of the request being served, kept up for consecutive tensors and requests that use them with the same
    // block size.
    std::optional<LanePipes> pipes_;
    // Bytes of each receiver's share the pusher has published as sent (runs in flight not counted), the bytes it has
    // written (runs in flight counted), and each receiver's staged count as last read: it only grows, so the pusher
    // reads it again only when the count it has does not cover the longest run it could write.
    uint32_t pushed_[fifo::kMaxReceivers] = {};
    uint32_t written_[fifo::kMaxReceivers] = {};
    uint32_t staged_seen_[fifo::kMaxReceivers] = {};
    // The runs in flight, slot k's writes carrying kTrids[k]. And each receiver's share of them: how many, and their
    // blocks.
    Run runs_[kMaxRuns] = {};
    TridQueue queue_;
    uint32_t runs_to_[fifo::kMaxReceivers] = {};
    uint32_t pending_blocks_[fifo::kMaxReceivers] = {};
    // The newest run's flush-tagged last packet, not issued yet (write_to_receiver_head), the pipe it goes through
    // and its transaction id. It goes out before the next write, so the run's payload streams while the pusher
    // credits and plans.
    PrefetcherPipe::WriteTail tail_ = {};
    PrefetcherPipe* tail_pipe_ = nullptr;
    uint32_t tail_trid_ = 0;
};

// kernel_main keeps the Pusher on its stack, which shares the dispatch data-movement core's local memory
// (MEM_DISPATCH_DM_LOCAL_SIZE) with the kernel's thread-local globals, the firmware's frames and the frames of every
// call the pusher makes. At -O2 kernel_main's whole frame, the Pusher included, is about 1.7 KiB, and the deepest call
// under it (credit_sent) adds 160 B. A quarter of the local memory for the Pusher leaves the rest several times what
// they use.
static_assert(
    sizeof(Pusher) <= MEM_DISPATCH_DM_LOCAL_SIZE / 4,
    "The Pusher outgrows its share of the dispatch data-movement core's stack");

}  // namespace

void kernel_main() {
    // On the stack: the kernel runs until fast dispatch ends, and a global would need a destructor at exit.
    Pusher pusher;
    tensor_prefetcher::serve_requests(
        [&pusher](const FifoConsumer& requests, volatile tt_l1_ptr fifo::SlotHeader* slot) {
            return pusher.serve(requests, slot);
        });
    // Termination: a request may have left runs in flight for the next one.
    pusher.abandon();
}
