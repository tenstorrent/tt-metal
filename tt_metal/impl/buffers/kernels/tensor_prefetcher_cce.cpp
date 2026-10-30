// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar CCE Tensor prefetcher sender, the CCE counterpart of tensor_prefetcher.cpp. It stays resident
// and takes request pages from its TL1 request ring (tensor_prefetcher_cce_abi.hpp), in order, acking
// each on the ring's kernel line once it is handled:
//   * STOP:     exit.
//   * DRAIN:    for each listed GlobalCircularBuffer, wait until every receiver has acked everything this
//               sender credited it.
//   * WAIT_CQ:  wait until a CQ signal slot reaches the requested value (wrap-safe).
//   * PREFETCH: stream each listed tensor from this CCE's own DRAM view into a GlobalCircularBuffer's receivers,
//               as the Blackhole sender's receiver-contiguous batched round does. A PrefetcherPipe target is BadCmd.
// Waits are bounded by the wait budget, rdcycle cycles where 0 never expires. On an error the kernel
// writes its status, acks the page and exits, so the host stops waiting on it.

#include <stdint.h>

#include "experimental/cce_gddr.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_remote_cb.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_cce_abi.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

namespace hart = experimental::cce_hart;
namespace remote_cb = experimental::cce_remote_cb;

using tt::tt_metal::CceSenderState;
using tt::tt_metal::CceSenderStatus;
using tt::tt_metal::CceSenderStatusCode;
using tt::tt_metal::kRequestPageBytes;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherRequestHeader;
using tt::tt_metal::TensorPrefetcherTensorLayout;

namespace {

constexpr uint32_t kRequestPageWords = kRequestPageBytes / sizeof(uint32_t);

// What handling one request found. Everything but code is detail for a failure.
struct RequestOutcome {
    CceSenderStatusCode code = CceSenderStatusCode::Ok;
    uint32_t target_state_addr = 0;
    uint32_t receiver = 0;
    uint32_t sent = 0;
    uint32_t acked = 0;
};

// Stores the detail words, then the state behind a fence, so a host that reads the new state reads the
// detail that goes with it.
void publish_status(uint32_t status_addr, CceSenderState state, uint32_t request_index, const RequestOutcome& outcome) {
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, code)) = static_cast<uint32_t>(outcome.code);
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, request_index)) = request_index;
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, target_state_addr)) = outcome.target_state_addr;
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, receiver)) = outcome.receiver;
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, sent)) = outcome.sent;
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, acked)) = outcome.acked;
    asm volatile("fence w, w" ::: "memory");
    *hart::own_tl1_words(status_addr + offsetof(CceSenderStatus, state)) = static_cast<uint32_t>(state);
}

// DRAIN: the target words follow the header. A GlobalCircularBuffer's counters live in the sender state the
// word names, so they are read from there rather than from a request that may have moved on.
RequestOutcome drain_targets(
    const TensorPrefetcherRequestHeader& header, const uint32_t* page_words, uint64_t budget_cycles) {
    RequestOutcome outcome;
    const uint32_t num_targets = header.drain.num_targets;
    if (num_targets > tt::tt_metal::kDrainTargetsPerPage) {
        outcome.code = CceSenderStatusCode::BadCmd;
        return outcome;
    }
    const uint32_t* target_words = page_words + sizeof(TensorPrefetcherRequestHeader) / sizeof(uint32_t);
    for (uint32_t target = 0; target < num_targets; ++target) {
        const uint32_t target_word = target_words[target];
        outcome.target_state_addr = target_word;
        if ((target_word & tt::tt_metal::kDrainTargetPipeBit) != 0) {
            outcome.code = CceSenderStatusCode::BadCmd;
            return outcome;
        }
        const remote_cb::Sender sender = remote_cb::sender_init(target_word);
        if (remote_cb::barrier_bounded(sender, budget_cycles) == remote_cb::WaitResult::Ready) {
            continue;
        }
        outcome.code = CceSenderStatusCode::DrainTimeout;
        for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
            const uint32_t sent = remote_cb::receiver_pages_sent(sender, receiver);
            const uint32_t acked = remote_cb::receiver_pages_acked(sender, receiver);
            if (acked != sent) {
                outcome.receiver = receiver;
                outcome.sent = sent;
                outcome.acked = acked;
                break;
            }
        }
        return outcome;
    }
    return outcome;
}

// WAIT_CQ: the dispatcher bumps the slot; compare the unsigned difference as signed, so the count may wrap.
RequestOutcome wait_for_cq_signal(
    const TensorPrefetcherRequestHeader& header, uint32_t control_base, uint64_t budget_cycles) {
    RequestOutcome outcome;
    const uint32_t cq_index = header.wait_cq.cq_index;
    if (cq_index >= tt::tt_metal::kNumCqSignalSlots) {
        outcome.code = CceSenderStatusCode::BadCmd;
        outcome.receiver = cq_index;
        return outcome;
    }
    const volatile uint32_t* slot = hart::own_tl1_words(
        control_base + tt::tt_metal::kCceCqSignalSlotsOffset + cq_index * tt::tt_metal::kCceCqSignalSlotStrideBytes);
    const uint32_t wait_value = header.wait_cq.cq_wait_value;
    const hart::WaitDeadline deadline(budget_cycles);
    uint32_t observed = *slot;
    while (static_cast<int32_t>(observed - wait_value) < 0) {
        if (deadline.expired()) {
            outcome.code = CceSenderStatusCode::CqTimeout;
            outcome.receiver = cq_index;
            outcome.sent = wait_value;
            outcome.acked = observed;
            return outcome;
        }
        observed = *slot;
    }
    return outcome;
}

// What a PREFETCH needs beyond the request page: where to stage and where this sender's view starts.
struct PrefetchContext {
    uint32_t stage_ring_base;
    uint32_t stage_third;
    uint32_t mimir;
    uint64_t view_offset;
    uint64_t budget_cycles;
};

// Copies `bytes` at `view_byte_offset` in this sender's view into one stage slot, then from the slot into
// receiver `receiver`'s FIFO, `fifo_byte_offset` past wr_ptr. The credit that publishes them comes later.
void deliver_chunk(
    const PrefetchContext& context,
    const remote_cb::Sender& sender,
    uint32_t receiver,
    uint64_t view_byte_offset,
    uint32_t stage_slot_addr,
    uint32_t fifo_byte_offset,
    uint32_t bytes) {
    // No invalidate before the loads. On the Mimir emulator a CCE re-reading GDDR after each host rewrite saw no stale
    // word in four rounds (MM-7 spike S4, 2026-10-19); the AI-generated DV reference puts these SPAs on the hart's MMIO
    // port, which a Rocket core does not cache, but that is not confirmed against RTL. The host makes the tensor's
    // DRAM writes land before it releases a WAIT_CQ (SDMeshCommandQueue::enqueue_write_dram_core_counter).
    const volatile uint32_t* source = reinterpret_cast<const volatile uint32_t*>(
        experimental::cce_gddr_address(context.mimir, context.view_offset + view_byte_offset));
    volatile uint32_t* stage = hart::own_tl1_words(stage_slot_addr);
    const uint32_t num_words = bytes / sizeof(uint32_t);
    for (uint32_t word = 0; word < num_words; ++word) {
        stage[word] = source[word];
    }
    volatile uint32_t* destination =
        remote_cb::receiver_back_words(sender, receiver) + fifo_byte_offset / sizeof(uint32_t);
    for (uint32_t word = 0; word < num_words; ++word) {
        destination[word] = stage[word];
    }
}

// One receiver-contiguous tensor, batched: receiver r reads slab recv_index_base + r, whose block_count blocks of
// page_bytes_per_recv are contiguous. Each round takes the most blocks every receiver has room for (up to
// target_per_visit_pages, and never across the FIFO's page-aligned limit), copies that many from each slab in
// stage-third chunks, then credits the round. Each chunk is a load loop and a store loop, not an iDMA copy. The CCE
// has no NoC, so the Blackhole round's NoC reads and writes have no counterpart. Its iDMA moves the same data (MM-8
// probes in tests/tt_metal/tt_metal/dram/README.md), but its done fires when the request is accepted (MIM-628), so
// the sender could not order a credit behind the data without loading the receiver's TL1 back, and a CCE kernel load
// of remote TL1 hung the emulator in those probes.
RequestOutcome deliver_tensor(
    remote_cb::Sender& sender,
    const TensorPrefetcherEntry& entry,
    const TensorPrefetcherTensorLayout& layout,
    const PrefetchContext& context) {
    RequestOutcome outcome;
    const uint32_t page_bytes = layout.page_bytes_per_recv;
    const uint32_t chunk_bytes = layout.sub_chunk_bytes;
    if (page_bytes == 0 || page_bytes % remote_cb::kCreditUnitBytes != 0 || chunk_bytes == 0 ||
        chunk_bytes > context.stage_third || page_bytes % chunk_bytes != 0 || page_bytes > sender.fifo_size) {
        outcome.code = CceSenderStatusCode::BadLayout;
        return outcome;
    }
    remote_cb::resize_sender(sender, page_bytes);
    const uint32_t stage_chunk_bytes = (context.stage_third / chunk_bytes) * chunk_bytes;
    const uint32_t units_per_block = page_bytes / remote_cb::kCreditUnitBytes;
    uint32_t chunk_index = 0;
    uint32_t blocks_sent = 0;
    while (blocks_sent < layout.block_count) {
        if (remote_cb::reserve_back_bounded(sender, page_bytes, context.budget_cycles) ==
            remote_cb::WaitResult::Expired) {
            outcome.code = CceSenderStatusCode::CreditTimeout;
            outcome.receiver = sender.slowest_receiver;
            outcome.sent = remote_cb::receiver_pages_sent(sender, sender.slowest_receiver);
            outcome.acked = remote_cb::receiver_pages_acked(sender, sender.slowest_receiver);
            return outcome;
        }
        uint32_t round_blocks = layout.target_per_visit_pages;
        const uint32_t free_blocks = remote_cb::min_free(sender) / units_per_block;
        const uint32_t remaining_blocks = layout.block_count - blocks_sent;
        const uint32_t blocks_to_wrap = (sender.fifo_limit_page_aligned - sender.fifo_wr_ptr) / page_bytes;
        round_blocks = round_blocks < free_blocks ? round_blocks : free_blocks;
        round_blocks = round_blocks < remaining_blocks ? round_blocks : remaining_blocks;
        round_blocks = round_blocks < blocks_to_wrap ? round_blocks : blocks_to_wrap;
        // No visit size, or no whole block before the page-aligned limit: no round would ever move a block.
        if (round_blocks == 0) {
            outcome.code = CceSenderStatusCode::BadLayout;
            return outcome;
        }
        const uint32_t round_bytes = round_blocks * page_bytes;
        for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
            const uint64_t round_source =
                static_cast<uint64_t>(entry.bank_local_base) +
                static_cast<uint64_t>(layout.recv_index_base + receiver) * layout.recv_stride_bytes +
                static_cast<uint64_t>(blocks_sent) * page_bytes;
            for (uint32_t offset = 0; offset < round_bytes; offset += stage_chunk_bytes) {
                const uint32_t bytes =
                    round_bytes - offset < stage_chunk_bytes ? round_bytes - offset : stage_chunk_bytes;
                deliver_chunk(
                    context,
                    sender,
                    receiver,
                    round_source + offset,
                    context.stage_ring_base + (chunk_index % 3) * context.stage_third,
                    offset,
                    bytes);
                ++chunk_index;
            }
        }
        remote_cb::finalize(sender, round_bytes);
        blocks_sent += round_blocks;
    }
    return outcome;
}

// PREFETCH: entries follow the header and name their layout slots by byte offset (tensor_prefetcher_request.hpp).
// The write pointer goes back to the state block on every exit, so it matches what this request credited.
RequestOutcome prefetch_request(
    const TensorPrefetcherRequestHeader& header, const uint32_t* page_words, const PrefetchContext& context) {
    RequestOutcome outcome;
    outcome.target_state_addr = header.prefetch.target_state_addr;
    if (header.prefetch.transport != tt::tt_metal::TENSOR_PREFETCHER_TRANSPORT_GLOBAL_CB) {
        outcome.code = CceSenderStatusCode::BadCmd;
        return outcome;
    }
    const uint8_t* page_bytes = reinterpret_cast<const uint8_t*>(page_words);
    const uint32_t num_entries = header.prefetch.num_entries;
    if (sizeof(TensorPrefetcherRequestHeader) + num_entries * sizeof(TensorPrefetcherEntry) > kRequestPageBytes) {
        outcome.code = CceSenderStatusCode::BadCmd;
        return outcome;
    }
    remote_cb::Sender sender = remote_cb::sender_init(header.prefetch.target_state_addr);
    for (uint32_t entry_index = 0; entry_index < num_entries; ++entry_index) {
        TensorPrefetcherEntry entry;
        __builtin_memcpy(
            &entry,
            page_bytes + sizeof(TensorPrefetcherRequestHeader) + entry_index * sizeof(TensorPrefetcherEntry),
            sizeof(entry));
        if (entry.layout_offset > kRequestPageBytes - sizeof(TensorPrefetcherTensorLayout)) {
            outcome.code = CceSenderStatusCode::BadCmd;
            break;
        }
        TensorPrefetcherTensorLayout layout;
        __builtin_memcpy(&layout, page_bytes + entry.layout_offset, sizeof(layout));
        if (layout.layout_mode != tt::tt_metal::kCceReceiverContiguousLayoutMode || layout.streaming != 0) {
            outcome.code = CceSenderStatusCode::BadLayout;
            break;
        }
        const RequestOutcome tensor_outcome = deliver_tensor(sender, entry, layout, context);
        if (tensor_outcome.code != CceSenderStatusCode::Ok) {
            outcome.code = tensor_outcome.code;
            outcome.receiver = tensor_outcome.receiver;
            outcome.sent = tensor_outcome.sent;
            outcome.acked = tensor_outcome.acked;
            break;
        }
    }
    remote_cb::store_sender_wr_ptr(sender);
    return outcome;
}

}  // namespace

void kernel_main() {
    const uint32_t control_base = get_arg_val<uint32_t>(tt::tt_metal::kCceSenderArgControlBase);
    const uint64_t budget_cycles =
        hart::arg_u64(tt::tt_metal::kCceSenderArgBudgetLo, tt::tt_metal::kCceSenderArgBudgetHi);
    const uint32_t stage_ring_bytes = get_arg_val<uint32_t>(tt::tt_metal::kCceSenderArgStageRingBytes);
    const PrefetchContext prefetch_context{
        .stage_ring_base = control_base + tt::tt_metal::kCceSenderControlBytes,
        .stage_third = stage_ring_bytes / 3,
        .mimir = get_arg_val<uint32_t>(tt::tt_metal::kCceSenderArgMimir),
        .view_offset = hart::arg_u64(tt::tt_metal::kCceSenderArgViewOffsetLo, tt::tt_metal::kCceSenderArgViewOffsetHi),
        .budget_cycles = budget_cycles,
    };

    const uint32_t ring_meta_addr = control_base + tt::tt_metal::kCceRingMetaOffset;
    const volatile uint32_t* bytes_sent =
        hart::own_tl1_words(ring_meta_addr + offsetof(tt::tt_metal::RingMeta, bytes_sent));
    volatile uint32_t* bytes_acked =
        hart::own_tl1_words(ring_meta_addr + offsetof(tt::tt_metal::RingMeta, bytes_acked));
    const uint32_t status_addr = control_base + tt::tt_metal::kCceSenderStatusOffset;
    const uint32_t ring_addr = control_base + tt::tt_metal::kCceRequestRingOffset;

    // The host zeroed both ring counts before launch.
    uint32_t bytes_consumed = 0;
    uint32_t request_index = 0;
    publish_status(status_addr, CceSenderState::Running, request_index, RequestOutcome{});

    while (true) {
        while (*bytes_sent == bytes_consumed) {
        }
        // The host stored the page before the count; read the page only after the count.
        asm volatile("fence r, r" ::: "memory");
        const uint32_t slot = (bytes_consumed / kRequestPageBytes) % tt::tt_metal::kCceRequestRingSlots;
        const volatile uint32_t* slot_words = hart::own_tl1_words(ring_addr + slot * kRequestPageBytes);
        // Copy the page word by word: packed header fields read in place could become byte loads.
        uint32_t page_words[kRequestPageWords];
        for (uint32_t word = 0; word < kRequestPageWords; ++word) {
            page_words[word] = slot_words[word];
        }
        TensorPrefetcherRequestHeader header;
        __builtin_memcpy(&header, page_words, sizeof(header));

        bool stop = false;
        bool left_running = false;
        RequestOutcome outcome;
        switch (header.base.cmd_id) {
            case tt::tt_metal::DRAM_PREFETCHER_CMD_STOP: stop = true; break;
            case tt::tt_metal::DRAM_PREFETCHER_CMD_DRAIN:
                publish_status(status_addr, CceSenderState::Draining, request_index, RequestOutcome{});
                left_running = true;
                outcome = drain_targets(header, page_words, budget_cycles);
                break;
            case tt::tt_metal::DRAM_PREFETCHER_CMD_WAIT_CQ:
                publish_status(status_addr, CceSenderState::WaitingCq, request_index, RequestOutcome{});
                left_running = true;
                outcome = wait_for_cq_signal(header, control_base, budget_cycles);
                break;
            case tt::tt_metal::DRAM_PREFETCHER_CMD_PREFETCH:
                outcome = prefetch_request(header, page_words, prefetch_context);
                break;
            default: outcome.code = CceSenderStatusCode::BadCmd; break;
        }
        bytes_consumed += kRequestPageBytes;

        const bool failed = outcome.code != CceSenderStatusCode::Ok;
        if (stop || failed) {
            publish_status(
                status_addr, failed ? CceSenderState::Failed : CceSenderState::Stopped, request_index, outcome);
        } else if (left_running) {
            publish_status(status_addr, CceSenderState::Running, request_index + 1, RequestOutcome{});
        }
        // Every read of the slot comes before the ack that lets the host reuse it.
        asm volatile("fence rw, w" ::: "memory");
        *bytes_acked = bytes_consumed;
        if (stop || failed) {
            return;
        }
        ++request_index;
    }
}
