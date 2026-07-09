// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Queueable DRISC prefetcher kernel — successor to tensor_prefetcher.cpp.
// Sits in a request loop on a per-(device, sender-core) H2D socket; each request
// payload identifies the target GlobalCircularBuffer (by its DRISC L1
// sender-state-block base, written by the GCB ctor) and carries the per-tensor
// geometry. The kernel loads the sender state block's RemoteSenderCBInterface
// region into cb_interface[], runs the chunk-loop logic, writes the mutable
// fifo_wr_ptr back to L1 so the next request to the same GCB resumes from the right
// ring offset, and acks the socket page.
//
// Request page wire format (one socket page): a TensorPrefetcherRequestHeader
// (one-byte command id + per-command union). The STOP command (all-zero page) exits
// the request loop; WAIT_CQ blocks on a per-CQ signal slot; PREFETCH is followed by a
// forward-growing table of per-tensor TensorPrefetcherEntry (address + layout index)
// and a backward-growing (from the end of the payload) deduplicated table of
// TensorPrefetcherTensorLayout. The kernel walks the entries in order, resolving each
// entry's geometry from the referenced layout. See
// tt_metal/impl/buffers/tensor_prefetcher_request.hpp.
//
// Per-GCB sender state block layout: see
// tt_metal/impl/buffers/dram_sender_state_block.hpp.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "api/socket_api.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"
#include "tt_metal/impl/buffers/dram_sender_state_block.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

using tt::tt_metal::DramSenderStateBlock;
using tt::tt_metal::kNumCqSignalSlots;
using tt::tt_metal::kRequestPageBytes;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherRequestHeader;
using tt::tt_metal::TensorPrefetcherTensorLayout;

// Per-stage cycle profiling — gated on watcher ring-buffer being enabled so
// production builds pay zero (the variables and timestamp reads don't exist).
// Decode tags ("0xA1 .. 0xAA") match the dump reader in
// tensor_prefetcher_drisc_profile.md.
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)
#include "api/debug/ring_buffer.h"
#include "internal/tt-1xx/risc_common.h"
#define PROF_DECL_ACC(name) uint32_t name = 0
#define PROF_DECL_TS(name) uint32_t name = 0
#define PROF_TICK(x) x = get_timestamp_32b()
#define PROF_ACC(acc, t_end, t_start) acc += (t_end) - (t_start)
#define PROF_INC(counter) ++counter
#define PROF_DUMP(tag, val) WATCHER_RING_BUFFER_PUSH(((tag) << 24) | ((val) & 0x00FFFFFFu))
#else
#define PROF_DECL_ACC(name)
#define PROF_DECL_TS(name)
#define PROF_TICK(x) ((void)0)
#define PROF_ACC(acc, t_end, t_start) ((void)0)
#define PROF_INC(counter) ((void)0)
#define PROF_DUMP(tag, val) ((void)0)
#endif

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

namespace {

template <bool single_row, bool single_page>
FORCE_INLINE void prefetcher_write_chunk(
    uint32_t src_l1_addr,
    uint32_t dest_l1_base,
    volatile tt_l1_ptr uint32_t* recv_xy,
    uint32_t num_receivers_in_chunk,
    uint32_t num_rows,
    uint32_t coalesced_num_pages_per_row,
    uint32_t coalesced_page_size,
    uint8_t noc) {
    const uint32_t row_bytes_per_recv = coalesced_num_pages_per_row * coalesced_page_size;
    const uint32_t row_stride_in_stage = row_bytes_per_recv * num_receivers_in_chunk;

    uint32_t recv_src_offset = 0;
    for (uint32_t i = 0; i < num_receivers_in_chunk; ++i) {
        const uint32_t remote_noc_xy =
            uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, recv_xy[2 * i]), DYNAMIC_NOC_Y(noc, recv_xy[2 * i + 1])));
        uint32_t dest_addr = dest_l1_base;
        const uint64_t set_state_dest = get_noc_addr_helper(remote_noc_xy, dest_addr);
        if constexpr (!(single_row && single_page)) {
            noc_async_write_one_packet_set_state</*posted=*/true>(set_state_dest, coalesced_page_size, noc);
        }

        uint32_t src_addr = src_l1_addr + recv_src_offset;
        if constexpr (single_row && single_page) {
            noc_async_write_one_packet</*enable_noc_tracing=*/false, /*posted=*/true>(
                src_addr, set_state_dest, coalesced_page_size, noc);
        } else if constexpr (single_row) {
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                const uint64_t dest_noc = get_noc_addr_helper(remote_noc_xy, dest_addr);
                noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_noc, noc);
                src_addr += coalesced_page_size;
                dest_addr += coalesced_page_size;
            }
        } else if constexpr (single_page) {
            for (uint32_t h = 0; h < num_rows; ++h) {
                const uint64_t dest_noc = get_noc_addr_helper(remote_noc_xy, dest_addr);
                noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_noc, noc);
                src_addr += row_stride_in_stage;
                dest_addr += coalesced_page_size;
            }
        } else {
            for (uint32_t h = 0; h < num_rows; ++h) {
                const uint32_t row_src_start = src_addr;
                for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                    const uint64_t dest_noc = get_noc_addr_helper(remote_noc_xy, dest_addr);
                    noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_noc, noc);
                    src_addr += coalesced_page_size;
                    dest_addr += coalesced_page_size;
                }
                src_addr = row_src_start + row_stride_in_stage;
            }
        }
        recv_src_offset += row_bytes_per_recv;
    }
}

template <bool skip_ptr_update>
FORCE_INLINE void prefetcher_finalize_block(
    RemoteSenderCBInterface& iface, uint32_t page_bytes_per_recv, uint32_t num_receivers, uint8_t noc) {
    uint32_t len_bytes = page_bytes_per_recv;
    uint32_t next_wr_ptr = iface.fifo_wr_ptr + page_bytes_per_recv;
    if (next_wr_ptr >= iface.fifo_limit_page_aligned) {
        const uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr)[3];
        len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
        next_wr_ptr = iface.fifo_start_addr + (next_wr_ptr - iface.fifo_limit_page_aligned);
    }
    const uint32_t fifo_pages_sent = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

    volatile tt_l1_ptr uint32_t* local_pages_sent =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
    uint32_t remote_sent_base = remote_cb_remote_pages_sent_ptr(iface.num_receivers_and_remote_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* recv_xy_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        const uint32_t remote_noc_xy =
            uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, recv_xy_ptr[0]), DYNAMIC_NOC_Y(noc, recv_xy_ptr[1])));
        *local_pages_sent += fifo_pages_sent;
        const uint64_t remote_sent_addr = get_noc_addr_helper(remote_noc_xy, remote_sent_base);
        noc_semaphore_inc<skip_ptr_update>(remote_sent_addr, fifo_pages_sent, noc);
        local_pages_sent += experimental::REMOTE_CB_LOCAL_PAGES_STRIDE / sizeof(uint32_t);
        remote_sent_base += 2 * L1_ALIGNMENT;
        recv_xy_ptr += 2;
    }
    iface.fifo_wr_ptr = next_wr_ptr;
}

// Non-blocking variant of remote_cb_reserve_back's polling loop: scans all
// receivers' (pages_sent - pages_acked) and returns the min free aligned-page
// count. Used by the recv-contig batched main loop to size the next round.
FORCE_INLINE uint32_t poll_min_free_aligned_pages(const RemoteSenderCBInterface& iface, uint32_t num_receivers) {
    const uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr)[3];
    const uint32_t fifo_aligned_num_pages = fifo_size / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        iface.aligned_pages_sent_ptr + experimental::REMOTE_CB_LOCAL_PAGES_ACKED_OFFSET);
    uint32_t min_free = fifo_aligned_num_pages;
    invalidate_l1_cache();
    for (uint32_t i = 0; i < num_receivers; ++i) {
        const uint32_t sent_minus_ack = *pages_sent_ptr - *pages_acked_ptr;
        // Clamp: a resize padding credit (pages_sent bumped without a free-space reserve) can
        // transiently push sent ahead of acked by more than the fifo holds; an unclamped
        // subtraction would underflow to a huge value and defeat receiver backpressure.
        const uint32_t free_pages =
            sent_minus_ack >= fifo_aligned_num_pages ? 0u : fifo_aligned_num_pages - sent_minus_ack;
        if (free_pages < min_free) {
            min_free = free_pages;
        }
        pages_sent_ptr += experimental::REMOTE_CB_LOCAL_PAGES_STRIDE / sizeof(uint32_t);
        pages_acked_ptr += experimental::REMOTE_CB_LOCAL_PAGES_STRIDE / sizeof(uint32_t);
    }
    return min_free;
}

// Loads the per-GCB sender state block's RemoteSenderCBInterface-compatible region
// from L1 into the static cb_interface[] slot for this request.
FORCE_INLINE void load_sender_state(volatile tt_l1_ptr DramSenderStateBlock* sb, RemoteSenderCBInterface& iface) {
    iface.config_ptr = sb->config_ptr;
    iface.fifo_start_addr = sb->fifo_start_addr;
    iface.fifo_wr_ptr = sb->fifo_wr_ptr;
    iface.receiver_noc_xy_ptr = sb->receiver_noc_xy_ptr;
    iface.aligned_pages_sent_ptr = sb->aligned_pages_sent_ptr;
    iface.num_receivers_and_remote_pages_sent_ptr = sb->num_receivers_and_remote_pages_sent_ptr;
    // fifo_limit_page_aligned / fifo_page_size are intentionally not loaded: they're
    // recomputed per-tensor by resize_remote_sender_cb_interface before any use.
}

// Writes back only the field that needs to persist across requests targeting this
// GCB. Per-tensor fields (fifo_limit / fifo_page_size) are overwritten by
// resize_remote_sender_cb_interface on every new request, so we don't round-trip them.
FORCE_INLINE void store_sender_state(
    volatile tt_l1_ptr DramSenderStateBlock* sb, const RemoteSenderCBInterface& iface) {
    sb->fifo_wr_ptr = iface.fifo_wr_ptr;
}

}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    // num_receivers used to live here, but it's now per-GCB: each request's state
    // block carries its own num_receivers (DramSenderStateBlock::num_receivers).
    // Different GCBs queued against the same prefetcher can have different receiver
    // counts.
    constexpr uint32_t stage_ring_base = get_compile_time_arg_val(0);
    constexpr uint32_t stage_ring_size = get_compile_time_arg_val(1);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t socket_page_size = get_compile_time_arg_val(3);
    // Base of this core's per-CQ signal slots (kNumCqSignalSlots uint32 counters).
    // WaitForCqOnTensorPrefetcher writes an incrementing value here from the
    // dispatcher; a WAIT_CQ request blocks until the requested slot reaches it.
    constexpr uint32_t cq_signal_l1_base = get_compile_time_arg_val(4);
    constexpr uint32_t ring_half = stage_ring_size / 2;
    constexpr uint32_t stage_slot_a = stage_ring_base;
    constexpr uint32_t stage_slot_b = stage_ring_base + ring_half;
    // Receiver-contiguous path uses 3 stage slots (vs K-row's 2 halves) so the
    // NIU has 2 iters to drain a slot's posted writes before that slot is
    // reused for the next DMA — flush latency drops from per-iter-blocking to
    // amortized over N-1 iters of DMA + write-enqueue.
    constexpr uint32_t stage_third = stage_ring_size / 3;
    constexpr uint32_t stage_slot_0 = stage_ring_base;
    constexpr uint32_t stage_slot_1 = stage_ring_base + stage_third;
    constexpr uint32_t stage_slot_2 = stage_ring_base + 2 * stage_third;

    // ---- Runtime args ----
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);
    (void)bank_id;
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_idx++);

    // ---- Init ----
    SocketReceiverInterface socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(socket, socket_page_size);

    experimental::drisc_set_stream_mode();
    RemoteSenderCBInterface& iface = get_remote_sender_cb_interface(remote_cb_id);
    bool has_loaded_sender_state = false;

    // Zero the per-CQ signal slots before parking on the socket. Safe to do here
    // (rather than from the host) because no WaitForCqOnTensorPrefetcher signal
    // can be enqueued until StartTensorPrefetcher returns to the single-threaded
    // host caller, long after this init runs.
    volatile tt_l1_ptr uint32_t* cq_signal_slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cq_signal_l1_base);
    for (uint32_t i = 0; i < kNumCqSignalSlots; ++i) {
        cq_signal_slots[i] = 0;
    }

    // ---- Request loop ----
    while (true) {
        socket_wait_for_pages(socket, 1);

        volatile tt_l1_ptr TensorPrefetcherRequestHeader* req =
            reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherRequestHeader*>(socket.read_ptr);
        const uint8_t cmd_id = req->base.cmd_id;
        if (cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_STOP) {
            // Stop sentinel. Receiver pages_acked atomics target DRISC L1 while
            // stream mode is active; wait for the last loaded GCB to drain before
            // exiting the request loop and restoring NoC2AXI mode.
            if (has_loaded_sender_state) {
                experimental::remote_cb_sender_barrier(remote_cb_id);
            }
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            break;
        }
        if (cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_WAIT_CQ) {
            // Block until the dispatcher has bumped this CQ's signal slot to the
            // requested value. Wrap-safe: compare the unsigned difference as signed.
            const uint32_t idx = req->wait_cq.cq_index;
            const uint32_t target = req->wait_cq.cq_wait_value;
            while ((int32_t)(cq_signal_slots[idx] - target) < 0) {
                invalidate_l1_cache();
            }
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            continue;
        }
        // DRAM_PREFETCHER_CMD_PREFETCH
        const uint32_t req_num_entries = req->prefetch.num_entries;
        const uint32_t gcb_state_addr = req->prefetch.gcb_state_addr;
        volatile tt_l1_ptr DramSenderStateBlock* state =
            reinterpret_cast<volatile tt_l1_ptr DramSenderStateBlock*>(gcb_state_addr);

        PROF_DECL_ACC(prof_rounds);
        PROF_DECL_ACC(prof_chunks);
        PROF_DECL_ACC(prof_poll);
        PROF_DECL_ACC(prof_dma_issue);
        PROF_DECL_ACC(prof_dma_wait);
        PROF_DECL_ACC(prof_set_state);
        PROF_DECL_ACC(prof_writes);
        PROF_DECL_ACC(prof_flush);
        PROF_DECL_ACC(prof_defer_flush);
        PROF_DECL_ACC(prof_finalize);

        load_sender_state(state, iface);
        has_loaded_sender_state = true;
        // num_receivers lives inside the GCB's state block (set by the GCB ctor).
        // Reading it per request lets a single prefetcher serve GCBs with different
        // receiver counts.
        const uint32_t num_receivers = state->num_receivers;
        // Bank-local slab index of this sender's first receiver. When two DRISC cores
        // split a bank's receivers, the second core's local receiver r maps to bank-local
        // slab (recv_index_base + r). 0 for a single sender. Receiver-contiguous only.
        const uint32_t recv_index_base = state->recv_index_base;

        // Entries follow the header (grow forward); the deduplicated layout table grows
        // backward from the end of the payload, so layout i lives at read_ptr +
        // kRequestPageBytes - (i+1)*sizeof(layout). See tensor_prefetcher_request.hpp.
        volatile tt_l1_ptr TensorPrefetcherEntry* entries = reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherEntry*>(
            socket.read_ptr + sizeof(TensorPrefetcherRequestHeader));
        const uint32_t layout_table_end = socket.read_ptr + kRequestPageBytes;

        for (uint32_t e = 0; e < req_num_entries; ++e) {
            const uint32_t tensor_base = entries[e].bank_local_base;
            volatile tt_l1_ptr TensorPrefetcherTensorLayout* g =
                reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherTensorLayout*>(
                    layout_table_end - (entries[e].layout_index + 1) * sizeof(TensorPrefetcherTensorLayout));
            const uint32_t t_num_sub = g->num_sub;
            const uint32_t t_M = g->M;
            const uint32_t t_rows_per_sub = g->rows_per_sub;
            const uint32_t t_coal_page_size = g->coalesced_page_size;
            const uint32_t t_coal_num_pages = g->coalesced_num_pages;
            const uint32_t t_chunk_bytes = g->sub_chunk_bytes;
            const uint32_t t_sub_stride = g->sub_stride_bytes;
            const uint32_t t_block_stride = g->block_stride_bytes;
            const uint32_t t_page_bytes_per_recv = g->page_bytes_per_recv;
            const uint32_t t_layout_mode = g->layout_mode;
            const uint32_t t_target_per_visit = g->target_per_visit_pages;
            const uint32_t t_recv_stride = g->recv_stride_bytes;
            const uint32_t t_block_count = g->block_count;

            // Set the sender fifo page size to one full per-receiver page. When resize skips
            // padding to reach the next aligned page (e.g. a larger page after a smaller one in a
            // mixed-page-size stream), credit that skip to the receiver over NoC so sender and
            // receiver pointers stay in lockstep; otherwise the next receiver is short of credits.
            experimental::resize_remote_sender_cb_interface</*update_remote_over_noc=*/true>(
                remote_cb_id, t_page_bytes_per_recv, noc_index);

            // stage_slot ping-pongs between stage_slot_a and stage_slot_b; toggle via
            // `(a + b) - slot`.
            constexpr uint32_t stage_slot_sum = stage_slot_a + stage_slot_b;

            // Branch on layout. The branch is per-tensor (100% predictable across the
            // inner chunk loop), so adding the second branch costs nothing in the
            // hot path.
            if (t_layout_mode == 0) {
                // ---- K-row-major main loop ----
                const uint32_t t_recv_per_chunk = num_receivers / t_M;
                const uint32_t t_sub_band_per_block = t_num_sub * t_M;
                const uint32_t total_chunks = t_block_count * t_sub_band_per_block;

                // Prologue: issue first DMA (chunk 0) into stage_slot_a. The body's
                // "issue next" lookahead populates subsequent slots.
                experimental::dma_async_read(/*stream=*/0, tensor_base, stage_slot_a, t_chunk_bytes);

                uint32_t fifo_snapshot = 0;
                uint32_t cum_offset_in_page = 0;
                // The flat chunk index `c` decomposes into three nested counters,
                // advanced by compare-and-increment (ch fastest, then sb, then blk):
                uint32_t blk = 0;  // K-block index within the tensor, in [0, t_block_count)
                uint32_t sb = 0;   // sub-band index within the block, in [0, t_num_sub)
                uint32_t ch = 0;   // chunk index within the sub-band, in [0, t_M)
                uint32_t stage_slot = stage_slot_a;
                // True for every chunk except the very last; flipped once the successor
                // counters cross t_block_count, so the hot path reads a flag instead of
                // recomputing the bound each iteration.
                bool has_next = (total_chunks > 1);

                for (uint32_t c = 0; c < total_chunks; ++c) {
                    if (sb == 0 && ch == 0) {
                        experimental::remote_cb_reserve_back(remote_cb_id, 1);
                        fifo_snapshot = iface.fifo_wr_ptr;
                        cum_offset_in_page = 0;
                    }

                    // Compute the successor (blk, sb, ch) by incrementing the nested counters.
                    uint32_t next_ch = ch + 1;
                    uint32_t next_sb = sb;
                    uint32_t next_blk = blk;
                    if (next_ch == t_M) {
                        next_ch = 0;
                        ++next_sb;
                        if (next_sb == t_num_sub) {
                            next_sb = 0;
                            ++next_blk;
                            if (next_blk == t_block_count) {
                                has_next = false;
                            }
                        }
                    }
                    const uint32_t next_slot = stage_slot_sum - stage_slot;

                    // Issue the next DMA before waiting on this one (ping-pong, depth 2).
                    if (has_next) {
                        const uint32_t next_src =
                            tensor_base + next_blk * t_block_stride + next_sb * t_sub_stride + next_ch * t_chunk_bytes;
                        experimental::dma_async_read(/*stream=*/0, next_src, next_slot, t_chunk_bytes);
                    }
                    const uint32_t outstanding_after_wait = has_next ? 1u : 0u;
                    experimental::dma_async_read_wait_n(/*stream=*/0, outstanding_after_wait);
                    volatile tt_l1_ptr uint32_t* chunk_recv_xy =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr) +
                        ch * t_recv_per_chunk * 2;
                    // Per-tensor-stable predicate; branches are 100% predictable across the
                    // chunk loop. Compiler folds the fast-path bodies via `if constexpr` in
                    // `prefetcher_write_chunk`.
                    if (t_rows_per_sub == 1) {
                        if (t_coal_num_pages == 1) {
                            prefetcher_write_chunk</*single_row=*/true, /*single_page=*/true>(
                                stage_slot,
                                fifo_snapshot + cum_offset_in_page,
                                chunk_recv_xy,
                                t_recv_per_chunk,
                                t_rows_per_sub,
                                t_coal_num_pages,
                                t_coal_page_size,
                                noc_index);
                        } else {
                            prefetcher_write_chunk</*single_row=*/true, /*single_page=*/false>(
                                stage_slot,
                                fifo_snapshot + cum_offset_in_page,
                                chunk_recv_xy,
                                t_recv_per_chunk,
                                t_rows_per_sub,
                                t_coal_num_pages,
                                t_coal_page_size,
                                noc_index);
                        }
                    } else {
                        if (t_coal_num_pages == 1) {
                            prefetcher_write_chunk</*single_row=*/false, /*single_page=*/true>(
                                stage_slot,
                                fifo_snapshot + cum_offset_in_page,
                                chunk_recv_xy,
                                t_recv_per_chunk,
                                t_rows_per_sub,
                                t_coal_num_pages,
                                t_coal_page_size,
                                noc_index);
                        } else {
                            prefetcher_write_chunk</*single_row=*/false, /*single_page=*/false>(
                                stage_slot,
                                fifo_snapshot + cum_offset_in_page,
                                chunk_recv_xy,
                                t_recv_per_chunk,
                                t_rows_per_sub,
                                t_coal_num_pages,
                                t_coal_page_size,
                                noc_index);
                        }
                    }

                    if (ch + 1 == t_M) {
                        cum_offset_in_page += t_rows_per_sub * t_coal_num_pages * t_coal_page_size;
                    }

                    if (sb + 1 == t_num_sub && ch + 1 == t_M) {
                        noc_async_posted_writes_flushed();
                        prefetcher_finalize_block</*skip_ptr_update=*/true>(
                            iface, t_page_bytes_per_recv, num_receivers, noc_index);
                    } else {
                        // The ping-pong DMA can reuse this stage slot two chunks later.
                        // Make sure all posted writes sourced from it have departed first.
                        noc_async_posted_writes_flushed();
                    }

                    // Advance counters to next chunk.
                    blk = next_blk;
                    sb = next_sb;
                    ch = next_ch;
                    stage_slot = next_slot;
                }
            } else {
                // ---- Receiver-contiguous main loop (dynamic batching + 3-slot pipeline) ----
                // Per round, we pick B = min(target_per_visit,
                // min_free_blocks_across_receivers, remaining_blocks, blocks_to_wrap).
                // Then for each receiver r in turn we DMA + NoC-write B blocks of
                // receiver r's slab (contiguous: slab = recv_stride, block stride =
                // page_bytes_per_recv). set_state once per receiver visit; with_state
                // for every packet — amortizes destination reprogramming.
                //
                // Stage layout is 3 thirds rotating (vs the K-row path's 2 halves).
                // Per iter we DMA into slot (gc+1)%3 and write from slot gc%3. The slot
                // we're about to overwrite at iter gc+1 was last written at iter gc-2,
                // giving the NIU ~2 iters of DMA+enqueue overhead to drain those
                // posted writes. We flush only RIGHT BEFORE the next DMA-into-reused-slot
                // (gc >= 2), turning the per-iter blocking flush (304 cyc on 8B_FF1_1d)
                // into mostly free overlapped work.
                //
                // After all num_receivers visits, finalize bumps pages_sent + remote
                // semaphore by B for every receiver and advances iface.fifo_wr_ptr by
                // B * t_page_bytes_per_recv. The pages_to_wrap clamp prevents B from
                // crossing fifo_limit_page_aligned mid-round.
                constexpr uint32_t kStageSlotBytes = stage_third;
                // Per-stage-slot capacity expressed in whole sub-bands; t_chunk_bytes
                // (= sub_chunk_bytes) is the natural alignment for both DMA and NoC
                // writes — it's a multiple of t_coal_page_size by construction, so
                // packet counts come out exact (no truncation loss).
                const uint32_t subs_per_stage_slot = kStageSlotBytes / t_chunk_bytes;
                const uint32_t max_chunk_bytes = subs_per_stage_slot * t_chunk_bytes;

                // fifo_pages_per_block converts aligned-page free space into
                // "blocks of t_page_bytes_per_recv" — the batch unit B uses.
                const uint32_t fifo_pages_per_block = t_page_bytes_per_recv / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

                constexpr uint32_t slot_addrs[3] = {stage_slot_0, stage_slot_1, stage_slot_2};

                uint32_t pages_sent_global = 0;

                while (pages_sent_global < t_block_count) {
                    PROF_INC(prof_rounds);
                    // ---- Pick B for this round ----
                    uint32_t min_free_blocks;
                    PROF_DECL_TS(t_poll_start);
                    PROF_DECL_TS(t_poll_end);
                    PROF_TICK(t_poll_start);
                    do {
                        const uint32_t min_free_aligned = poll_min_free_aligned_pages(iface, num_receivers);
                        min_free_blocks = min_free_aligned / fifo_pages_per_block;
                    } while (min_free_blocks == 0);
                    PROF_TICK(t_poll_end);
                    PROF_ACC(prof_poll, t_poll_end, t_poll_start);

                    const uint32_t bytes_to_wrap = iface.fifo_limit_page_aligned - iface.fifo_wr_ptr;
                    const uint32_t pages_to_wrap = bytes_to_wrap / t_page_bytes_per_recv;
                    const uint32_t remaining = t_block_count - pages_sent_global;

                    uint32_t B = t_target_per_visit;
                    if (B > min_free_blocks) {
                        B = min_free_blocks;
                    }
                    if (B > remaining) {
                        B = remaining;
                    }
                    if (B > pages_to_wrap) {
                        B = pages_to_wrap;
                    }

                    const uint32_t fifo_snapshot = iface.fifo_wr_ptr;
                    const uint32_t bytes_per_recv = B * t_page_bytes_per_recv;
                    const uint32_t chunks_per_visit = (bytes_per_recv + max_chunk_bytes - 1) / max_chunk_bytes;
                    const uint32_t total_chunks_round = num_receivers * chunks_per_visit;

                    // ---- Depth-2 software pipeline over 3 stage slots ----
                    // Per chunk gc we: wait gc's DMA, enqueue gc's NoC writes (posted), THEN
                    // issue chunk gc+2's DMA. Enqueuing the writes before the DMA lets them
                    // drain on the NoC while the core spins on the DMA read_ready gate and the
                    // engine reads gc+2 — overlapping the ~190 cyc write drain with the ~320 cyc
                    // read instead of serializing them.
                    //
                    // Slot safety: chunk gc+2's DMA targets slot (gc+2)%3 == (gc-1)%3, last
                    // NoC-written by chunk gc-1. We drain it with a flush placed BEFORE
                    // writes(gc) — so the flush waits only on chunks <= gc-1 (already mostly
                    // drained during this iter's DMA wait) and never on chunk gc's just-issued
                    // writes, preserving the overlap. The two in-flight DMAs (gc+1, gc+2) and
                    // the write source (gc) occupy all three distinct slots.

                    // Compute chunk c's receiver-contiguous DRAM source + size and issue its DMA
                    // into stage slot c%3. cr = receiver index within the round, civ = chunk index
                    // within that receiver's visit. Shared by the prologue and the in-loop gc+2
                    // issue so the address arithmetic lives in one place.
                    auto issue_chunk_dma = [&](uint32_t c) {
                        const uint32_t cr = c / chunks_per_visit;
                        const uint32_t civ = c - cr * chunks_per_visit;
                        const uint32_t boff = civ * max_chunk_bytes;
                        const uint32_t rem = bytes_per_recv - boff;
                        const uint32_t cb = rem < max_chunk_bytes ? rem : max_chunk_bytes;
                        const uint32_t src = tensor_base + (recv_index_base + cr) * t_recv_stride +
                                             pages_sent_global * t_page_bytes_per_recv + boff;
                        experimental::dma_async_read(/*stream=*/0, src, slot_addrs[c % 3u], cb);
                    };

                    // Prologue: prime the pipe with the first up-to-2 chunk DMAs.
                    for (uint32_t c = 0; c < 2u && c < total_chunks_round; ++c) {
                        PROF_DECL_TS(t_p0);
                        PROF_DECL_TS(t_p1);
                        PROF_TICK(t_p0);
                        issue_chunk_dma(c);
                        PROF_TICK(t_p1);
                        PROF_ACC(prof_dma_issue, t_p1, t_p0);
                    }

                    uint32_t cur_r = uint32_t(-1);
                    uint32_t remote_noc_xy = 0;
                    for (uint32_t gc = 0; gc < total_chunks_round; ++gc) {
                        PROF_INC(prof_chunks);
                        const uint32_t r = gc / chunks_per_visit;
                        const uint32_t chunk_in_visit = gc - r * chunks_per_visit;
                        const uint32_t byte_offset = chunk_in_visit * max_chunk_bytes;
                        const uint32_t remaining_visit = bytes_per_recv - byte_offset;
                        const uint32_t chunk_bytes =
                            remaining_visit < max_chunk_bytes ? remaining_visit : max_chunk_bytes;
                        const bool keep_one = (gc + 1u) < total_chunks_round;    // gc+1 stays in flight
                        const bool will_issue = (gc + 2u) < total_chunks_round;  // a gc+2 chunk exists

                        // ---- Wait for chunk gc's DMA (keep chunk gc+1 in flight). ----
                        PROF_DECL_TS(t_dw0);
                        PROF_DECL_TS(t_dw1);
                        PROF_TICK(t_dw0);
                        experimental::dma_async_read_wait_n(/*stream=*/0, keep_one ? 1u : 0u);
                        PROF_TICK(t_dw1);
                        PROF_ACC(prof_dma_wait, t_dw1, t_dw0);

                        // ---- Pre-write slot-protect flush: drain chunks <= gc-1 so chunk gc+2's
                        //      DMA can safely overwrite slot (gc+2)%3 == (gc-1)%3. Cheap (those
                        //      writes drained during the wait above); does not touch chunk gc's
                        //      writes (not yet enqueued). Not needed for gc<1 (virgin slot). ----
                        if (will_issue && gc >= 1u) {
                            PROF_DECL_TS(t_df0);
                            PROF_DECL_TS(t_df1);
                            PROF_TICK(t_df0);
                            noc_async_posted_writes_flushed();
                            PROF_TICK(t_df1);
                            PROF_ACC(prof_defer_flush, t_df1, t_df0);
                        }

                        // ---- set_state on the first chunk of a new receiver ----
                        if (r != cur_r) {
                            cur_r = r;
                            volatile tt_l1_ptr uint32_t* xy_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr) + r * 2;
                            remote_noc_xy = uint32_t(NOC_XY_ENCODING(
                                DYNAMIC_NOC_X(noc_index, xy_ptr[0]), DYNAMIC_NOC_Y(noc_index, xy_ptr[1])));
                            const uint64_t set_state_dest = get_noc_addr_helper(remote_noc_xy, fifo_snapshot);
                            PROF_DECL_TS(t_ss0);
                            PROF_DECL_TS(t_ss1);
                            PROF_TICK(t_ss0);
                            noc_async_write_one_packet_set_state</*posted=*/true>(
                                set_state_dest, t_coal_page_size, noc_index);
                            PROF_TICK(t_ss1);
                            PROF_ACC(prof_set_state, t_ss1, t_ss0);
                        }

                        // ---- Enqueue chunk gc's NoC writes (posted) from slot gc%3. They drain
                        //      on the NoC while we issue+read chunk gc+2 below. ----
                        const uint32_t num_packets = chunk_bytes / t_coal_page_size;
                        uint32_t src_addr = slot_addrs[gc % 3u];
                        uint32_t dest_addr = fifo_snapshot + byte_offset;
                        PROF_DECL_TS(t_w0);
                        PROF_DECL_TS(t_w1);
                        PROF_TICK(t_w0);
                        for (uint32_t p = 0; p < num_packets; ++p) {
                            noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_addr, noc_index);
                            src_addr += t_coal_page_size;
                            dest_addr += t_coal_page_size;
                        }
                        PROF_TICK(t_w1);
                        PROF_ACC(prof_writes, t_w1, t_w0);

                        // ---- Issue chunk gc+2's DMA into slot (gc+2)%3 (read overlaps gc's drain). ----
                        if (will_issue) {
                            PROF_DECL_TS(t_di0);
                            PROF_DECL_TS(t_di1);
                            PROF_TICK(t_di0);
                            issue_chunk_dma(gc + 2u);
                            PROF_TICK(t_di1);
                            PROF_ACC(prof_dma_issue, t_di1, t_di0);
                        }
                    }

                    // ---- Final flush: drain any remaining writes before finalize bumps semaphore.
                    // (Finalize's per-receiver semaphore inc rides the same NoC VC, so a
                    // drained NIU also implies data writes are committed in-order on the wire.)
                    PROF_DECL_TS(t_f0);
                    PROF_DECL_TS(t_f1);
                    PROF_TICK(t_f0);
                    noc_async_posted_writes_flushed();
                    PROF_TICK(t_f1);
                    PROF_ACC(prof_flush, t_f1, t_f0);

                    // ---- Finalize round: bump pages_sent + semaphores by B blocks ----
                    PROF_DECL_TS(t_fn0);
                    PROF_DECL_TS(t_fn1);
                    PROF_TICK(t_fn0);
                    prefetcher_finalize_block</*skip_ptr_update=*/true>(
                        iface, B * t_page_bytes_per_recv, num_receivers, noc_index);
                    PROF_TICK(t_fn1);
                    PROF_ACC(prof_finalize, t_fn1, t_fn0);
                    pages_sent_global += B;
                }
            }
        }

        // ---- Profile dump (only emitted when watcher ring-buffer is enabled) ----
        // Tags: 0xA1 rounds, 0xA2 chunks, 0xA3 poll, 0xA4 dma_issue, 0xA5 dma_wait,
        //       0xA6 set_state, 0xA7 writes, 0xA8 flush, 0xA9 finalize.
        // All cycle totals are 24-bit (16M wall-clock cycles ≈ 16 ms at 1 GHz).
        PROF_DUMP(0xA1u, prof_rounds);
        PROF_DUMP(0xA2u, prof_chunks);
        PROF_DUMP(0xA3u, prof_poll);
        PROF_DUMP(0xA4u, prof_dma_issue);
        PROF_DUMP(0xA5u, prof_dma_wait);
        PROF_DUMP(0xA6u, prof_set_state);
        PROF_DUMP(0xA7u, prof_writes);
        PROF_DUMP(0xA8u, prof_flush);
        PROF_DUMP(0xA9u, prof_finalize);
        PROF_DUMP(0xAAu, prof_defer_flush);
        PROF_DUMP(0xB0u, stage_ring_size);
        PROF_DUMP(0xB1u, stage_third);

        // Persist mutable state (fifo_wr_ptr) so the next request to this GCB
        // resumes at the right ring offset.
        store_sender_state(state, iface);

        socket_pop_pages(socket, 1);
        socket_notify_sender(socket);
    }

    // Restore NoC2AXI mode. No NoC drain is needed here: the stream is already
    // flushed end-to-end by the remote_cb_sender_barrier at the stop sentinel, which
    // spins until every receiver's pages_acked == pages_sent -- and the receiver can
    // only ack pages whose posted pages_sent atomics and data writes have already
    // landed. The pages_sent increments are posted (noc_semaphore_inc<skip_ptr_update=
    // true>), so a noc_async_atomic_barrier would do nothing anyway: it waits on
    // non-posted atomics, of which this kernel issues none. (Cross-request fifo_wr_ptr
    // persistence is handled per-request by store_sender_state, not by a config
    // writeback here.)
    experimental::drisc_set_noc2axi_mode();
}
