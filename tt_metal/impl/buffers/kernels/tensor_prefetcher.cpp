// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Queueable DRISC prefetcher kernel — successor to tensor_prefetcher.cpp.
// Sits in a request loop on a per-(device, sender-core) H2D socket; each request
// payload names the PrefetcherPipe it delivers into by the DRISC L1 address of this
// sender's config page for that pipe, and carries the per-tensor geometry. For each
// tensor the kernel loads the pipe's credit counters and write cursor from that page,
// runs the chunk loop, and writes the cursor back into the page, where each receiver
// keeps its own copy, so the next request (or the next program) resumes where this
// one stopped.
//
// Request page wire format (one socket page): a TensorPrefetcherRequestHeader
// (one-byte command id + per-command union). The STOP command (all-zero page) exits
// the request loop; WAIT_CQ blocks on a per-CQ signal slot; DRAIN waits out the acks of
// the pipes it lists; PREFETCH is followed by a forward-growing table of per-tensor
// TensorPrefetcherEntry (address + layout offset) and a backward-growing (from the end of
// the payload) deduplicated table of TensorPrefetcherTensorLayout. The kernel walks the
// entries in order, resolving each entry's geometry from the referenced layout. See
// tt_metal/impl/buffers/tensor_prefetcher_request.hpp.
//
// PrefetcherPipe sender config page layout: see
// tt_metal/hw/inc/hostdev/remote_dfb_config_layout.h.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "experimental/gddr_dma.h"
#include "experimental/gddr_mc.h"
#include "hostdev/remote_dfb_config_layout.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "internal/prefetcher_pipe_dram_sender.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_request.hpp"

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

namespace {

// The pipe a request delivers into, as the loops below drive it: its config page, loaded per tensor,
// plus the write cursor and the end of the whole-block region for that tensor's block size. Every
// receiver of a pipe is credited the same bytes each round, so one cursor stands for all of them.
struct PipeSender {
    experimental::PipeSenderCtx ctx;
    // Ring base plus the bytes that hold whole blocks: no block starts at or past it, and the
    // trailing gap after it is credited by whichever block reaches it.
    uint32_t fifo_limit;
    // Receiver-L1 address the next block lands at.
    uint32_t fifo_wr_ptr;
};

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

// Credit one round of `bytes_per_recv` bytes to every receiver and advance the write cursor past it.
// A round that reaches the usable limit also credits the trailing gap and wraps the cursor to the
// ring base. Credits count L1_ALIGNMENT-byte units, one SENT slot per receiver at both ends.
template <bool skip_ptr_update>
FORCE_INLINE void prefetcher_finalize_block(PipeSender& snd, uint32_t bytes_per_recv, uint8_t noc) {
    const experimental::PipeSenderCtx& ctx = snd.ctx;
    uint32_t len_bytes = bytes_per_recv;
    uint32_t next_wr_ptr = snd.fifo_wr_ptr + bytes_per_recv;
    if (next_wr_ptr >= snd.fifo_limit) {
        len_bytes += ctx.fifo_start_addr + ctx.ring_bytes - snd.fifo_limit;
        next_wr_ptr = ctx.fifo_start_addr + (next_wr_ptr - snd.fifo_limit);
    }
    const uint32_t credit_units = len_bytes / L1_ALIGNMENT;

    volatile tt_l1_ptr uint32_t* local_sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.local_sent_base);
    uint32_t remote_sent = ctx.remote_sent_base;
    volatile tt_l1_ptr uint32_t* recv_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.receiver_noc_xy_ptr);
    for (uint32_t i = 0; i < ctx.num_receivers; ++i) {
        const uint32_t remote_noc_xy =
            uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, recv_xy_ptr[0]), DYNAMIC_NOC_Y(noc, recv_xy_ptr[1])));
        *local_sent += credit_units;
        noc_semaphore_inc<skip_ptr_update>(get_noc_addr_helper(remote_noc_xy, remote_sent), credit_units, noc);
        local_sent += L1_ALIGNMENT / sizeof(uint32_t);
        remote_sent += L1_ALIGNMENT;
        recv_xy_ptr += 2;
    }
    snd.fifo_wr_ptr = next_wr_ptr;
}

// Load the pipe whose sender config page is at `config_page_addr`. The write cursor comes from that
// page, where every receiver has a durable copy beside its credit counters; delivery credits them
// all alike, so receiver 0's is representative. That holds under the streaming rotation too -- it
// varies which DRAM block feeds a receiver, not how much each one is credited.
FORCE_INLINE PipeSender load_pipe_sender(uint32_t config_page_addr) {
    PipeSender snd;
    experimental::pipe_load_sender_ctx(snd.ctx, config_page_addr);
    // pipe_usable_bytes owns the trailing-gap rule; do not re-derive it here.
    snd.fifo_limit = snd.ctx.fifo_start_addr + experimental::pipe_usable_bytes(snd.ctx);
    snd.fifo_wr_ptr = snd.ctx.fifo_start_addr + experimental::pipe_sender_wr_offset(snd.ctx, 0);
    return snd;
}

// Put a tensor's finishing write cursor back into every receiver's slot in the pipe's config page,
// which is where the next tensor's snap and the next program's load read it.
FORCE_INLINE void store_pipe_sender(const PipeSender& snd) {
    experimental::pipe_store_wr_offset(snd.ctx, snd.fifo_wr_ptr - snd.ctx.fifo_start_addr);
}

}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    // The receiver count is not among them: each request's pipe carries its own in its config
    // page, so one prefetcher can serve pipes with different receiver counts.
    constexpr uint32_t stage_ring_base = get_compile_time_arg_val(0);
    constexpr uint32_t stage_ring_size = get_compile_time_arg_val(1);
    constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
    // Base of this core's per-CQ signal slots (kNumCqSignalSlots uint32 counters).
    // WaitForCqOnTensorPrefetcher writes an incrementing value here from the
    // dispatcher; a WAIT_CQ request blocks until the requested slot reaches it.
    // Slots are `cq_signal_slot_stride` bytes apart, not packed: each is the target of
    // its own dispatcher write, which only lands on an L1-aligned address.
    constexpr uint32_t cq_signal_l1_base = get_compile_time_arg_val(3);
    constexpr uint32_t cq_signal_slot_stride = get_compile_time_arg_val(4);
    constexpr bool controls_ordinary_mpfe = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t own_active_mpfe_weight = get_compile_time_arg_val(6);
    constexpr uint32_t ordinary_mpfe_weight = get_compile_time_arg_val(7);
    constexpr bool dynamic_mpfe_weighting = get_compile_time_arg_val(8) != 0;
    constexpr bool mpfe_enabled = get_compile_time_arg_val(9) != 0;
    // kTensorPrefetcherSelectorScratchBytes of DRISC L1, DRAM-aligned, that a selector page is read
    // into before the groups it selects are streamed (see tensor_prefetcher_request.hpp).
    constexpr uint32_t selector_scratch_l1_addr = get_compile_time_arg_val(10);
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
    const uint32_t own_mpfe_port = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t ordinary_mpfe_port = get_arg_val<uint32_t>(rt_idx++);

    // ---- Init ----
    SocketReceiverInterface socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(socket, socket_page_size);

    // Each sender owns only its private MPFE slot. The primary sender also owns
    // the ordinary-operation slot, which remains static for the whole lifetime.
    if constexpr (mpfe_enabled) {
        gddr_mc_write_mpfe_weight(
            own_mpfe_port, dynamic_mpfe_weighting ? ordinary_mpfe_weight : own_active_mpfe_weight);
        if constexpr (controls_ordinary_mpfe) {
            gddr_mc_write_mpfe_weight(ordinary_mpfe_port, ordinary_mpfe_weight);
        }
    }

    // Zero the per-CQ signal slots before parking on the socket. Safe to do here
    // (rather than from the host) because no WaitForCqOnTensorPrefetcher signal
    // can be enqueued until StartTensorPrefetcher returns to the single-threaded
    // host caller, long after this init runs.
    auto cq_signal_slot = [](uint32_t index) -> volatile tt_l1_ptr uint32_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cq_signal_l1_base + index * cq_signal_slot_stride);
    };
    for (uint32_t i = 0; i < kNumCqSignalSlots; ++i) {
        *cq_signal_slot(i) = 0;
    }

    // ---- Request loop ----
    while (true) {
        socket_wait_for_pages(socket, 1);

        volatile tt_l1_ptr TensorPrefetcherRequestHeader* req =
            reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherRequestHeader*>(socket.read_ptr);
        const uint8_t cmd_id = req->base.cmd_id;
        if (cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_STOP) {
            // Stop sentinel. The host sends it only after DRAIN pages covering every target whose
            // receivers could still be acking into this DRISC's L1, so nothing is left in flight.
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            break;
        }
        if (cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_DRAIN) {
            // Receivers ack into counters in this DRISC's L1, which the next program is free to
            // reuse once the kernel exits, so wait for the acks of every listed pipe. Each is named
            // by its sender config page, which holds both its credit blocks.
            volatile tt_l1_ptr uint32_t* config_pages =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.read_ptr + sizeof(TensorPrefetcherRequestHeader));
            const uint32_t num_drain_targets = req->drain.num_targets;
            for (uint32_t t = 0; t < num_drain_targets; ++t) {
                experimental::PipeSenderCtx ctx;
                experimental::pipe_load_sender_ctx(ctx, config_pages[t]);
                experimental::pipe_sender_barrier(ctx);
            }
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            continue;
        }
        if (cmd_id == tt::tt_metal::DRAM_PREFETCHER_CMD_WAIT_CQ) {
            // Block until the dispatcher has bumped this CQ's signal slot to the
            // requested value. Wrap-safe: compare the unsigned difference as signed.
            volatile tt_l1_ptr uint32_t* slot = cq_signal_slot(req->wait_cq.cq_index);
            const uint32_t target = req->wait_cq.cq_wait_value;
            while ((int32_t)(*slot - target) < 0) {
                invalidate_l1_cache();
            }
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            continue;
        }
        // DRAM_PREFETCHER_CMD_PREFETCH
        if constexpr (mpfe_enabled && dynamic_mpfe_weighting) {
            gddr_mc_write_mpfe_weight(own_mpfe_port, own_active_mpfe_weight);
        }

        const uint32_t req_num_entries = req->prefetch.num_entries;
        // This sender's config page for the pipe the request delivers into.
        const uint32_t config_page_addr = req->prefetch.target_config_page_addr;

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

        // Entries follow the header (grow forward); the deduplicated layout slots grow backward
        // from the end of the payload, and each entry names its own slot by byte offset from the
        // page start -- so how the host packed the slots stays host business. See
        // tensor_prefetcher_request.hpp.
        volatile tt_l1_ptr TensorPrefetcherEntry* entries = reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherEntry*>(
            socket.read_ptr + sizeof(TensorPrefetcherRequestHeader));

        for (uint32_t e = 0; e < req_num_entries; ++e) {
            volatile tt_l1_ptr TensorPrefetcherTensorLayout* g =
                reinterpret_cast<volatile tt_l1_ptr TensorPrefetcherTensorLayout*>(
                    socket.read_ptr + entries[e].layout_offset);
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
            // Bank-local slab index of this sender's first receiver. When two DRISC cores split a
            // bank's receivers, the second core's local receiver r maps to bank-local slab
            // (recv_index_base + r). 0 for a single sender. Receiver-contiguous only. Per-sender
            // rather than per-tensor, but it is patched into every slot of this sender's page.
            const uint32_t recv_index_base = g->recv_index_base;
            // Streaming (receiver-contiguous only) is a per-tensor layout attribute: deliver
            // this tensor's blocks in ring-rotated order so the matmul can consume them FIFO.
            const bool streaming = g->streaming != 0;
            // Per-receiver streaming rotation table, appended right after this tensor's geometry
            // in the page (host-sliced for this sender). rotation_local[r] is the lead block for
            // local receiver r, so the kernel sources block (rotation_local[r] + p) mod block_count.
            // Only read when streaming.
            volatile tt_l1_ptr uint32_t* rotation_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(g) + sizeof(TensorPrefetcherTensorLayout));

            // A grouped tensor (an expert stack) streams each group's slabs as if it were its own
            // tensor, in ascending group order. A selector narrows that to the groups whose mask word
            // is non-zero; the mask is read here, per request rather than baked into the page, so a
            // replayed (traced) request picks up whatever the device wrote there since.
            const uint32_t t_group_stride = g->group_stride_bytes;
            const uint32_t num_groups = entries[e].num_groups;
            const bool selected = entries[e].selector_mode == tt::tt_metal::TENSOR_PREFETCHER_SELECTOR_NONZERO_MASK;
            volatile tt_l1_ptr uint16_t* selector =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(selector_scratch_l1_addr);
            if (selected) {
                const uint32_t selector_xy = entries[e].selector_noc_xy;
                const uint32_t selector_noc_xy = uint32_t(NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(noc_index, selector_xy & 0xFFFFu), DYNAMIC_NOC_Y(noc_index, selector_xy >> 16)));
                const uint32_t selector_bytes =
                    (num_groups * sizeof(uint16_t) + L1_ALIGNMENT - 1) & ~(L1_ALIGNMENT - 1);
                noc_async_read(
                    get_noc_addr_helper(selector_noc_xy, entries[e].selector_addr),
                    selector_scratch_l1_addr,
                    selector_bytes,
                    noc_index);
                noc_async_read_barrier(noc_index);
            }
            for (uint32_t group = 0; group < num_groups; ++group) {
                if (selected && selector[group] == 0) {
                    continue;
                }
                const uint32_t tensor_base = entries[e].bank_local_base + group * t_group_stride;

                // Apply this tensor's block size to the pipe before loading it. When it differs from the
                // size the pipe last applied, the PrefetcherPipe constructor snaps every receiver's stored
                // write cursor onto the new entry grid and publishes the skipped bytes as pad credits. The
                // consumer's PrefetcherPipe constructor runs the matching snap when its Attach entry size
                // differs from the one last applied and blocks on exactly these credits, so a tensor whose
                // per-receiver block size differs from the previous one still starts every entry on an
                // entry boundary. A tensor with the size already applied changes nothing.
                {
                    experimental::PrefetcherPipe pipe(
                        experimental::DramSenderConfigPage{config_page_addr}, t_page_bytes_per_recv);
                }
                // Load the pipe from its config page each tensor. The write cursors come out of that page,
                // which is what makes them resume correctly across requests and across programs, and the
                // snap above left this tensor's entry size there too.
                PipeSender snd = load_pipe_sender(config_page_addr);

                // stage_slot ping-pongs between stage_slot_a and stage_slot_b; toggle via
                // `(a + b) - slot`.
                constexpr uint32_t stage_slot_sum = stage_slot_a + stage_slot_b;

                // Converts the credit units of free space the poll reports into whole blocks of
                // t_page_bytes_per_recv, the unit both loops push in.
                const uint32_t credit_units_per_block = t_page_bytes_per_recv / L1_ALIGNMENT;

                // Branch on layout. The branch is per-tensor (100% predictable across the
                // inner chunk loop), so adding the second branch costs nothing in the
                // hot path.
                if (t_layout_mode == 0) {
                    // ---- K-row-major main loop ----
                    const uint32_t t_recv_per_chunk = snd.ctx.num_receivers / t_M;
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
                            // Wait for room for this whole block at every receiver. A block that reaches
                            // the usable limit also credits the trailing gap, but that needs no extra
                            // room: the gap never holds data, so it is free whenever the block is.
                            while (experimental::pipe_sender_min_free_units(snd.ctx) < credit_units_per_block) {
                            }
                            fifo_snapshot = snd.fifo_wr_ptr;
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
                            const uint32_t next_src = tensor_base + next_blk * t_block_stride + next_sb * t_sub_stride +
                                                      next_ch * t_chunk_bytes;
                            experimental::dma_async_read(/*stream=*/0, next_src, next_slot, t_chunk_bytes);
                        }
                        const uint32_t outstanding_after_wait = has_next ? 1u : 0u;
                        experimental::dma_async_read_wait_n(/*stream=*/0, outstanding_after_wait);
                        volatile tt_l1_ptr uint32_t* chunk_recv_xy =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(snd.ctx.receiver_noc_xy_ptr) +
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
                            prefetcher_finalize_block</*skip_ptr_update=*/true>(snd, t_page_bytes_per_recv, noc_index);
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
                    // semaphore by B for every receiver and advances snd.fifo_wr_ptr by
                    // B * t_page_bytes_per_recv. The pages_to_wrap clamp prevents B from
                    // crossing snd.fifo_limit mid-round.
                    constexpr uint32_t kStageSlotBytes = stage_third;
                    // Per-stage-slot capacity expressed in whole sub-bands; t_chunk_bytes
                    // (= sub_chunk_bytes) is the natural alignment for both DMA and NoC
                    // writes — it's a multiple of t_coal_page_size by construction, so
                    // packet counts come out exact (no truncation loss).
                    const uint32_t subs_per_stage_slot = kStageSlotBytes / t_chunk_bytes;
                    const uint32_t max_chunk_bytes = subs_per_stage_slot * t_chunk_bytes;

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
                            min_free_blocks =
                                experimental::pipe_sender_min_free_units(snd.ctx) / credit_units_per_block;
                        } while (min_free_blocks == 0);
                        PROF_TICK(t_poll_end);
                        PROF_ACC(prof_poll, t_poll_end, t_poll_start);

                        const uint32_t bytes_to_wrap = snd.fifo_limit - snd.fifo_wr_ptr;
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

                        const uint32_t fifo_snapshot = snd.fifo_wr_ptr;
                        const uint32_t bytes_per_recv = B * t_page_bytes_per_recv;

                        // blk (first source block) and wrap_boff (byte offset within the visit where
                        // the source wraps N->0, or bytes_per_recv if it doesn't) for receiver r.
                        // Streaming leaves B un-clamped; with B <= remaining <= block_count, a visit
                        // wraps at most once. Batched never wraps, so its plan stays uniform.
                        auto recv_wrap = [&](uint32_t r, uint32_t& blk_out, uint32_t& wrap_boff_out) {
                            uint32_t blk = pages_sent_global;
                            if (streaming) {
                                blk += rotation_local[r];
                                if (blk >= t_block_count) {
                                    blk -= t_block_count;
                                }
                            }
                            blk_out = blk;
                            wrap_boff_out = (blk + B > t_block_count) ? (t_block_count - blk) * t_page_bytes_per_recv
                                                                      : bytes_per_recv;
                        };

                        // Uniform max-chunk geometry: every receiver contributes the same chunk count,
                        // and the single chunk per wrapping receiver that straddles the source wrap
                        // issues two contiguous DMA reads (tail then head) into one stage slot. Batched
                        // never wraps. (This is the two-DMA-read streaming split; the older legacy-clamp
                        // and ragged-tail/head modes have been removed.)
                        const uint32_t uniform_chunks_per_visit =
                            (bytes_per_recv + max_chunk_bytes - 1) / max_chunk_bytes;
                        const uint32_t total_chunks_round = snd.ctx.num_receivers * uniform_chunks_per_visit;

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

                        // Per-chunk plan stored in a 3-deep ring (one entry per stage slot): the
                        // generator runs <=2 chunks ahead of the writer, and slot (gc+2)%3 ==
                        // (gc-1)%3 is only reused after chunk gc-1's writes drain, so 3 entries hold
                        // every chunk the writer still needs (dst offset, size, receiver).
                        uint32_t desc_dst_off[3] = {0, 0, 0};
                        uint32_t desc_cb[3] = {0, 0, 0};
                        uint32_t desc_r[3] = {0, 0, 0};
                        uint32_t desc_reads[3] = {1, 1, 1};  // DMA reads issued for this slot's chunk (mode 2: 1 or 2)

                        // Generator state: walk receivers in order; within a receiver, walk its visit
                        // in max_chunk_bytes steps, breaking at the source wrap so each chunk is one
                        // contiguous DMA read (the depth-2 pipeline / dma_async_read_wait_n count
                        // reads, not bytes -- a chunk must never straddle the wrap).
                        uint32_t gen_r = 0;
                        uint32_t gen_boff = 0;
                        uint32_t gen_blk = 0;
                        uint32_t gen_wrap_boff = 0;
                        recv_wrap(0, gen_blk, gen_wrap_boff);

                        // Emit chunk c: size + source of the next contiguous segment, record the
                        // dst/size/receiver for the writer, issue the single DMA into slot c%3, and
                        // advance the generator. Called in order (prologue c=0,1; loop c=gc+2).
                        auto issue_chunk_dma = [&](uint32_t c) {
                            uint32_t cb = bytes_per_recv - gen_boff;
                            if (cb > max_chunk_bytes) {
                                cb = max_chunk_bytes;
                            }
                            const uint32_t idx = c % 3u;
                            const uint32_t slab = tensor_base + (recv_index_base + gen_r) * t_recv_stride;
                            const bool straddles = (gen_boff < gen_wrap_boff && gen_boff + cb > gen_wrap_boff);
                            // Uniform max-chunk: a chunk that straddles the source wrap is read as two
                            // contiguous DMAs into the same slot (tail then head); the slot stays one
                            // contiguous run for the writer. Otherwise a single read.
                            if (straddles) {
                                const uint32_t cb1 = gen_wrap_boff - gen_boff;  // tail [blk..N)
                                const uint32_t src1 = slab + gen_blk * t_page_bytes_per_recv + gen_boff;
                                const uint32_t src2 = slab;  // head resumes at physical block 0
                                experimental::dma_async_read(/*stream=*/0, src1, slot_addrs[idx], cb1);
                                experimental::dma_async_read(/*stream=*/0, src2, slot_addrs[idx] + cb1, cb - cb1);
                                desc_reads[idx] = 2;
                            } else {
                                const uint32_t src = (gen_boff < gen_wrap_boff)
                                                         ? slab + gen_blk * t_page_bytes_per_recv + gen_boff
                                                         : slab + (gen_boff - gen_wrap_boff);
                                experimental::dma_async_read(/*stream=*/0, src, slot_addrs[idx], cb);
                                desc_reads[idx] = 1;
                            }
                            desc_dst_off[idx] = gen_boff;
                            desc_cb[idx] = cb;
                            desc_r[idx] = gen_r;
                            gen_boff += cb;
                            if (gen_boff >= bytes_per_recv) {
                                gen_r += 1;
                                gen_boff = 0;
                                if (gen_r < snd.ctx.num_receivers) {
                                    recv_wrap(gen_r, gen_blk, gen_wrap_boff);
                                }
                            }
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
                            const uint32_t idx = gc % 3u;
                            const uint32_t r = desc_r[idx];
                            const uint32_t byte_offset = desc_dst_off[idx];
                            const uint32_t chunk_bytes = desc_cb[idx];
                            const bool keep_one = (gc + 1u) < total_chunks_round;    // gc+1 stays in flight
                            const bool will_issue = (gc + 2u) < total_chunks_round;  // a gc+2 chunk exists

                            // ---- Wait for chunk gc's DMA (keep chunk gc+1 in flight). ----
                            PROF_DECL_TS(t_dw0);
                            PROF_DECL_TS(t_dw1);
                            PROF_TICK(t_dw0);
                            // Keep chunk gc+1 in flight: wait until outstanding reads drop to that chunk's
                            // read count (1, or 2 for a chunk that straddles the source wrap), or 0 if gc is last.
                            const uint32_t keep_reads = keep_one ? desc_reads[(gc + 1u) % 3u] : 0u;
                            experimental::dma_async_read_wait_n(/*stream=*/0, keep_reads);
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
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(snd.ctx.receiver_noc_xy_ptr) + r * 2;
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
                        prefetcher_finalize_block</*skip_ptr_update=*/true>(snd, B * t_page_bytes_per_recv, noc_index);
                        PROF_TICK(t_fn1);
                        PROF_ACC(prof_finalize, t_fn1, t_fn0);
                        pages_sent_global += B;
                    }
                }
                // Per tensor rather than per request: the next entry in this same request snaps its
                // cursors onto its own entry grid, and that snap reads them from the page.
                store_pipe_sender(snd);
            }  // group
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

        if constexpr (mpfe_enabled && dynamic_mpfe_weighting) {
            gddr_mc_write_mpfe_weight(own_mpfe_port, ordinary_mpfe_weight);
        }

        socket_pop_pages(socket, 1);
        socket_notify_sender(socket);
    }

    if constexpr (mpfe_enabled) {
        gddr_mc_write_mpfe_weight(own_mpfe_port, GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_DEFAULT);
        if constexpr (controls_ordinary_mpfe) {
            gddr_mc_write_mpfe_weight(ordinary_mpfe_port, GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_DEFAULT);
        }
    }
}
