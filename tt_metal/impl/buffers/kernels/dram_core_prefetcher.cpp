// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Queueable DRISC prefetcher kernel — successor to dram_core_prefetcher.cpp.
// Sits in a request loop on a per-(device, sender-core) H2D socket; each request
// payload identifies the target GlobalCircularBuffer (by its DRISC L1
// sender-state-block base, written by the GCB ctor) and carries the per-tensor
// geometry. The kernel loads the sender state block's RemoteSenderCBInterface
// region into cb_interface[], runs the chunk-loop logic, writes the mutable
// fifo_wr_ptr back to L1 so the next request to the same GCB resumes from the right
// ring offset, and acks the socket page.
//
// Request page wire format (one socket page): a DramCorePrefetcherRequestHeader, then
// a forward-growing table of per-tensor DramCorePrefetcherEntry (address + layout index)
// and a backward-growing (from the end of the payload) deduplicated table of
// DramCorePrefetcherTensorLayout. The kernel walks the entries in order, resolving each
// entry's geometry from the referenced layout. See
// tt_metal/impl/buffers/dram_core_prefetcher_request.hpp. A header with
// num_entries == 0 is the stop sentinel.
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
#include "tt_metal/impl/buffers/dram_core_prefetcher_request.hpp"

using tt::tt_metal::DramCorePrefetcherEntry;
using tt::tt_metal::DramCorePrefetcherRequestHeader;
using tt::tt_metal::DramCorePrefetcherTensorLayout;
using tt::tt_metal::DramSenderStateBlock;
using tt::tt_metal::kRequestPageBytes;

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
    constexpr uint32_t ring_half = stage_ring_size / 2;
    constexpr uint32_t stage_slot_a = stage_ring_base;
    constexpr uint32_t stage_slot_b = stage_ring_base + ring_half;

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

    // ---- Request loop ----
    while (true) {
        socket_wait_for_pages(socket, 1);

        volatile tt_l1_ptr DramCorePrefetcherRequestHeader* req =
            reinterpret_cast<volatile tt_l1_ptr DramCorePrefetcherRequestHeader*>(socket.read_ptr);
        const uint32_t req_num_entries = req->num_entries;
        if (req_num_entries == 0) {
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
        const uint32_t gcb_state_addr = req->gcb_state_addr;
        volatile tt_l1_ptr DramSenderStateBlock* state =
            reinterpret_cast<volatile tt_l1_ptr DramSenderStateBlock*>(gcb_state_addr);

        load_sender_state(state, iface);
        has_loaded_sender_state = true;
        // num_receivers lives inside the GCB's state block (set by the GCB ctor).
        // Reading it per request lets a single prefetcher serve GCBs with different
        // receiver counts.
        const uint32_t num_receivers = state->num_receivers;

        // Entries follow the header (grow forward); the deduplicated layout table grows
        // backward from the end of the payload, so layout i lives at read_ptr +
        // kRequestPageBytes - (i+1)*sizeof(layout). See dram_core_prefetcher_request.hpp.
        volatile tt_l1_ptr DramCorePrefetcherEntry* entries =
            reinterpret_cast<volatile tt_l1_ptr DramCorePrefetcherEntry*>(
                socket.read_ptr + sizeof(DramCorePrefetcherRequestHeader));
        const uint32_t layout_table_end = socket.read_ptr + kRequestPageBytes;

        for (uint32_t e = 0; e < req_num_entries; ++e) {
            const uint32_t tensor_base = entries[e].bank_local_base;
            volatile tt_l1_ptr DramCorePrefetcherTensorLayout* g =
                reinterpret_cast<volatile tt_l1_ptr DramCorePrefetcherTensorLayout*>(
                    layout_table_end - (entries[e].layout_index + 1) * sizeof(DramCorePrefetcherTensorLayout));
            const uint32_t t_num_sub = g->num_sub;
            const uint32_t t_M = g->M;
            const uint32_t t_rows_per_sub = g->rows_per_sub;
            const uint32_t t_coal_page_size = g->coalesced_page_size;
            const uint32_t t_coal_num_pages = g->coalesced_num_pages;
            const uint32_t t_chunk_bytes = g->sub_chunk_bytes;
            const uint32_t t_sub_stride = g->sub_stride_bytes;
            const uint32_t t_block_stride = g->block_stride_bytes;
            const uint32_t t_page_bytes_per_recv = g->page_bytes_per_recv;
            const uint32_t t_block_count = g->block_count;
            const uint32_t t_recv_per_chunk = num_receivers / t_M;
            const uint32_t t_sub_band_per_block = t_num_sub * t_M;

            // Set the sender fifo page size to one full per-receiver page so
            // remote_cb_reserve_back reserves exactly one page per receiver.
            experimental::resize_remote_sender_cb_interface</*update_remote_over_noc=*/false>(
                remote_cb_id, t_page_bytes_per_recv, noc_index);

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
            // stage_slot ping-pongs between stage_slot_a and stage_slot_b; toggle via
            // `(a + b) - slot`.
            constexpr uint32_t stage_slot_sum = stage_slot_a + stage_slot_b;
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
        }

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
