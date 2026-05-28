// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Queueable DRISC prefetcher kernel — successor to dram_core_prefetcher.cpp.
// Sits in a request loop on a per-(device, sender-core) H2D socket; each
// request payload identifies the target GlobalCircularBuffer (by its DRISC L1
// sender-state-block base, written by the GCB ctor) and carries the per-tensor
// geometry. The kernel memcpys the sender state block into cb_interface[],
// runs the existing chunk-loop logic, writes the mutable fifo_wr_ptr back to
// L1 so the next request to the same GCB resumes from the right ring offset,
// and acks the socket page.
//
// Stop sentinel = a request page whose first word (num_tensors) is zero.
//
// Per-request payload layout (one socket page, fixed max size):
//   [0]  num_tensors        (0 = stop)
//   [1]  num_layers
//   [2]  num_blocks         (= num_senders * num_receivers, set by host)
//   [3]  gcb_state_addr     (DRISC L1 base of this GCB's sender state block)
//   [4 + 10*t + 0..9]       TensorGeom block per tensor t:
//                             bank_local_base, num_sub, M, rows_per_sub,
//                             coalesced_page_size, coalesced_num_pages,
//                             sub_chunk_bytes, sub_stride_bytes,
//                             block_stride_bytes, page_bytes_per_recv
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

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

namespace {

// Short aliases for the host-side layout constants — single source of truth in
// tt_metal/impl/buffers/dram_sender_state_block.hpp. The fields beyond byte
// 0x20 are persistent prefetcher state; only fifo_wr_ptr (inside the 32 B
// iface region) needs to round-trip on each request.
constexpr uint32_t kStateConfigPtr = tt::tt_metal::kDramSenderStateBlockConfigPtrOffset;
constexpr uint32_t kStateFifoStartAddr = tt::tt_metal::kDramSenderStateBlockFifoStartAddrOffset;
constexpr uint32_t kStateFifoLimit = tt::tt_metal::kDramSenderStateBlockFifoLimitOffset;
constexpr uint32_t kStateFifoPageSize = tt::tt_metal::kDramSenderStateBlockFifoPageSizeOffset;
constexpr uint32_t kStateFifoWrPtr = tt::tt_metal::kDramSenderStateBlockFifoWrPtrOffset;
constexpr uint32_t kStateReceiverNocXyPtr = tt::tt_metal::kDramSenderStateBlockReceiverNocXyPtrOffset;
constexpr uint32_t kStateAlignedPagesSentPtr = tt::tt_metal::kDramSenderStateBlockAlignedPagesSentPtrOffset;
constexpr uint32_t kStateNumRecvAndRemotePtr = tt::tt_metal::kDramSenderStateBlockNumRecvAndRemotePtrOffset;

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

// Loads the per-GCB sender state block (first 32 B, byte-compatible with
// RemoteSenderCBInterface) from L1 into the static cb_interface[] slot for
// this request.
FORCE_INLINE void load_sender_state(uint32_t state_addr, RemoteSenderCBInterface& iface) {
    volatile tt_l1_ptr uint32_t* sb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_addr);
    iface.config_ptr = sb[0];
    iface.fifo_start_addr = sb[1];
    iface.fifo_limit_page_aligned = sb[2];
    iface.fifo_page_size = sb[3];
    iface.fifo_wr_ptr = sb[4];
    iface.receiver_noc_xy_ptr = sb[5];
    iface.aligned_pages_sent_ptr = sb[6];
    iface.num_receivers_and_remote_pages_sent_ptr = sb[7];
}

// Writes back only the field that needs to persist across requests targeting
// this GCB. Per-tensor fields (fifo_limit / fifo_page_size) are overwritten by
// resize_remote_sender_cb_interface on every new request, so we don't need to
// round-trip them.
FORCE_INLINE void store_sender_state(uint32_t state_addr, const RemoteSenderCBInterface& iface) {
    volatile tt_l1_ptr uint32_t* sb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(state_addr);
    sb[4] = iface.fifo_wr_ptr;
}

}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    // num_receivers used to live here, but it's now per-GCB: each request's
    // state block carries its own num_receivers (see GCB ctor's
    // DramSenderStateBlockHeader::num_receivers at offset 0x34). Different
    // GCBs queued against the same prefetcher can have different receiver
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

    // ---- Request loop ----
    while (true) {
        socket_wait_for_pages(socket, 1);

        volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.read_ptr);
        const uint32_t req_num_tensors = payload[0];
        if (req_num_tensors == 0) {
            // Stop sentinel.
            socket_pop_pages(socket, 1);
            socket_notify_sender(socket);
            break;
        }
        const uint32_t req_num_layers = payload[1];
        const uint32_t req_num_blocks = payload[2];
        const uint32_t gcb_state_addr = payload[3];

        load_sender_state(gcb_state_addr, iface);
        // num_receivers lives inside the GCB's state block at the sender-side
        // config block offset (see DramSenderStateBlockHeader::num_receivers).
        // Reading it per request lets a single prefetcher serve GCBs with
        // different receiver counts.
        const uint32_t num_receivers = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gcb_state_addr + 0x30)[1];

        for (uint32_t layer = 0; layer < req_num_layers; ++layer) {
            for (uint32_t t = 0; t < req_num_tensors; ++t) {
                volatile tt_l1_ptr uint32_t* g = payload + 4 + 10 * t;
                const uint32_t tensor_base = g[0];
                const uint32_t t_num_sub = g[1];
                const uint32_t t_M = g[2];
                const uint32_t t_rows_per_sub = g[3];
                const uint32_t t_coal_page_size = g[4];
                const uint32_t t_coal_num_pages = g[5];
                const uint32_t t_chunk_bytes = g[6];
                const uint32_t t_sub_stride = g[7];
                const uint32_t t_block_stride = g[8];
                const uint32_t t_page_bytes_per_recv = g[9];
                const uint32_t t_recv_per_chunk = num_receivers / t_M;
                const uint32_t t_sub_band_per_block = t_num_sub * t_M;

                experimental::resize_remote_sender_cb_interface</*update_remote_over_noc=*/false>(
                    remote_cb_id, t_page_bytes_per_recv, noc_index);

                const uint32_t total_chunks = req_num_blocks * t_sub_band_per_block;

                experimental::dma_async_read(/*stream=*/0, tensor_base, stage_slot_a, t_chunk_bytes);

                uint32_t fifo_snapshot = 0;
                uint32_t cum_offset_in_page = 0;
                uint32_t blk = 0;
                uint32_t sb = 0;
                uint32_t ch = 0;
                constexpr uint32_t stage_slot_sum = stage_slot_a + stage_slot_b;
                uint32_t stage_slot = stage_slot_a;
                bool has_next = (total_chunks > 1);

                for (uint32_t c = 0; c < total_chunks; ++c) {
                    if (sb == 0 && ch == 0) {
                        experimental::remote_cb_reserve_back(remote_cb_id, 1);
                        fifo_snapshot = iface.fifo_wr_ptr;
                        cum_offset_in_page = 0;
                    }

                    uint32_t next_ch = ch + 1;
                    uint32_t next_sb = sb;
                    uint32_t next_blk = blk;
                    if (next_ch == t_M) {
                        next_ch = 0;
                        ++next_sb;
                        if (next_sb == t_num_sub) {
                            next_sb = 0;
                            ++next_blk;
                            if (next_blk == req_num_blocks) {
                                has_next = false;
                            }
                        }
                    }
                    const uint32_t next_slot = stage_slot_sum - stage_slot;

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

                    blk = next_blk;
                    sb = next_sb;
                    ch = next_ch;
                    stage_slot = next_slot;
                }

                if (t == req_num_tensors - 1) {
                    experimental::remote_cb_sender_barrier(remote_cb_id);
                }
            }
        }

        // Persist mutable state (fifo_wr_ptr) so the next request to this GCB
        // resumes at the right ring offset.
        store_sender_state(gcb_state_addr, iface);

        socket_pop_pages(socket, 1);
        socket_notify_sender(socket);
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    experimental::drisc_set_noc2axi_mode();
}
