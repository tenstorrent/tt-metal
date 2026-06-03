// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC prefetcher kernel for the DRAM-core mode of ttnn.dram_prefetcher.
// See tt_metal/impl/buffers/prefetcher_matmul_design.md for the architecture
// and contract; in particular §3 (what a "block" is), §6 (DRAM-core path: fit
// ladder, L1 layout, helpers), and §8 (cross-component invariants).
//
// Pipeline summary:
//   For each (layer, tensor, block), reserve one fifo page on every receiver,
//   then iterate (sub-band sb, chunk ch) over num_sub[t] * M[t] (sb, ch)
//   pairs, streaming DMA chunks of sub_chunk_bytes[t] into the ping-pong
//   stage ring and pushing each via prefetcher_write_chunk. At the end of
//   the block, prefetcher_finalize_block bumps pages_sent once (per receiver)
//   and advances iface.fifo_wr_ptr by page_bytes_per_recv[t].
//
//   When (rows_per_sub, M) == (k_block_w_tiles, 1) the inner loop reduces to
//   a single chunk + finalize per block (the fast path).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER)
#include "api/debug/ring_buffer.h"
#include "internal/tt-1xx/risc_common.h"
#define DRISC_PROFILE 1
#endif

#ifdef DRISC_PROFILE
#define PROFILE_T0() uint32_t _prof_t0 = get_timestamp_32b()
#define PROFILE_ACCUM(acc)                       \
    do {                                         \
        uint32_t _prof_t1 = get_timestamp_32b(); \
        (acc) += _prof_t1 - _prof_t0;            \
        _prof_t0 = _prof_t1;                     \
    } while (0)
#else
#define PROFILE_T0() ((void)0)
#define PROFILE_ACCUM(acc) ((void)0)
#endif

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

namespace {

// Writes one sub-band chunk's data to `num_receivers_in_chunk` receivers, each
// receiver getting `num_rows` rows of (coalesced_num_pages_per_row *
// coalesced_page_size) bytes, starting at `dest_l1_base` on the receiver. The
// stage holds data laid out so receiver r's row h is at
//
//     src_l1_addr
//       + h * (num_receivers_in_chunk * coalesced_num_pages_per_row * coalesced_page_size)
//       + r * (coalesced_num_pages_per_row * coalesced_page_size)
//
// Mirrors the inner write loop of remote_cb_push_back_and_write_pages at
// tt_metal/hw/inc/api/remote_circular_buffer.h:368-389, but takes an explicit
// `dest_l1_base` and `recv_xy` so the caller drives receiver subset selection
// and dest offset directly. No pages_sent / semaphore side effects, no
// iface.fifo_wr_ptr advance — those happen exactly once per block in
// prefetcher_finalize_block.
// Template-specialized fast paths for the common cases:
//   - `single_row`        — num_rows == 1 (no outer h loop)
//   - `single_page`       — coalesced_num_pages_per_row == 1 (no inner w loop)
// Dispatched at per-tensor scope so the per-chunk hot path has no branch on
// loop-shape. The general path remains for `<false,false>`.
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
            // Multi-write paths use with_state, which requires set_state to have configured
            // the cmd_buf first.
            noc_async_write_one_packet_set_state</*posted=*/true>(set_state_dest, coalesced_page_size, noc);
        }

        uint32_t src_addr = src_l1_addr + recv_src_offset;
        if constexpr (single_row && single_page) {
            // 1 write per receiver. Use the fused noc_async_write_one_packet to skip
            // the redundant `set_state` cmd_buf_ready spin — only one cmd is issued per
            // receiver, so amortizing state across multiple `with_state` calls buys
            // nothing here. One spin per receiver instead of two.
            noc_async_write_one_packet</*enable_noc_tracing=*/false, /*posted=*/true>(
                src_addr, set_state_dest, coalesced_page_size, noc);
        } else if constexpr (single_row) {
            // num_rows == 1; only inner w loop runs.
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                const uint64_t dest_noc = get_noc_addr_helper(remote_noc_xy, dest_addr);
                noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_noc, noc);
                src_addr += coalesced_page_size;
                dest_addr += coalesced_page_size;
            }
        } else if constexpr (single_page) {
            // coalesced_num_pages_per_row == 1; only outer h loop runs.
            for (uint32_t h = 0; h < num_rows; ++h) {
                const uint64_t dest_noc = get_noc_addr_helper(remote_noc_xy, dest_addr);
                noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dest_noc, noc);
                src_addr += row_stride_in_stage;
                dest_addr += coalesced_page_size;
            }
        } else {
            // General: nested h × w loops.
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

// Finalizes one ring-block: bumps local pages_sent + remote NoC semaphore for
// every receiver by the per-block aligned-page count, then advances
// iface.fifo_wr_ptr by page_bytes_per_recv (wrapping at fifo_limit). Called
// once per block, after all chunks have been written via prefetcher_write_chunk.
//
// The aligned-page count includes any wrap "gap" (mirrors the len_bytes
// adjustment in remote_cb_push_back_and_write_pages at remote_circular_buffer.h:347-350)
// so the receiver's pages_sent counter advances correctly across the buffer
// boundary.
//
// With skip_ptr_update=true the NoC semaphore increments are posted; the caller
// drains them once at end of stream via update_remote_cb_config_in_l1 +
// noc_async_atomic_barrier (the same pattern remote_cb_push_back_and_write_pages
// uses when its template flag is true).
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

}  // namespace

void kernel_main() {
    // ---- Compile-time args (13) ----
    constexpr uint32_t num_layers = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(3);
    constexpr uint32_t stage_ring_base = get_compile_time_arg_val(4);
    constexpr uint32_t stage_ring_size = get_compile_time_arg_val(5);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t pages_sent_l1_addr = get_compile_time_arg_val(7);
    constexpr uint32_t noc_xy_l1_addr = get_compile_time_arg_val(8);
    constexpr uint32_t config_l1_addr = get_compile_time_arg_val(9);
    constexpr uint32_t fifo_size_per_receiver = get_compile_time_arg_val(10);
    constexpr uint32_t receiver_buffer_address = get_compile_time_arg_val(11);
    constexpr uint32_t remote_pages_sent_worker_l1_addr = get_compile_time_arg_val(12);
    constexpr uint32_t ring_half = stage_ring_size / 2;
    constexpr uint32_t stage_slot_a = stage_ring_base;
    constexpr uint32_t stage_slot_b = stage_ring_base + ring_half;
    (void)fifo_size_per_receiver;  // stored in cfg[3]; iface uses it via config_ptr.

    // ---- Runtime args ----
    // [0]: bank_id
    // Then per-tensor blocks of length num_tensors each:
    //   bank_local_base, num_sub, M, rows_per_sub, coalesced_page_size,
    //   coalesced_num_pages, sub_chunk_bytes, sub_stride_bytes,
    //   block_stride_bytes, page_bytes_per_recv
    // Finally [2 * num_receivers] noc_x/noc_y pairs.
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);
    (void)bank_id;

    const uint32_t bank_local_base_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t num_sub_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t M_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t rows_per_sub_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t coal_page_size_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t coal_num_pages_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t sub_chunk_bytes_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t sub_stride_bytes_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t block_stride_bytes_idx = rt_idx;
    rt_idx += num_tensors;
    const uint32_t page_bytes_per_recv_idx = rt_idx;
    rt_idx += num_tensors;

    // ---- One-time L1 setup ----
    volatile tt_l1_ptr uint32_t* noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_l1_addr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        noc_xy_ptr[2 * i] = get_arg_val<uint32_t>(rt_idx++);
        noc_xy_ptr[2 * i + 1] = get_arg_val<uint32_t>(rt_idx++);
    }
    volatile tt_l1_ptr uint32_t* psem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pages_sent_l1_addr);
    const uint32_t psem_uints = (2 * L1_ALIGNMENT * num_receivers) / sizeof(uint32_t);
    for (uint32_t i = 0; i < psem_uints; ++i) {
        psem[i] = 0;
    }
    volatile tt_l1_ptr uint32_t* cfg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);
    cfg[0] = 1;
    cfg[1] = num_receivers;
    cfg[2] = receiver_buffer_address;
    cfg[3] = fifo_size_per_receiver;

    RemoteSenderCBInterface& iface = get_remote_sender_cb_interface(remote_cb_id);
    iface.config_ptr = config_l1_addr;
    iface.fifo_start_addr = receiver_buffer_address;
    iface.fifo_wr_ptr = receiver_buffer_address;
    iface.receiver_noc_xy_ptr = noc_xy_l1_addr;
    iface.aligned_pages_sent_ptr = pages_sent_l1_addr;
    iface.num_receivers_and_remote_pages_sent_ptr = remote_cb_pack(num_receivers, remote_pages_sent_worker_l1_addr);

    experimental::drisc_set_stream_mode();

    const uint32_t* bank_local_bases = reinterpret_cast<uint32_t*>(get_arg_addr(bank_local_base_idx));
    const uint32_t* num_subs = reinterpret_cast<uint32_t*>(get_arg_addr(num_sub_idx));
    const uint32_t* Ms = reinterpret_cast<uint32_t*>(get_arg_addr(M_idx));
    const uint32_t* rows_per_subs = reinterpret_cast<uint32_t*>(get_arg_addr(rows_per_sub_idx));
    const uint32_t* coal_page_sizes = reinterpret_cast<uint32_t*>(get_arg_addr(coal_page_size_idx));
    const uint32_t* coal_num_pages = reinterpret_cast<uint32_t*>(get_arg_addr(coal_num_pages_idx));
    const uint32_t* sub_chunk_bytes = reinterpret_cast<uint32_t*>(get_arg_addr(sub_chunk_bytes_idx));
    const uint32_t* sub_stride_bytes = reinterpret_cast<uint32_t*>(get_arg_addr(sub_stride_bytes_idx));
    const uint32_t* block_stride_bytes = reinterpret_cast<uint32_t*>(get_arg_addr(block_stride_bytes_idx));
    const uint32_t* page_bytes_per_recv = reinterpret_cast<uint32_t*>(get_arg_addr(page_bytes_per_recv_idx));

    // ---- Profiling accumulators (zero-cost when WATCHER_DISABLE_RING_BUFFER) ----
#ifdef DRISC_PROFILE
    uint32_t prof_reserve = 0;   // 0xA3
    uint32_t prof_issue = 0;     // 0xA1 (next-DMA issue)
    uint32_t prof_wait = 0;      // 0xA2 (DMA wait)
    uint32_t prof_push = 0;      // 0xA4 (prefetcher_write_chunk)
    uint32_t prof_flush = 0;     // 0xA5 (noc_async_posted_writes_flushed)
    uint32_t prof_finalize = 0;  // 0xA6 (prefetcher_finalize_block)
    uint32_t prof_chunks = 0;    // 0xFF (chunk count divisor)
#endif

    // ---- Main loop ----
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t t = 0; t < num_tensors; ++t) {
            const uint32_t tensor_base = bank_local_bases[t];
            const uint32_t t_num_sub = num_subs[t];
            const uint32_t t_M = Ms[t];
            const uint32_t t_rows_per_sub = rows_per_subs[t];
            const uint32_t t_coal_page_size = coal_page_sizes[t];
            const uint32_t t_coal_num_pages = coal_num_pages[t];
            const uint32_t t_chunk_bytes = sub_chunk_bytes[t];
            const uint32_t t_sub_stride = sub_stride_bytes[t];
            const uint32_t t_block_stride = block_stride_bytes[t];
            const uint32_t t_page_bytes_per_recv = page_bytes_per_recv[t];
            const uint32_t t_recv_per_chunk = num_receivers / t_M;
            const uint32_t t_sub_band_per_block = t_num_sub * t_M;

            // Set the sender fifo page size to one full per-receiver page so
            // remote_cb_reserve_back reserves exactly one page per receiver.
            experimental::resize_remote_sender_cb_interface</*update_remote_over_noc=*/false>(
                remote_cb_id, t_page_bytes_per_recv, noc_index);

            const uint32_t total_chunks = num_blocks * t_sub_band_per_block;

            // Prologue: issue first DMA (chunk 0) into stage_slot_a. The body's
            // "issue next" lookahead populates subsequent slots.
            experimental::dma_async_read(/*stream=*/0, tensor_base, stage_slot_a, t_chunk_bytes);

            uint32_t fifo_snapshot = 0;
            uint32_t cum_offset_in_page = 0;

            // The flat chunk index `c` decomposes into three nested counters,
            // advanced by compare-and-increment (ch fastest, then sb, then blk):
            uint32_t blk = 0;  // K-block index within the layer, in [0, num_blocks)
            uint32_t sb = 0;   // sub-band index within the block, in [0, t_num_sub)
            uint32_t ch = 0;   // chunk index within the sub-band, in [0, t_M)
            // stage_slot ping-pongs between stage_slot_a and stage_slot_b; toggle via
            // `(a + b) - slot`.
            constexpr uint32_t stage_slot_sum = stage_slot_a + stage_slot_b;
            uint32_t stage_slot = stage_slot_a;
            // True for every chunk except the very last; flipped once the successor
            // counters cross num_blocks, so the hot path reads a flag instead of
            // recomputing the bound each iteration.
            bool has_next = (total_chunks > 1);

            for (uint32_t c = 0; c < total_chunks; ++c) {
                PROFILE_T0();

                if (sb == 0 && ch == 0) {
                    experimental::remote_cb_reserve_back(remote_cb_id, 1);
                    fifo_snapshot = iface.fifo_wr_ptr;
                    cum_offset_in_page = 0;
                    PROFILE_ACCUM(prof_reserve);
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
                        if (next_blk == num_blocks) {
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
                PROFILE_ACCUM(prof_issue);
                const uint32_t outstanding_after_wait = has_next ? 1u : 0u;
                experimental::dma_async_read_wait_n(/*stream=*/0, outstanding_after_wait);
                PROFILE_ACCUM(prof_wait);
                volatile tt_l1_ptr uint32_t* chunk_recv_xy =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_l1_addr) + ch * t_recv_per_chunk * 2;
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
                PROFILE_ACCUM(prof_push);

                if (ch + 1 == t_M) {
                    cum_offset_in_page += t_rows_per_sub * t_coal_num_pages * t_coal_page_size;
                }

                if (sb + 1 == t_num_sub && ch + 1 == t_M) {
                    noc_async_posted_writes_flushed();
                    PROFILE_ACCUM(prof_flush);
                    prefetcher_finalize_block</*skip_ptr_update=*/true>(
                        iface, t_page_bytes_per_recv, num_receivers, noc_index);
                    PROFILE_ACCUM(prof_finalize);
                } else {
                    // The ping-pong DMA can reuse this stage slot two chunks later.
                    // Make sure all posted writes sourced from it have departed first.
                    noc_async_posted_writes_flushed();
                    PROFILE_ACCUM(prof_flush);
                }

                // Advance counters to next chunk.
                blk = next_blk;
                sb = next_sb;
                ch = next_ch;
                stage_slot = next_slot;
#ifdef DRISC_PROFILE
                prof_chunks++;
#endif
            }

            if (t == num_tensors - 1) {
                experimental::remote_cb_sender_barrier(remote_cb_id);
            }
        }
    }

    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
    experimental::drisc_set_noc2axi_mode();

#ifdef DRISC_PROFILE
    // Tag-prefixed totals; decode newest-first via `grep debug_ring_buffer`. Each entry's
    // top byte is the stage tag; the low 24 bits are the cycle count (sufficient for
    // a few hundred thousand cycles — wraps cleanly at large counts).
    WATCHER_RING_BUFFER_PUSH(0xA1000000u | (prof_issue & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xA2000000u | (prof_wait & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xA3000000u | (prof_reserve & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xA4000000u | (prof_push & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xA5000000u | (prof_flush & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xA6000000u | (prof_finalize & 0x00FFFFFFu));
    WATCHER_RING_BUFFER_PUSH(0xFF000000u | (prof_chunks & 0x00FFFFFFu));
#endif
}
