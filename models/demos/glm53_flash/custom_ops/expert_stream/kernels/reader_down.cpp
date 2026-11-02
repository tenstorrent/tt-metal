// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert down-projection weight streamer (NCRISC, NOC0) for the GLM-5.3-Flash expert stream
// (custom_ops/expert_stream/op.py, after models/demos/gpt_oss/tt/fused_decode/kernels/stream_down_reader.cpp).
//
// The chip's down slice [E, I, H] is stored per DRAM bank: bank b holds output columns [b * C, (b + 1) * C) of every
// expert, expert-major then column-major (seg_tiles K tiles per column), so expert e's `cols` columns of this core
// are one contiguous run of cols * seg_tiles tiles at e * bank_expert_stride + c0 * seg_bytes. For every active
// expert (the routing matrix is scanned here) that block is read in packets of page_bytes (one column segment at
// I = 256, less for wider slices),
// triple-buffered with transaction ids (one expert in flight ahead of compute).
//
// runtime args: [w_addr, routing_addr, bank_id, vc, c0, (ROUTING_FOLD) base_addr]
// E1 variants ROUTING_FOLD / SCAN_WORDS as reader_gate_up.cpp (the routing window of this chip's E experts inside the
// router's full matrix via the per-chip base page; the word-wise scan).
// REGIONS_SRC (2026-10-27, regions_src.hpp / regions.py): the weights are the regions prefill path's per-expert
// DRAM-interleaved down tensors [I, H] (w_addr unused; common runtime args = down_addr[0..E)): tile (i, h) sits in
// bank h % banks at (i * C + h / banks) * tile_bytes (C = H / 32 / banks = bank_expert_stride / seg_bytes columns per
// bank), so this core's `cols` columns [c0, c0 + cols) of K tile i are ONE packet of cols tiles at
// (i * C + c0) * tile_bytes: seg_tiles packets per expert, k-major in the CB (compute_down.cpp REGIONS_SRC indexes
// tile i * cols + c). The same bytes and packet count as the stacked block at the 8x4 shapes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "regions_src.hpp"
#include "routing_scan.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t M = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t bank_expert_stride = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(8);  // NOC packet (divides the column segment)
    constexpr auto routing_args = TensorAccessorArgs<9>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
#endif

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = cols * seg_tiles;
    constexpr uint32_t block_bytes = cols * seg_bytes;
    constexpr uint32_t pages_per_block = block_bytes / page_bytes;
    static_assert(pages_per_block * page_bytes == block_bytes, "the packet must divide the expert block");
    static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "one packet must fit one NOC burst");

#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;
    cb_reserve_back(cb_r, R_TILES + 1);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const uint32_t base_addr = get_arg_val<uint32_t>(5);
    const auto s_b = TensorAccessor(base_args, base_addr, 32);
    const uint32_t b_l1 = r_l1 + R_TILES * 2048;
    noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
    noc_async_read_barrier();
    const uint32_t col_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
    const uint32_t tb = fold_first_tile(col_base);
    const uint32_t col_off = fold_col_off(col_base);
    const uint32_t n_tiles = fold_tiles<E>(col_base);
#else
    constexpr uint32_t R_TILES = E_TILES;
    cb_reserve_back(cb_r, E_TILES);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    constexpr uint32_t tb = 0;
    constexpr uint32_t col_off = 0;
    constexpr uint32_t n_tiles = E_TILES;
#endif
    const auto s_r = TensorAccessor(routing_args, routing_addr, 2048);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        noc_async_read(s_r.get_noc_addr(tb + i), r_l1 + i * 2048, 2048);
    }
    noc_async_read_barrier();
    uint16_t active[E];
    const volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);
#if defined(SCAN_WORDS)
    const uint32_t n = scan_active_words<E, M, R_TILES>(routing, active, col_off, n_tiles);
#elif defined(ROUTING_FOLD)
    const uint32_t n = scan_active_fold<E, M>(routing, active, col_off);
#else
    const uint32_t n = scan_active<E, M>(routing, active);
#endif

#ifdef REGIONS_SRC
    constexpr uint32_t C = bank_expert_stride / seg_bytes;  // output columns per bank
    constexpr uint32_t src_step = C * tile_bytes;           // the next K tile of the core's columns
    static_assert(page_bytes == cols * tile_bytes, "regions-sourced down packets are one K tile of the core's columns");
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
    (void)w_addr;  // the per-expert addresses come from the common runtime args
#else
    constexpr uint32_t src_step = page_bytes;
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
#endif
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;

    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read_set_trid(trid);
#ifdef REGIONS_SRC
        uint32_t src = get_common_arg_val<uint32_t>(active[i]) + c0 * tile_bytes;
#else
        uint32_t src = static_cast<uint32_t>(active[i]) * bank_expert_stride + c0 * seg_bytes;
#endif
        for (uint32_t c = 0; c < pages_per_block; ++c) {
            noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
            src += src_step;
            l1_write += page_bytes;
        }
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, block_tiles);
            ++pushed;
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
    while (pushed < n) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, block_tiles);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
