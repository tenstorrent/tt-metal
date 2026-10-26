// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up weight streamer (NCRISC, NOC0) for the GLM-5.3-Flash expert stream
// (custom_ops/expert_stream/op.py, after models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_reader.cpp).
//
// The chip's gate|up slice [E, K, 2I] is stored per DRAM bank: bank b holds (gate column pair b, up column pair b)
// of every expert, column-major, expert-major, so expert e's two columns for this bank are one contiguous range of
// 2 * ppb * kt tiles at e * expert_stride (ppb column pairs per bank). This core belongs to group g of G: it streams
// the active experts active[g], active[g + G], ... (the active list comes from the routing matrix scanned here), one
// column at a time, triple-buffered with transaction ids (one column in flight ahead of compute).
//
// runtime args: [w_addr, routing_addr, bank_id, vc, group, (ROUTING_FOLD) base_addr]
//
// E1 variants (decode roofline, 2026-10-26; op.py ``routing_fold`` / ``scan_words``, preprocessor-gated so the default
// program is the one above):
// * ROUTING_FOLD: the routing tensor is the router's full [M, E_total] matrix; this chip's column base (the first of
//   its E experts) comes from a 32 B uint16 page (`base_addr`, one per chip, mesh-sharded) and the kernel reads the
//   window of tiles covering [base, base + E) (routing_scan.hpp fold_*). Replaces the per-column selector matmul.
// * SCAN_WORDS: the word-wise active scan (routing_scan.hpp scan_active_words) instead of the element scan.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "routing_scan.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t group = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t M = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t ppb = get_compile_time_arg_val(9);  // column pairs per bank
    constexpr auto routing_args = TensorAccessorArgs<10>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
#endif

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the column");

    // routing matrix tiles -> active experts -> this group's share
#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;  // the worst-case fold window
    cb_reserve_back(cb_r, R_TILES + 1);          // + the base page
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

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    // Triple-buffered columns, one extra column in flight (transaction ids 1..3).
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;
    uint32_t issued = 0;

    for (uint32_t i = group; i < n; i += groups) {
        uint32_t src = static_cast<uint32_t>(active[i]) * expert_stride;
        for (uint32_t c = 0; c < 2 * ppb; ++c) {  // per pair: gate column, then up column
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < pages_per_col; ++p) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                src += page_bytes;
                l1_write += page_bytes;
            }
            ++issued;
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, kt);
                ++pushed;
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    while (pushed < issued) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, kt);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
