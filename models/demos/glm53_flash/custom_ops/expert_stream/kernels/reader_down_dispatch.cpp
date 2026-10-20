// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection weight streamer (NCRISC, NOC0) of the per-expert dispatch op (custom_ops/expert_stream/op.py,
// ExpertDownDispatch): as reader_down.cpp (this core's `cols` output columns of every active expert, one
// contiguous block per expert, triple-buffered), the active experts taken from the per-expert token counts; each
// expert's block is streamed once and kept by the compute over the expert's passes.
//
// runtime args: [w_addr, counts_addr (EP: idx_addr), bank_id, vc, c0, (EP) base_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatch_common.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t counts_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_c = get_compile_time_arg_val(1);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t bank_expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t EP = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(11);  // expert block slots (2 or 3)
    constexpr auto counts_args = TensorAccessorArgs<12>();
    constexpr auto base_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = cols * seg_tiles;
    constexpr uint32_t block_bytes = cols * seg_bytes;
    constexpr uint32_t pages_per_block = block_bytes / page_bytes;
    static_assert(pages_per_block * page_bytes == block_bytes, "the packet must divide the expert block");
    static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "one packet must fit one NOC burst");

    constexpr uint32_t c_tiles = EP ? T / 32 : E_TILES;
    cb_reserve_back(cb_c, c_tiles);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    const auto s_c = TensorAccessor(counts_args, counts_addr, 2048);
    for (uint32_t i = 0; i < c_tiles; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * 2048, 2048);
    }
    uint16_t n_tok[E];
    if constexpr (EP) {
        const uint32_t base_addr = get_arg_val<uint32_t>(5);
        const auto s_b = TensorAccessor(base_args, base_addr, 32);
        cb_reserve_back(cb_c, 1);
        const uint32_t b_l1 = c_l1 + c_tiles * 2048;
        noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
        noc_async_read_barrier();
        const uint32_t expert_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
        count_from_indices<E, T, K>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), expert_base, n_tok);
    } else {
        noc_async_read_barrier();
        read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    }

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    static_assert(num_buffers == 2 || num_buffers == 3, "two or three expert block slots");
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;
    uint32_t issued = 0;

    for (uint32_t e = 0; e < E; ++e) {
        if (n_tok[e] != 0) {  // the expert's block once; the compute keeps it over the expert's passes
            noc_async_read_set_trid(trid);
            uint32_t src = e * bank_expert_stride + c0 * seg_bytes;
            for (uint32_t c = 0; c < pages_per_block; ++c) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                src += page_bytes;
                l1_write += page_bytes;
            }
            ++issued;
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
    }
    while (pushed < issued) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, block_tiles);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
