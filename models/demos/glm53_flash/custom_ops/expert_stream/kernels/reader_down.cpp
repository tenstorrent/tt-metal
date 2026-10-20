// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert down-projection weight streamer (NCRISC, NOC0) for the GLM-5.3-Flash expert stream
// (custom_ops/expert_stream/op.py, after models/demos/gpt_oss/tt/fused_decode/kernels/stream_down_reader.cpp).
//
// The chip's down slice [E, I, H] is stored per DRAM bank: bank b holds output columns [b * C, (b + 1) * C) of every
// expert, expert-major then column-major (seg_tiles K tiles per column), so expert e's `cols` columns of this core
// are one contiguous run of cols * seg_tiles tiles at e * bank_expert_stride + c0 * seg_bytes. For every active
// expert (the routing matrix is scanned here) that block is read in `cols` packets of one column segment each,
// triple-buffered with transaction ids (one expert in flight ahead of compute).
//
// runtime args: [w_addr, routing_addr, bank_id, vc, c0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
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
    constexpr auto routing_args = TensorAccessorArgs<8>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = cols * seg_tiles;
    constexpr uint32_t block_bytes = cols * seg_bytes;
    static_assert(seg_bytes <= NOC_MAX_BURST_SIZE, "one column segment must fit one NOC packet");

    cb_reserve_back(cb_r, E_TILES);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const auto s_r = TensorAccessor(routing_args, routing_addr, 2048);
    for (uint32_t i = 0; i < E_TILES; ++i) {
        noc_async_read(s_r.get_noc_addr(i), r_l1 + i * 2048, 2048);
    }
    noc_async_read_barrier();
    uint16_t active[E];
    const uint32_t n = scan_active<E, M>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1), active);

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, seg_bytes, vc);

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
        uint32_t src = static_cast<uint32_t>(active[i]) * bank_expert_stride + c0 * seg_bytes;
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
            src += seg_bytes;
            l1_write += seg_bytes;
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
}
