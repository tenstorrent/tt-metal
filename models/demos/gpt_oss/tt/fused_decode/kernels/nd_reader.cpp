// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert weight streamer (NCRISC, NOC0) of the TP=1 decode MoE (nd.py): per group (g output columns starting
// at tile col0) and routed expert, CB slots of kb K rows x g columns (row-major [k][c], the custom_mm in1 block with
// ct_dim = g) streamed from the ND-sharded expert weights by nd_stream.hpp; with streams = 2 only the even K blocks
// (the writer streams the odd ones). sets = 2 (gate|up): every block is followed by the same block of the up weight.
//
// runtime args: [w_addr, u_addr, b_addr, idx_addr, vc, n_groups, e0, col0 * n_groups] (routed experts e0 ..
// e0 + num_sel - 1 of the router's index list)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "nd_stream.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);    // weight K rows
    constexpr uint32_t nt = get_compile_time_arg_val(4);    // weight columns
    constexpr uint32_t nt_b = get_compile_time_arg_val(5);  // bias columns
    constexpr uint32_t kb = get_compile_time_arg_val(6);    // weight rows per slot, divides kt
    constexpr uint32_t g = get_compile_time_arg_val(7);
    constexpr uint32_t num_sel = get_compile_time_arg_val(8);
    constexpr uint32_t sets = get_compile_time_arg_val(9);
    constexpr uint32_t up_off = get_compile_time_arg_val(10);
    constexpr uint32_t depth = get_compile_time_arg_val(11);
    constexpr uint32_t pkt_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t streams = get_compile_time_arg_val(13);
    constexpr auto w_args = TensorAccessorArgs<14>();
    constexpr auto u_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t u_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t idx_addr = get_arg_val<uint32_t>(a++);
    const uint32_t vc = get_arg_val<uint32_t>(a++);
    const uint32_t n_groups = get_arg_val<uint32_t>(a++);
    const uint32_t e0 = get_arg_val<uint32_t>(a++);

    const auto s_w = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto s_u = TensorAccessor(u_args, u_addr, tile_bytes);
    const auto s_b = TensorAccessor(b_args, b_addr, tile_bytes);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, 64);

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    noc_async_read(s_idx.get_noc_addr(0), idx_l1, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    uint32_t experts[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        experts[e] = ids[e0 + e];
    }

    gpt_oss_nd::nd_stream_slots<tile_bytes, kt, nt, nt_b, kb, g, num_sel, sets, up_off, depth, pkt_tiles, streams>(
        cb_w, s_w, s_u, s_b, experts, n_groups, a, 0, vc);
}
