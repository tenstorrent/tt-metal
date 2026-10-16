// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): routing of users [0, local_users) (the writer routes the rest), then every pair's gate|up expert
// slice streamed from this core's DRAM bank; bank 0 finally writes the uint16 expert-id stick for the down matmul.
//
// The DRAM-interleaved w_gate_up [1, E, K, 2 * 8 * 32] puts tile column n in bank n % 8, so bank b holds gate
// column b and up column b of every expert, contiguous per expert as [kt][gate, up] (one ct = 2 matmul block).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_route_select.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(3);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t blocks_per_pair = get_compile_time_arg_val(8);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t packets_per_block = get_compile_time_arg_val(10);
    constexpr uint32_t expert_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(12);
    constexpr uint32_t in_flight = get_compile_time_arg_val(13);
    constexpr uint32_t local_users = get_compile_time_arg_val(14);
    constexpr uint32_t stage_pages = get_compile_time_arg_val(15);
    constexpr uint32_t expert_tiles = get_compile_time_arg_val(16);
    constexpr bool norm = get_compile_time_arg_val(17) == 1;
    constexpr bool has_scale = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(19);
    constexpr bool has_mask = get_compile_time_arg_val(20) == 1;

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t biased_addr = get_arg_val<uint32_t>(3);
    const uint32_t scores_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);
    const uint32_t ids_addr = get_arg_val<uint32_t>(6);

    using Stage = mimo_moe::RouteStage<expert_tiles>;
    cb_reserve_back(cb_stage, stage_pages);
    const uint32_t stage = get_write_ptr(cb_stage);
    mimo_moe::stage_route_rows<expert_tiles, has_mask>(biased_addr, scores_addr, mask_addr, stage);
    noc_async_read_barrier();

    // Route pages (fp32 weight bits for compute): page 0 holds the local pairs [0, local_pairs), page 1 the writer's
    // pairs. Page 0 is published before streaming; page 1 when the stream reaches the first remote pair.
    constexpr uint32_t local_pairs = local_users * topk;
    constexpr uint32_t remote_pairs = num_pairs - local_pairs;
    cb_reserve_back(cb_route, 2);
    const uint32_t route_l1 = get_write_ptr(cb_route);
    volatile tt_l1_ptr uint32_t* w_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_l1);
    volatile tt_l1_ptr uint32_t* w_remote =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(route_l1 + mimo_moe::kFp32TileBytes);
    volatile tt_l1_ptr uint16_t* stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + Stage::stick);
    uint32_t offsets[num_pairs];
    for (uint32_t u = 0; u < local_users; u++) {
        uint32_t ids[topk];
        uint32_t weights[topk];
        mimo_moe::select_user<topk, expert_tiles, norm, has_scale, scale_bits, has_mask>(stage, u, ids, weights);
        for (uint32_t i = 0; i < topk; i++) {
            const uint32_t p = u * topk + i;
            offsets[p] = ids[i] * expert_bytes;
            w_out[p] = weights[i];
            stick[p] = static_cast<uint16_t>(ids[i]);
        }
    }
    cb_push_back(cb_route, 1);

    // The writer's users: [remote ids | remote weight bits]; taken just before the first remote block is issued.
    auto take_remote = [&]() {
        cb_wait_front(cb_sel, 1);
        volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sel));
        for (uint32_t j = 0; j < remote_pairs; j++) {
            const uint32_t p = local_pairs + j;
            const uint32_t expert = sel[j];
            offsets[p] = expert * expert_bytes;
            w_remote[j] = sel[remote_pairs + j];
            stick[p] = static_cast<uint16_t>(expert);
        }
        cb_pop_front(cb_sel, 1);
        cb_push_back(cb_route, 1);
    };

    // Stream: block i goes to CB slot i % num_buffers with trid (i % num_buffers) + 1; at most in_flight blocks
    // are issued but not pushed, and each reserve covers them plus the new block.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint64_t w_noc_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(w_noc_base, packet_bytes, vc);
    const uint32_t cb_base = get_write_ptr(cb_w);

    constexpr uint32_t total_blocks = num_pairs * blocks_per_pair;
    constexpr uint32_t local_blocks = local_pairs * blocks_per_pair;
    uint32_t issued = 0;
    uint32_t oldest = 0;
    for (uint32_t i = 0; i < total_blocks; i++) {
        if (i == local_blocks) {
            take_remote();
        }
        if (issued == in_flight) {
            noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
            cb_push_back(cb_w, block_tiles);
            oldest++;
            issued--;
        }
        cb_reserve_back(cb_w, block_tiles * (issued + 1));
        const uint32_t trid = (i % num_buffers) + 1;
        noc_async_read_set_trid(trid);
        uint32_t src = offsets[i / blocks_per_pair] + (i % blocks_per_pair) * block_bytes;
        uint32_t dst = cb_base + (i % num_buffers) * block_bytes;
        for (uint32_t k = 0; k < packets_per_block; k++) {
            noc_async_read_one_packet_with_state_with_trid(w_noc_base, src, dst, trid);
            src += packet_bytes;
            dst += packet_bytes;
        }
        issued++;
    }
    while (issued > 0) {
        noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
        cb_push_back(cb_w, block_tiles);
        oldest++;
        issued--;
    }
    if constexpr (local_blocks >= total_blocks) {
        take_remote();  // no remote pairs: still consume cb_sel and publish page 1
    }

    // Expert ids for the down sparse_matmul: row-major uint16 [1, 1, 1, pairs], one page in bank 0.
    if (bank_id == 0) {
        noc_async_write(stage + Stage::stick, get_noc_addr_from_bank_id<true>(0, ids_addr), num_pairs * 2);
        noc_async_write_barrier();
    }
}
