// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0) of the deduplicated MoE stream with gate|up and down in one launch (tt/moe_stream.py FusedStream):
// the routing of user `bank` (bank core b routes user b, and with two row halves also user b + banks), shared with
// every bank core through the routing table,
// then from this core's DRAM bank the gate|up slice of every distinct expert the active users selected, then the
// down slice of the same experts, one stream.
//
// Routing table (cb_tbl, the same L1 address on every bank core): entry u (entry_bytes apart) = [active, ids[topk],
// weight bits[topk]] of user u. Core b writes entry b into every bank core's table, then raises their route_sem; a
// core reads the table once route_sem counts banks. The distinct experts are listed in user order (the active users'
// top-k in order, each expert at its first selection; with no active user, user 0's experts, weighted zero) by
// mimo_moe::table_experts, on every bank core and both of its data-movement RISCs alike. With two row halves (16
// users) each half lists its own (table_experts_halves), half 0's first: an expert both halves selected streams twice,
// and compute multiplies each listing by its half's 8 rows only.
//
// w_gate_up [1, E, K, 2 * 8 * 32]: bank b holds gate column b and up column b of every expert, contiguous per
// expert as [kt][gate, up]. w_down [1, E, I, 8 * cols * 32]: bank b holds expert e as I / 32 rows of cols tiles
// (output columns b, b + 8, ...), contiguous per expert. Both stream in blocks of the same size: block_k rows of
// (gate, up) for gate|up, down_block_rows rows of cols tiles for down, so the down blocks are each expert's slice in
// order and the ring of num_buffers blocks stays aligned.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_route_select.hpp"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tbl = get_compile_time_arg_val(2);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(3);
    constexpr uint32_t topk = get_compile_time_arg_val(4);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t blocks_per_pair = get_compile_time_arg_val(7);
    constexpr uint32_t packet_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t packets_per_block = get_compile_time_arg_val(9);
    constexpr uint32_t expert_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(11);
    constexpr uint32_t in_flight = get_compile_time_arg_val(12);
    constexpr uint32_t stage_pages = get_compile_time_arg_val(13);
    constexpr uint32_t expert_tiles = get_compile_time_arg_val(14);
    constexpr bool norm = get_compile_time_arg_val(15) == 1;
    constexpr bool has_scale = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(17);
    constexpr bool has_mask = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t rows = get_compile_time_arg_val(19);         // the user rows of the score tiles: 4, 8 or 16
    constexpr uint32_t down_blocks = get_compile_time_arg_val(20);  // blocks per expert's down slice
    constexpr uint32_t down_expert_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t banks = get_compile_time_arg_val(22);
    constexpr uint32_t route_sem = get_compile_time_arg_val(23);
    constexpr uint32_t entry_bytes = get_compile_time_arg_val(24);
    // Timing experiments only (MIMO_MOE_STREAM_DEBUG; wrong results): 1 = no weight reads, the stream only cycles
    // the ring (compute's own rate).
    constexpr uint32_t debug = get_compile_time_arg_val(25);
    constexpr uint32_t halves = get_compile_time_arg_val(26);  // row halves of 8 users: 1, or 2 (16 users)
    constexpr uint32_t users = num_pairs / topk;
    static_assert(down_blocks * block_bytes == down_expert_bytes);
    static_assert(users <= banks * halves && (2 * topk + 1) * 4 <= entry_bytes && (halves == 1 || users == 16));

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t biased_addr = get_arg_val<uint32_t>(3);
    const uint32_t scores_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);
    const uint32_t w_down_addr = get_arg_val<uint32_t>(6);
    constexpr uint32_t peers_at = 7;  // [x0, y0, ..., x_{banks-1}, y_{banks-1}]: the bank cores, bank order

    MOE_MARK(10);
    const uint32_t tbl = get_write_ptr(cb_tbl);
    cb_reserve_back(cb_stage, stage_pages);
    const uint32_t stage = get_write_ptr(cb_stage);
    mimo_moe::stage_route_rows<expert_tiles, has_mask, rows>(biased_addr, scores_addr, mask_addr, stage);
    noc_async_read_barrier();
    MOE_MARK(11);

    // This core's users: their entries into every bank core's table (user 0's ids even when idle: the fallback).
    for (uint32_t u = bank_id; u < users; u += banks) {
        const uint32_t mine = tbl + u * entry_bytes;
        volatile tt_l1_ptr uint32_t* entry = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine);
        const bool active = mimo_moe::user_active<expert_tiles, has_mask, rows>(stage, u);
        uint32_t ids[topk];
        uint32_t weights[topk];
        if (active || u == 0) {
            mimo_moe::select_user<topk, expert_tiles, norm, has_scale, scale_bits, has_mask, rows>(
                stage, u, ids, weights);
        } else {
            for (uint32_t i = 0; i < topk; i++) {
                ids[i] = 0;
                weights[i] = 0;
            }
        }
        entry[0] = active ? 1 : 0;
        for (uint32_t i = 0; i < topk; i++) {
            entry[1 + i] = ids[i];
            entry[1 + topk + i] = weights[i];
        }
        for (uint32_t c = 0; c < banks; c++) {
            if (c != bank_id) {
                const uint32_t x = get_arg_val<uint32_t>(peers_at + 2 * c);
                const uint32_t y = get_arg_val<uint32_t>(peers_at + 2 * c + 1);
                noc_async_write(mine, get_noc_addr(x, y, mine), entry_bytes);
            }
        }
    }
    noc_async_write_barrier();
    const uint32_t sem_addr = get_semaphore(route_sem);
    for (uint32_t c = 0; c < banks; c++) {
        const uint32_t x = get_arg_val<uint32_t>(peers_at + 2 * c);
        const uint32_t y = get_arg_val<uint32_t>(peers_at + 2 * c + 1);
        noc_semaphore_inc(get_noc_addr(x, y, sem_addr), 1);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), banks);
    MOE_MARK(12);

    uint32_t uniq[num_pairs];
    uint32_t counts[halves];
    const uint32_t n = halves == 1 ? mimo_moe::table_experts<topk, users, expert_tiles * 32>(tbl, entry_bytes, uniq)
                                   : mimo_moe::table_experts_halves<topk, users / halves, halves, expert_tiles * 32>(
                                         tbl, entry_bytes, uniq, counts);
    MOE_MARK(13);

    // Stream: block i goes to CB slot i % num_buffers with trid (i % num_buffers) + 1; at most in_flight blocks
    // are issued but not pushed, and each reserve covers them plus the new block. Blocks [0, gate_up) are gate|up,
    // the rest down; both tensors live in this bank, so one NoC state serves them.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t gu_base = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(bank_id, w_addr));
    const uint32_t down_base = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(bank_id, w_down_addr));
    noc_async_read_one_packet_set_state<true>(get_noc_addr_from_bank_id<true>(bank_id, w_addr), packet_bytes, vc);
    const uint32_t cb_base = get_write_ptr(cb_w);

    const uint32_t gate_up = n * blocks_per_pair;
    const uint32_t total = gate_up + n * down_blocks;
    uint32_t issued = 0;
    uint32_t oldest = 0;
    for (uint32_t i = 0; i < total; i++) {
        if (issued == in_flight) {
            noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
            cb_push_back(cb_w, block_tiles);
            oldest++;
            issued--;
        }
        cb_reserve_back(cb_w, block_tiles * (issued + 1));
        const uint32_t trid = (i % num_buffers) + 1;
        noc_async_read_set_trid(trid);
        uint32_t base, src;
        if (i < gate_up) {
            const uint32_t e = i / blocks_per_pair;
            base = gu_base;
            src = uniq[e] * expert_bytes + (i - e * blocks_per_pair) * block_bytes;
        } else {
            const uint32_t d = i - gate_up;
            const uint32_t e = d / down_blocks;
            base = down_base;
            src = uniq[e] * down_expert_bytes + (d - e * down_blocks) * block_bytes;
        }
        uint32_t dst = cb_base + (i % num_buffers) * block_bytes;
        for (uint32_t k = 0; k < packets_per_block && debug != 1; k++) {
            noc_async_read_one_packet_with_state_with_trid(base, src, dst, trid);
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
    MOE_MARK(14);
}
