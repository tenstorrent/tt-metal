// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0) of the deduplicated gate|up stream: routing of users [0, local_users) (the writer routes the rest),
// then the gate|up slice of every distinct expert the active users selected, streamed once from this core's DRAM
// bank; bank 0 finally writes the expert stick [count, ids...] for the down stream.
//
// The distinct experts are listed in user order (the active users' top-k in order, each expert at its first
// selection); the writer builds the same list for the per-row routing weights. The local users' experts stream
// first; the writer's users' selections (cb_sel) extend the list when the stream reaches its end. With no active
// user, user 0's experts are streamed (their weights are masked to zero).
//
// The DRAM-interleaved w_gate_up [1, E, K, 2 * 8 * 32] puts tile column n in bank n % 8, so bank b holds gate
// column b and up column b of every expert, contiguous per expert as [kt][gate, up] (one ct = 2 matmul block).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_route_select.hpp"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_local = get_compile_time_arg_val(1);
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
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(21);  // >= 2 * (pairs + 1), DRAM-write aligned
    constexpr uint32_t rows = get_compile_time_arg_val(22);         // the user rows of the score tiles: 4 or 8
    constexpr uint32_t users = num_pairs / topk;
    constexpr uint32_t remote_users = users - local_users;

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t biased_addr = get_arg_val<uint32_t>(3);
    const uint32_t scores_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);
    const uint32_t ids_addr = get_arg_val<uint32_t>(6);

    MOE_MARK(10);
    using Stage = mimo_moe::RouteStage<expert_tiles, rows>;
    cb_reserve_back(cb_stage, stage_pages);
    const uint32_t stage = get_write_ptr(cb_stage);
    mimo_moe::stage_route_rows<expert_tiles, has_mask, rows>(biased_addr, scores_addr, mask_addr, stage);
    noc_async_read_barrier();
    MOE_MARK(11);

    // The local users' selections go to the writer (it weights every distinct expert's rows): cb_local page =
    // [active flag per local user | ids | weight bits], each local_users * topk (flags: local_users) words.
    uint32_t uniq[num_pairs];
    uint32_t ids0[topk];
    uint32_t n = 0;
    cb_reserve_back(cb_local, 1);
    volatile tt_l1_ptr uint32_t* local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_local));
    for (uint32_t u = 0; u < local_users; u++) {
        uint32_t ids[topk];
        uint32_t weights[topk];
        const bool active = mimo_moe::user_active<expert_tiles, has_mask, rows>(stage, u);
        if (active || u == 0) {  // user 0's experts stand in when no user is active
            mimo_moe::select_user<topk, expert_tiles, norm, has_scale, scale_bits, has_mask, rows>(
                stage, u, ids, weights);
        } else {
            for (uint32_t i = 0; i < topk; i++) {
                ids[i] = 0;
                weights[i] = 0;
            }
        }
        local[u] = active ? 1 : 0;
        for (uint32_t i = 0; i < topk; i++) {
            local[local_users + u * topk + i] = ids[i];
            local[local_users + local_users * topk + u * topk + i] = weights[i];
            if (u == 0) {
                ids0[i] = ids[i];
            }
        }
        if (active) {
            n = mimo_moe::append_unique<topk>(uniq, n, ids);
        }
    }
    cb_push_back(cb_local, 1);
    MOE_MARK(12);

    // The writer's users: [active flag per remote user | ids]; taken when the stream reaches the end of the list.
    auto take_remote = [&]() {
        cb_wait_front(cb_sel, 1);
        volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sel));
        for (uint32_t r = 0; r < remote_users; r++) {
            if (sel[r] != 0) {
                uint32_t ids[topk];
                for (uint32_t i = 0; i < topk; i++) {
                    ids[i] = sel[remote_users + r * topk + i];
                }
                n = mimo_moe::append_unique<topk>(uniq, n, ids);
            }
        }
        cb_pop_front(cb_sel, 1);
        if (n == 0) {
            n = mimo_moe::append_unique<topk>(uniq, 0, ids0);
        }
        MOE_MARK(13);
    };

    // Stream: block i goes to CB slot i % num_buffers with trid (i % num_buffers) + 1; at most in_flight blocks
    // are issued but not pushed, and each reserve covers them plus the new block.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint64_t w_noc_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(w_noc_base, packet_bytes, vc);
    const uint32_t cb_base = get_write_ptr(cb_w);

    uint32_t total = n * blocks_per_pair;
    bool taken = false;
    uint32_t issued = 0;
    uint32_t oldest = 0;
    for (uint32_t i = 0;; i++) {
        if (i == total) {
            if (taken) {
                break;
            }
            take_remote();
            taken = true;
            total = n * blocks_per_pair;
            if (i == total) {
                break;
            }
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
        const uint32_t e = i / blocks_per_pair;
        uint32_t src = uniq[e] * expert_bytes + (i - e * blocks_per_pair) * block_bytes;
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
    MOE_MARK(14);

    // The down stream's expert stick: row-major uint16 [1, 1, 1, >= pairs + 1] = [count, ids...], one page in bank 0.
    if (bank_id == 0) {
        volatile tt_l1_ptr uint16_t* stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + Stage::stick);
        stick[0] = static_cast<uint16_t>(n);
        for (uint32_t j = 0; j < n; j++) {
            stick[1 + j] = static_cast<uint16_t>(uniq[j]);
        }
        noc_async_write(stage + Stage::stick, get_noc_addr_from_bank_id<true>(0, ids_addr), stick_bytes);
        noc_async_write_barrier();
    }
}
