// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1) of the deduplicated gate|up stream: x rows 0..rows-1 as [rows, 32] tiny tiles; routing of users
// [first_user, users), handed to the reader through cb_sel; with the reader's users (cb_local) the same list of
// distinct experts the reader streams, its count for compute (cb_cnt) and per distinct expert a routing-weight tile
// (cb_wt: row u = user u's weight of that expert, 0 when u did not select it or is masked); then every distinct
// expert's weighted h tile (all rows) into the zeroed row-major activation (page j = distinct expert j's kt
// [rows, 32] tiny tiles, the down stream's in0; tiny tile = this core's bank).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"
#include "moe_route_select.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(3);
    constexpr uint32_t topk = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles_per_pair = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(7);
    constexpr uint32_t first_user = get_compile_time_arg_val(8);
    constexpr uint32_t users = get_compile_time_arg_val(9);
    constexpr uint32_t stage_pages = get_compile_time_arg_val(10);
    constexpr uint32_t expert_tiles = get_compile_time_arg_val(11);
    constexpr bool norm = get_compile_time_arg_val(12) == 1;
    constexpr bool has_scale = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(14);
    constexpr bool has_mask = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t cb_local = get_compile_time_arg_val(16);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(17);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(18);
    constexpr uint32_t rows = get_compile_time_arg_val(19);  // the tiny tiles' rows: users <= rows
    constexpr uint32_t bf16_tile_bytes = 2048;
    constexpr uint32_t tiny_face_bytes = rows * mimo_moe::kFaceDim * 2;  // rows 0..rows-1 of one face
    constexpr uint32_t row_bytes = mimo_moe::kFaceDim * 2;               // one face row: 32 B
    constexpr uint32_t local_users = first_user;
    constexpr uint32_t sel_users = users - first_user;
    static_assert(users <= rows && rows <= mimo_moe::kFaceDim);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t col = get_arg_val<uint32_t>(2);
    const uint32_t biased_addr = get_arg_val<uint32_t>(3);
    const uint32_t scores_addr = get_arg_val<uint32_t>(4);
    const uint32_t mask_addr = get_arg_val<uint32_t>(5);

    MOE_MARK(20);
    const InterleavedAddrGen<true> x_gen = {.bank_base_address = x_addr, .page_size = bf16_tile_bytes};
    cb_reserve_back(cb_x, kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kt; t++) {
        const uint32_t dst = x_l1 + t * 2 * tiny_face_bytes;
        noc_async_read(x_gen.get_noc_addr(t, 0), dst, tiny_face_bytes);
        noc_async_read(x_gen.get_noc_addr(t, mimo_moe::kBf16FaceBytes), dst + tiny_face_bytes, tiny_face_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);
    MOE_MARK(21);

    uint32_t stage = 0;
    if constexpr (sel_users > 0) {
        cb_reserve_back(cb_stage, stage_pages);
        stage = get_write_ptr(cb_stage);
        mimo_moe::stage_route_rows<expert_tiles, has_mask, rows>(biased_addr, scores_addr, mask_addr, stage);
        noc_async_read_barrier();
    }
    MOE_MARK(22);

    // Every user's selection: ids, weight bits, active.
    uint32_t sel_ids[users][topk];
    uint32_t sel_w[users][topk];
    bool active[users];
    // [active flag per remote user | ids]
    cb_reserve_back(cb_sel, 1);
    volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sel));
    for (uint32_t u = first_user; u < users; u++) {
        active[u] = mimo_moe::user_active<expert_tiles, has_mask, rows>(stage, u);
        if (active[u]) {
            mimo_moe::select_user<topk, expert_tiles, norm, has_scale, scale_bits, has_mask, rows>(
                stage, u, sel_ids[u], sel_w[u]);
        } else {
            for (uint32_t i = 0; i < topk; i++) {
                sel_ids[u][i] = 0;
                sel_w[u][i] = 0;
            }
        }
        sel[u - first_user] = active[u] ? 1 : 0;
        for (uint32_t i = 0; i < topk; i++) {
            sel[sel_users + (u - first_user) * topk + i] = sel_ids[u][i];
        }
    }
    cb_push_back(cb_sel, 1);
    MOE_MARK(23);

    // The reader's users: [active flag per local user | ids | weight bits].
    cb_wait_front(cb_local, 1);
    volatile tt_l1_ptr uint32_t* local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_local));
    for (uint32_t u = 0; u < local_users; u++) {
        active[u] = local[u] != 0;
        for (uint32_t i = 0; i < topk; i++) {
            sel_ids[u][i] = local[local_users + u * topk + i];
            sel_w[u][i] = local[local_users + local_users * topk + u * topk + i];
        }
    }
    cb_pop_front(cb_local, 1);
    MOE_MARK(24);

    // The distinct experts, in the reader's order.
    uint32_t uniq[num_pairs];
    uint32_t n = 0;
    for (uint32_t u = 0; u < users; u++) {
        if (active[u]) {
            n = mimo_moe::append_unique<topk>(uniq, n, sel_ids[u]);
        }
    }
    if (n == 0) {
        n = mimo_moe::append_unique<topk>(uniq, 0, sel_ids[0]);
    }
    cb_reserve_back(cb_cnt, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt)) = n;
    cb_push_back(cb_cnt, 1);
    MOE_MARK(25);

    // Weight tiles (bf16 32 x 32; only rows 0..rows-1 are packed, the rest is never read): row u of faces 0 and 1.
    for (uint32_t j = 0; j < n; j++) {
        cb_reserve_back(cb_wt, 1);
        const uint32_t wt = get_write_ptr(cb_wt);
        for (uint32_t u = 0; u < rows; u++) {
            uint32_t bits = 0;
            if (u < users && active[u]) {
                for (uint32_t i = 0; i < topk; i++) {
                    if (sel_ids[u][i] == uniq[j]) {
                        bits = sel_w[u][i] >> 16;
                    }
                }
            }
            const uint32_t pair = (bits << 16) | bits;
            volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wt + u * row_bytes);
            volatile tt_l1_ptr uint32_t* f1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wt + mimo_moe::kBf16FaceBytes + u * row_bytes);
            for (uint32_t c = 0; c < mimo_moe::kFaceDim / 2; c++) {
                f0[c] = pair;
                f1[c] = pair;
            }
        }
        cb_push_back(cb_wt, 1);
    }
    MOE_MARK(26);

    constexpr uint32_t tiny_tile_bytes = 2 * tiny_face_bytes;
    const InterleavedAddrGen<false> out_gen = {
        .bank_base_address = out_addr, .page_size = out_tiles_per_pair * tiny_tile_bytes};
    const uint32_t col_base = col * tiny_tile_bytes;
    for (uint32_t j = 0; j < n; j++) {
        cb_wait_front(cb_h, 1);
        noc_async_write(get_read_ptr(cb_h), out_gen.get_noc_addr(j, col_base), tiny_tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_h, 1);
        if (j == 0) {
            MOE_MARK(27);
        }
    }
    MOE_MARK(28);
}
