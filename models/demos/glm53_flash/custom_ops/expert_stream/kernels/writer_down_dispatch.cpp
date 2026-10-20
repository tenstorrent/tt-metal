// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation loader, token scatter and output writer (BRISC, NOC1) of the per-expert dispatch down op
// (custom_ops/expert_stream/op.py, ExpertDownDispatch). This core owns `cols` output tile columns (col0 ..) of
// all T token rows.
//
// Phase 1 (per-expert pass counts through a metadata page): loads every pass's activation tiles act[k] ([32, I] =
// seg_tiles BF16 tiles, batches of three passes, one batch in flight) for the compute, which packs h_k = act[k] @
// W2[e][:, cols] (BF16) into the resident h buffer (cb_h, never popped). Phase 2 (per token row tile r): for every pass
// k and each of its rows i with token t in the tile, copies the row h_k[i] into contribution slot j = (number of
// earlier contributions of t) of the gather tiles G[j] (<= J per token), tells the compute how many slots are used
// (mailbox) and writes its summed output tiles out[r, col0 ..]. The pass rows come from the table the gate|up op wrote
// (uint16 rows [n, e, t_0 ..]).
//
// runtime args: [counts_addr (EP: idx_addr), table_addr, act_addr, out_addr, col0, (EP) base_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "dispatch_common.hpp"

void kernel_main() {
    const uint32_t counts_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t col0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_c = get_compile_time_arg_val(0);
    constexpr uint32_t cb_t = get_compile_time_arg_val(1);
    constexpr uint32_t cb_m = get_compile_time_arg_val(15);  // metadata page: [n_active, passes per active expert]
    constexpr uint32_t EP = get_compile_time_arg_val(16);    // 1: counts from the top-k index list + expert base
    constexpr uint32_t K = get_compile_time_arg_val(17);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h = get_compile_time_arg_val(4);
    constexpr uint32_t cb_g = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t E = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t cols = get_compile_time_arg_val(10);
    constexpr uint32_t J = get_compile_time_arg_val(11);
    constexpr uint32_t P_MAX = get_compile_time_arg_val(12);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(13);  // output tile columns of the whole row (H / 32)
    constexpr uint32_t table_bytes = get_compile_time_arg_val(14);
    constexpr auto counts_args = TensorAccessorArgs<18>();  // EP: the index tiles
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t RT = T / 32;
    constexpr uint32_t tile_page = 2048;
    constexpr uint32_t slot_tiles = J * cols;
    constexpr uint32_t table_elems = table_bytes / 2;

    const auto s_c = TensorAccessor(counts_args, counts_addr, tile_page);
    const auto s_tab = TensorAccessor(table_args, table_addr, table_bytes);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_page);

    // counts -> passes
    constexpr uint32_t c_tiles = EP ? RT : E_TILES;
    cb_reserve_back(cb_c, c_tiles);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    for (uint32_t i = 0; i < c_tiles; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * tile_page, tile_page);
    }
    uint16_t n_tok[E], passes[E], base[E];
    if constexpr (EP) {
        const uint32_t base_addr = get_arg_val<uint32_t>(5);
        const auto s_b = TensorAccessor(base_args, base_addr, 32);
        cb_reserve_back(cb_c, 1);
        const uint32_t b_l1 = c_l1 + c_tiles * tile_page;
        noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
        noc_async_read_barrier();
        const uint32_t expert_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
        count_from_indices<E, T, K>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), expert_base, n_tok);
    } else {
        noc_async_read_barrier();
        read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    }
    uint32_t n_pass = pass_bases<E>(n_tok, passes, base);
    if (n_pass > P_MAX) {
        n_pass = P_MAX;  // never with <= 8 routed experts per token (op.py's P_MAX); keeps the kernel from hanging
    }

    // the pass table
    cb_reserve_back(cb_t, P_MAX);
    const uint32_t t_l1 = get_write_ptr(cb_t);
    for (uint32_t k = 0; k < n_pass; ++k) {
        noc_async_read(s_tab.get_noc_addr(k), t_l1 + k * table_bytes, table_bytes);
    }
    // the identity tile (BF16 1.0 = 0x3F80 on the diagonal) for the gather sum
    cb_reserve_back(cb_id, 1);
    const uint32_t id_l1 = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* id32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(id_l1);
    for (uint32_t j = 0; j < tile_page / 4; ++j) {
        id32[j] = 0;
    }
    volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(id_l1);
    for (uint32_t t = 0; t < 32; ++t) {
        id16[tile_elem(t, t)] = 0x3F80;
    }
    cb_push_back(cb_id, 1);
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint16_t* tab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t_l1);

    // ---- phase 1: the per-expert pass counts for the compute (metadata page), then every pass's activations
    cb_reserve_back(cb_m, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_m));
    uint32_t n_active = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (passes[e] != 0) {
            meta[1 + n_active++] = passes[e];
        }
    }
    meta[0] = n_active;
    cb_push_back(cb_m, 1);
    // batches of a_batch passes (one read barrier each), two batches of slots: one batch lands while the compute
    // works on the other
    constexpr uint32_t a_batch = 1;  // one pass per read barrier (batches of 3 were slower end to end)
    for (uint32_t k = 0; k < n_pass; k += a_batch) {
        const uint32_t nb = (n_pass - k) < a_batch ? (n_pass - k) : a_batch;
        cb_reserve_back(cb_a, nb * seg_tiles);
        const uint32_t a_l1 = get_write_ptr(cb_a);
        for (uint32_t s = 0; s < nb * seg_tiles; ++s) {
            noc_async_read(s_act.get_noc_addr(k * seg_tiles + s), a_l1 + s * tile_page, tile_page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, nb * seg_tiles);
    }

    // ---- phase 2: gather the token rows of every pass's h_k into the contribution slots, row tile by row tile
    uint32_t h_base = 0;
    if (n_pass > 0) {
        cb_wait_front(cb_h, n_pass * cols);
        h_base = get_read_ptr(cb_h);
    }
    uint8_t cursor[P_MAX];
    for (uint32_t k = 0; k < n_pass; ++k) {
        cursor[k] = 0;
    }
    // per gather slot (two alternate): the rows written last time, re-zeroed before reuse (whole slot the first time)
    uint16_t written[2][J * 32];
    uint32_t n_written[2] = {0, 0};
    bool fresh[2] = {true, true};
    for (uint32_t r = 0; r < RT; ++r) {
        cb_reserve_back(cb_g, slot_tiles);
        const uint32_t g_l1 = get_write_ptr(cb_g);
        const uint32_t slot = r & 1;
        if (fresh[slot]) {
            volatile tt_l1_ptr uint32_t* g32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_l1);
            for (uint32_t j = 0; j < slot_tiles * tile_page / 4; ++j) {
                g32[j] = 0;
            }
            fresh[slot] = false;
        } else {
            for (uint32_t w = 0; w < n_written[slot]; ++w) {
                const uint32_t j = written[slot][w] >> 5;
                const uint32_t tl = written[slot][w] & 31;
                for (uint32_t c = 0; c < cols; ++c) {
                    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        g_l1 + (j * cols + c) * tile_page + tile_row_bytes(tl));
                    for (uint32_t q = 0; q < 8; ++q) {
                        d[q] = 0;
                        d[128 + q] = 0;
                    }
                }
            }
        }
        n_written[slot] = 0;
        uint8_t cnt[32];
        for (uint32_t t = 0; t < 32; ++t) {
            cnt[t] = 0;
        }
        const uint32_t t_end = (r + 1) * 32;
        uint32_t j_max = 0;
        for (uint32_t k = 0; k < n_pass; ++k) {
            const volatile tt_l1_ptr uint16_t* row = tab + k * table_elems;
            const uint32_t nr = row[0];
            uint32_t i = cursor[k];
            while (i < nr && row[2 + i] < t_end) {
                const uint32_t tl = row[2 + i] - r * 32;
                const uint32_t j = cnt[tl]++;
                if (j < J) {
                    if (j + 1 > j_max) {
                        j_max = j + 1;
                    }
                    written[slot][n_written[slot]++] = static_cast<uint16_t>((j << 5) | tl);
                    for (uint32_t c = 0; c < cols; ++c) {
                        const volatile tt_l1_ptr uint32_t* s = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(
                            h_base + (k * cols + c) * tile_page + tile_row_bytes(i));
                        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            g_l1 + (j * cols + c) * tile_page + tile_row_bytes(tl));
                        for (uint32_t q = 0; q < 8; ++q) {
                            d[q] = s[q];
                            d[128 + q] = s[128 + q];
                        }
                    }
                }
                ++i;
            }
            cursor[k] = static_cast<uint8_t>(i);
        }
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, j_max);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, j_max);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, j_max);
        cb_push_back(cb_g, slot_tiles);

        cb_wait_front(cb_out, cols);
        const uint32_t o_l1 = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(o_l1 + c * tile_page, s_out.get_noc_addr(r * h_tiles + col0 + c), tile_page);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, cols);
    }
}
