// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Second half of the split expert-major plan (DevicePlan in tt/moe_prefill.py), after plan_slice.cpp; the result is
// the one plan.cpp builds (same rows, same team plan).
//
// Every core reads all slices' per-expert counts, so it knows every expert's count and first group. Core i owns the
// rows of experts [i * E / cores, (i + 1) * E / cores), one contiguous range of packed rows (the last core also the
// unused rows up to rows_max). From each slice it reads only the records of its experts (contiguous: a slice's
// records are sorted by expert) and puts them in its rows in slice order, so an expert's rows stay in (t, k) order.
// Rows nothing lands in keep token 0, weight 0 and row_dest 0xFFFFFFFF. Core 0 also deals out the team plan, as
// plan.cpp does.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t teams = get_compile_time_arg_val(1);
    constexpr uint32_t plan_words = get_compile_time_arg_val(2);
    constexpr uint32_t rows_max = get_compile_time_arg_val(3);
    constexpr uint32_t cores = get_compile_time_arg_val(4);
    constexpr uint32_t slice_pairs = get_compile_time_arg_val(5);
    constexpr uint32_t cb_counts = get_compile_time_arg_val(6);    // cores x E words
    constexpr uint32_t cb_state = get_compile_time_arg_val(7);     // count, cursor, first group: 3 x E + 1 words
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(8);       // one slice's records: tokens, dest, weights
    constexpr uint32_t cb_tokens = get_compile_time_arg_val(9);    // rows_max words
    constexpr uint32_t cb_weights = get_compile_time_arg_val(10);  // rows_max bf16
    constexpr uint32_t cb_dest = get_compile_time_arg_val(11);     // rows_max words
    constexpr uint32_t cb_plan = get_compile_time_arg_val(12);     // teams * plan_words words (core 0)
    constexpr uint32_t align = 64;                                 // DRAM read alignment
    constexpr uint32_t tmp_words = slice_pairs + align / 4;        // one record array, read from an aligned start

    const uint32_t counts_addr = get_arg_val<uint32_t>(0);
    const uint32_t rtok_addr = get_arg_val<uint32_t>(1);
    const uint32_t rw_addr = get_arg_val<uint32_t>(2);
    const uint32_t rdest_addr = get_arg_val<uint32_t>(3);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(4);
    const uint32_t weights_addr = get_arg_val<uint32_t>(5);
    const uint32_t dest_addr = get_arg_val<uint32_t>(6);
    const uint32_t plan_addr = get_arg_val<uint32_t>(7);
    const uint32_t core = get_arg_val<uint32_t>(8);

    tt_l1_ptr uint32_t* cnt = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_counts));
    const InterleavedAddrGen<true> counts_gen = {.bank_base_address = counts_addr, .page_size = E * 4};
    for (uint32_t c = 0; c < cores; c++) {
        noc_async_read(counts_gen.get_noc_addr(c, 0), get_write_ptr(cb_counts) + c * E * 4, E * 4);
    }
    noc_async_read_barrier();

    tt_l1_ptr uint32_t* count = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_state));
    tt_l1_ptr uint32_t* cursor = count + E;
    tt_l1_ptr uint32_t* first_group = cursor + E;  // E + 1 entries
    uint32_t groups = 0;
    for (uint32_t e = 0; e < E; e++) {
        uint32_t n = 0;
        for (uint32_t c = 0; c < cores; c++) {
            n += cnt[c * E + e];
        }
        count[e] = n;
        first_group[e] = groups;
        cursor[e] = groups * 32;
        groups += (n + 31) >> 5;
    }
    first_group[E] = groups;

    const uint32_t e0 = core * E / cores, e1 = (core + 1) * E / cores;
    const uint32_t row0 = first_group[e0] * 32;
    const uint32_t row1 = core == cores - 1 ? rows_max : first_group[e1] * 32;
    tt_l1_ptr uint32_t* row_token = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_tokens));
    tt_l1_ptr uint32_t* row_weight2 = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_weights));
    tt_l1_ptr uint16_t* row_weight = reinterpret_cast<tt_l1_ptr uint16_t*>(get_write_ptr(cb_weights));
    tt_l1_ptr uint32_t* row_dest = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_dest));
    for (uint32_t r = 0; r < row1 - row0; r++) {
        row_token[r] = 0;
        row_dest[r] = 0xFFFFFFFF;
    }
    for (uint32_t r = 0; r < (row1 - row0) / 2; r++) {
        row_weight2[r] = 0;
    }

    if (first_group[e1] > first_group[e0]) {
        const InterleavedAddrGen<true> rtok_gen = {.bank_base_address = rtok_addr, .page_size = slice_pairs * 4};
        const InterleavedAddrGen<true> rw_gen = {.bank_base_address = rw_addr, .page_size = slice_pairs * 2};
        const InterleavedAddrGen<true> rdest_gen = {.bank_base_address = rdest_addr, .page_size = slice_pairs * 4};
        const uint32_t tmp = get_write_ptr(cb_tmp);
        const uint32_t tmp_tok = tmp, tmp_dest = tmp + tmp_words * 4, tmp_w = tmp + 2 * tmp_words * 4;
        for (uint32_t c = 0; c < cores; c++) {
            uint32_t off = 0, len = 0;
            for (uint32_t e = 0; e < e0; e++) {
                off += cnt[c * E + e];
            }
            for (uint32_t e = e0; e < e1; e++) {
                len += cnt[c * E + e];
            }
            if (len == 0) {
                continue;
            }
            // Byte ranges of the records [off, off + len) of each array, widened to the DRAM read alignment.
            const uint32_t w4 = (off * 4) & ~(align - 1), n4 = ((off + len) * 4 + align - 1) / align * align - w4;
            const uint32_t w2 = (off * 2) & ~(align - 1), n2 = ((off + len) * 2 + align - 1) / align * align - w2;
            noc_async_read(rtok_gen.get_noc_addr(c, w4), tmp_tok, n4);
            noc_async_read(rdest_gen.get_noc_addr(c, w4), tmp_dest, n4);
            noc_async_read(rw_gen.get_noc_addr(c, w2), tmp_w, n2);
            noc_async_read_barrier();
            tt_l1_ptr uint32_t* tok = reinterpret_cast<tt_l1_ptr uint32_t*>(tmp_tok) + (off - w4 / 4);
            tt_l1_ptr uint32_t* dest = reinterpret_cast<tt_l1_ptr uint32_t*>(tmp_dest) + (off - w4 / 4);
            tt_l1_ptr uint16_t* w = reinterpret_cast<tt_l1_ptr uint16_t*>(tmp_w) + (off - w2 / 2);
            uint32_t j = 0;
            for (uint32_t e = e0; e < e1; e++) {
                for (uint32_t n = cnt[c * E + e]; n > 0; n--, j++) {
                    const uint32_t row = cursor[e]++ - row0;
                    row_token[row] = tok[j];
                    row_weight[row] = w[j];
                    row_dest[row] = dest[j];
                }
            }
        }
    }
    if (row1 > row0) {
        const InterleavedAddrGen<true> tokens_gen = {.bank_base_address = tokens_addr, .page_size = rows_max * 4};
        const InterleavedAddrGen<true> weights_gen = {.bank_base_address = weights_addr, .page_size = rows_max * 2};
        const InterleavedAddrGen<true> dest_gen = {.bank_base_address = dest_addr, .page_size = rows_max * 4};
        noc_async_write(get_write_ptr(cb_tokens), tokens_gen.get_noc_addr(0, row0 * 4), (row1 - row0) * 4);
        noc_async_write(get_write_ptr(cb_weights), weights_gen.get_noc_addr(0, row0 * 2), (row1 - row0) * 2);
        noc_async_write(get_write_ptr(cb_dest), dest_gen.get_noc_addr(0, row0 * 4), (row1 - row0) * 4);
    }

    if (core == 0) {
        // The team plan, exactly as plan.cpp deals it out.
        tt_l1_ptr uint32_t* plan = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
        for (uint32_t i = 0; i < teams * plan_words; i++) {
            plan[i] = 0;
        }
        const uint32_t cap = groups > teams ? (groups + teams - 1) / teams : 1;
        uint32_t load[teams];
        for (uint32_t t = 0; t < teams; t++) {
            load[t] = 0;
        }
        auto deal = [&](uint32_t e, uint32_t off, uint32_t n) {
            uint32_t best = 0;
            for (uint32_t t = 1; t < teams; t++) {
                if (load[t] < load[best]) {
                    best = t;
                }
            }
            tt_l1_ptr uint32_t* page = plan + best * plan_words;
            const uint32_t j = page[0]++;
            page[1 + 3 * j] = e;
            page[2 + 3 * j] = first_group[e] + off;
            page[3 + 3 * j] = n;
            load[best] += 2 * n + 1;
        };
        // plan.cpp's order without its cap x E scan: first every expert's whole chunks of cap groups (experts from
        // the last, chunks from the last), then for n = cap - 1 .. 1 the remainder chunk of every expert whose
        // remainder is n (experts from the last). The slice counts are done with: cnt holds full chunks per expert,
        // the next expert of the same remainder, and the first expert of each remainder.
        tt_l1_ptr uint32_t* full = cnt;
        tt_l1_ptr uint32_t* next = cnt + E;
        tt_l1_ptr uint32_t* head = cnt + 2 * E;  // cap entries
        for (uint32_t n = 0; n < cap; n++) {
            head[n] = E;  // none
        }
        for (uint32_t e = 0; e < E; e++) {
            const uint32_t eg = (count[e] + 31) >> 5;
            const uint32_t f = eg / cap;
            const uint32_t rest = eg - f * cap;
            full[e] = f;
            if (rest != 0) {
                next[e] = head[rest];
                head[rest] = e;  // pushed in increasing e: the list runs from the last expert
            }
        }
        for (uint32_t e = E; e-- > 0;) {
            for (uint32_t c = full[e]; c-- > 0;) {
                deal(e, c * cap, cap);
            }
        }
        for (uint32_t n = cap - 1; n > 0; n--) {
            for (uint32_t e = head[n]; e != E; e = next[e]) {
                deal(e, full[e] * cap, n);
            }
        }
        const InterleavedAddrGen<true> plan_gen = {.bank_base_address = plan_addr, .page_size = plan_words * 4};
        for (uint32_t t = 0; t < teams; t++) {
            noc_async_write(get_write_ptr(cb_plan) + t * plan_words * 4, plan_gen.get_noc_addr(t, 0), plan_words * 4);
        }
    }
    noc_async_write_barrier();
}
