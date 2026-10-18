// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Data-movement cores building the expert-major prefill plan from the routing (DevicePlan in tt/moe_prefill.py), the
// same plan PrefillExperts.plan / GroupedExperts.plan build on the host.
//
// The T x K top-k expert ids (uint32 tiles [T, 32], columns 0..K-1 valid) are counted per expert; expert e gets
// ceil(count / 32) groups of 32 packed rows, experts in order. Pair (t, k), taken in (t, k) order, goes to the next
// free row of its expert's groups: row_token[row] = t, row_weight[row] = its routing weight (bf16, round to nearest
// even), row_dest[row] = k * T + t (the pair's row of the k-major expert output); rows nothing lands in keep token 0,
// weight 0 and row_dest 0xFFFFFFFF.
//
// Work is split by expert, so no core waits for another: every core counts all pairs (and so knows every expert's
// first group), then fills the rows of its own experts [core * E / cores, (core + 1) * E / cores), which are one
// contiguous range of packed rows, and writes that range of the three row arrays (the last core also the unused rows
// up to rows_max). Core 0 also deals out the team plan: every expert's groups in chunks of at most ceil(G / teams),
// longest first (then higher expert, then later chunk) to the least loaded team (ties: lower team), a chunk of n
// groups adding 2n + 1 to its team's load. Plan page t: [jobs, (expert, first group, groups) per job].
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t teams = get_compile_time_arg_val(3);
    constexpr uint32_t plan_words = get_compile_time_arg_val(4);
    constexpr uint32_t rows_max = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(6);    // 2 x 4 KB: an ids tile and a weights tile
    constexpr uint32_t cb_state = get_compile_time_arg_val(7);    // count, cursor, first group: 3 x E + 1 words
    constexpr uint32_t cb_tokens = get_compile_time_arg_val(8);   // rows_max words
    constexpr uint32_t cb_weights = get_compile_time_arg_val(9);  // rows_max bf16
    constexpr uint32_t cb_dest = get_compile_time_arg_val(10);    // rows_max words
    constexpr uint32_t cb_plan = get_compile_time_arg_val(11);    // teams * plan_words words (core 0)
    constexpr uint32_t cores = get_compile_time_arg_val(12);
    constexpr uint32_t tile_bytes = 4096;  // uint32 / fp32 32 x 32
    constexpr uint32_t tile_rows = T / 32;

    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(2);
    const uint32_t weights_addr = get_arg_val<uint32_t>(3);
    const uint32_t dest_addr = get_arg_val<uint32_t>(4);
    const uint32_t plan_addr = get_arg_val<uint32_t>(5);
    const uint32_t core = get_arg_val<uint32_t>(6);

    const uint32_t ids_l1 = get_write_ptr(cb_tiles);
    const uint32_t w_l1 = ids_l1 + tile_bytes;
    tt_l1_ptr uint32_t* ids = reinterpret_cast<tt_l1_ptr uint32_t*>(ids_l1);
    tt_l1_ptr uint32_t* wbits = reinterpret_cast<tt_l1_ptr uint32_t*>(w_l1);
    tt_l1_ptr uint32_t* count = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_state));
    tt_l1_ptr uint32_t* cursor = count + E;
    tt_l1_ptr uint32_t* first_group = cursor + E;  // E + 1 entries
    tt_l1_ptr uint32_t* row_token = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_tokens));
    tt_l1_ptr uint32_t* row_weight2 = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_weights));
    tt_l1_ptr uint16_t* row_weight = reinterpret_cast<tt_l1_ptr uint16_t*>(get_write_ptr(cb_weights));
    tt_l1_ptr uint32_t* row_dest = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_dest));

    // Element (r, c), c < 16, of a 32 x 32 tile of 4-byte values: face (r / 16) * 2, row r % 16 of it.
    auto at = [](uint32_t r, uint32_t c) { return (r >> 4) * 512 + (r & 15) * 16 + c; };
    const InterleavedAddrGen<true> ids_gen = {.bank_base_address = ids_addr, .page_size = tile_bytes};
    const InterleavedAddrGen<true> w_gen = {.bank_base_address = w_addr, .page_size = tile_bytes};

    for (uint32_t e = 0; e < E; e++) {
        count[e] = 0;
    }
    for (uint32_t tr = 0; tr < tile_rows; tr++) {
        noc_async_read(ids_gen.get_noc_addr(tr, 0), ids_l1, tile_bytes);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 32; r++) {
            for (uint32_t k = 0; k < K; k++) {
                count[ids[at(r, k)]]++;
            }
        }
    }
    uint32_t groups = 0;
    for (uint32_t e = 0; e < E; e++) {
        first_group[e] = groups;
        cursor[e] = groups * 32;
        groups += (count[e] + 31) >> 5;
    }
    first_group[E] = groups;

    // This core's experts and rows (the last core also takes the unused rows up to rows_max).
    const uint32_t e0 = core * E / cores, e1 = (core + 1) * E / cores;
    const uint32_t row0 = first_group[e0] * 32;
    const uint32_t row1 = core == cores - 1 ? rows_max : first_group[e1] * 32;
    for (uint32_t r = 0; r < row1 - row0; r++) {
        row_token[r] = 0;
        row_dest[r] = 0xFFFFFFFF;
    }
    for (uint32_t r = 0; r < (row1 - row0) / 2; r++) {
        row_weight2[r] = 0;
    }
    if (first_group[e1] > first_group[e0]) {
        for (uint32_t tr = 0; tr < tile_rows; tr++) {
            noc_async_read(ids_gen.get_noc_addr(tr, 0), ids_l1, tile_bytes);
            noc_async_read(w_gen.get_noc_addr(tr, 0), w_l1, tile_bytes);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t t = tr * 32 + r;
                for (uint32_t k = 0; k < K; k++) {
                    const uint32_t i = at(r, k);
                    const uint32_t e = ids[i];
                    if (e - e0 < e1 - e0) {
                        const uint32_t row = cursor[e]++ - row0;
                        const uint32_t b = wbits[i];
                        row_token[row] = t;
                        row_weight[row] = static_cast<uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
                        row_dest[row] = k * T + t;
                    }
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
        tt_l1_ptr uint32_t* plan = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
        for (uint32_t i = 0; i < teams * plan_words; i++) {
            plan[i] = 0;
        }
        const uint32_t cap = groups > teams ? (groups + teams - 1) / teams : 1;
        uint32_t load[teams];
        for (uint32_t t = 0; t < teams; t++) {
            load[t] = 0;
        }
        for (uint32_t n = cap; n > 0; n--) {
            for (uint32_t e = E; e-- > 0;) {
                const uint32_t eg = (count[e] + 31) >> 5;
                const uint32_t full = eg / cap;
                const uint32_t rest = eg - full * cap;
                uint32_t chunks = 0, last = 0;  // chunks of size n of expert e, the last (highest) chunk offset
                if (n == cap) {
                    chunks = full;
                    last = full > 0 ? (full - 1) * cap : 0;
                }
                if (rest == n && rest != cap) {
                    chunks = 1;
                    last = full * cap;
                }
                for (uint32_t c = 0; c < chunks; c++) {
                    const uint32_t off = last - c * cap;
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
                }
            }
        }
        const InterleavedAddrGen<true> plan_gen = {.bank_base_address = plan_addr, .page_size = plan_words * 4};
        for (uint32_t t = 0; t < teams; t++) {
            noc_async_write(get_write_ptr(cb_plan) + t * plan_words * 4, plan_gen.get_noc_addr(t, 0), plan_words * 4);
        }
    }
    noc_async_write_barrier();
}
