// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// First half of the split expert-major plan (DevicePlan in tt/moe_prefill.py; plan_gather.cpp is the second half).
//
// Core i takes tile rows [i * TR / cores, (i + 1) * TR / cores) of the T x K top-k (TR = T / 32), counts its pairs
// per expert and writes them sorted by expert, in (t, k) order within an expert, as three records per pair: the
// token t, the routing weight (bf16, round to nearest even) and the destination row k * T + t. Page i of
// ``counts`` gets the slice's per-expert counts (E words), page i of ``rec_*`` its records.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t cores = get_compile_time_arg_val(3);
    constexpr uint32_t slice_pairs = get_compile_time_arg_val(4);  // records per page: ceil(TR / cores) * 32 * K
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(5);     // ids then weights tiles of the slice
    constexpr uint32_t cb_state = get_compile_time_arg_val(6);     // count, cursor: 2 x E words
    constexpr uint32_t cb_rec = get_compile_time_arg_val(7);       // tokens, dest (words), weights (bf16)
    constexpr uint32_t tile_bytes = 4096;                          // uint32 / fp32 32 x 32
    constexpr uint32_t TR = T / 32;

    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t rtok_addr = get_arg_val<uint32_t>(3);
    const uint32_t rw_addr = get_arg_val<uint32_t>(4);
    const uint32_t rdest_addr = get_arg_val<uint32_t>(5);
    const uint32_t core = get_arg_val<uint32_t>(6);

    const uint32_t tr0 = core * TR / cores, tr1 = (core + 1) * TR / cores;
    const uint32_t n_rows = tr1 - tr0;
    const uint32_t ids_l1 = get_write_ptr(cb_tiles);
    const uint32_t w_l1 = ids_l1 + n_rows * tile_bytes;
    const InterleavedAddrGen<true> ids_gen = {.bank_base_address = ids_addr, .page_size = tile_bytes};
    const InterleavedAddrGen<true> w_gen = {.bank_base_address = w_addr, .page_size = tile_bytes};
    for (uint32_t r = 0; r < n_rows; r++) {
        noc_async_read(ids_gen.get_noc_addr(tr0 + r, 0), ids_l1 + r * tile_bytes, tile_bytes);
        noc_async_read(w_gen.get_noc_addr(tr0 + r, 0), w_l1 + r * tile_bytes, tile_bytes);
    }
    noc_async_read_barrier();

    tt_l1_ptr uint32_t* count = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_state));
    tt_l1_ptr uint32_t* cursor = count + E;
    const uint32_t rec_l1 = get_write_ptr(cb_rec);
    tt_l1_ptr uint32_t* rtok = reinterpret_cast<tt_l1_ptr uint32_t*>(rec_l1);
    tt_l1_ptr uint32_t* rdest = rtok + slice_pairs;
    tt_l1_ptr uint16_t* rw = reinterpret_cast<tt_l1_ptr uint16_t*>(rdest + slice_pairs);

    // Element (r, c), c < 16, of a 32 x 32 tile of 4-byte values: face (r / 16) * 2, row r % 16 of it.
    auto at = [](uint32_t r, uint32_t c) { return (r >> 4) * 512 + (r & 15) * 16 + c; };
    for (uint32_t e = 0; e < E; e++) {
        count[e] = 0;
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        tt_l1_ptr uint32_t* ids = reinterpret_cast<tt_l1_ptr uint32_t*>(ids_l1 + r * tile_bytes);
        for (uint32_t row = 0; row < 32; row++) {
            for (uint32_t k = 0; k < K; k++) {
                count[ids[at(row, k)]]++;
            }
        }
    }
    uint32_t pos = 0;
    for (uint32_t e = 0; e < E; e++) {
        cursor[e] = pos;
        pos += count[e];
    }
    for (uint32_t r = 0; r < n_rows; r++) {
        tt_l1_ptr uint32_t* ids = reinterpret_cast<tt_l1_ptr uint32_t*>(ids_l1 + r * tile_bytes);
        tt_l1_ptr uint32_t* wbits = reinterpret_cast<tt_l1_ptr uint32_t*>(w_l1 + r * tile_bytes);
        for (uint32_t row = 0; row < 32; row++) {
            const uint32_t t = (tr0 + r) * 32 + row;
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t i = at(row, k);
                const uint32_t j = cursor[ids[i]]++;
                const uint32_t b = wbits[i];
                rtok[j] = t;
                rw[j] = static_cast<uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
                rdest[j] = k * T + t;
            }
        }
    }

    const InterleavedAddrGen<true> counts_gen = {.bank_base_address = counts_addr, .page_size = E * 4};
    const InterleavedAddrGen<true> rtok_gen = {.bank_base_address = rtok_addr, .page_size = slice_pairs * 4};
    const InterleavedAddrGen<true> rw_gen = {.bank_base_address = rw_addr, .page_size = slice_pairs * 2};
    const InterleavedAddrGen<true> rdest_gen = {.bank_base_address = rdest_addr, .page_size = slice_pairs * 4};
    noc_async_write(get_write_ptr(cb_state), counts_gen.get_noc_addr(core, 0), E * 4);
    if (pos > 0) {
        noc_async_write(rec_l1, rtok_gen.get_noc_addr(core, 0), slice_pairs * 4);
        noc_async_write(reinterpret_cast<uint32_t>(rdest), rdest_gen.get_noc_addr(core, 0), slice_pairs * 4);
        noc_async_write(reinterpret_cast<uint32_t>(rw), rw_gen.get_noc_addr(core, 0), slice_pairs * 2);
    }
    noc_async_write_barrier();
}
