// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Greedy decode head, step 2 (tt/greedy_head.py): merge every chip's per-core maxima into each user's token.
//
// Input: the row all-gather of the scan partials, [1, 1, 32, 32 * tp] uint32 TILE (interleaved): tile `chip` holds
// chip `chip`'s partials, row `core` columns 2u, 2u+1 = (key, local index) of user u (users 8-15 in the row's right
// face: faces 1 and 3). Cores cover increasing index
// ranges and chips increasing vocab shards, so taking the first strict maximum in (chip, core) order keeps the lowest
// global id on ties, like torch.argmax.
// Output: global token ids (chip * vocab_local + local index) of users 0..users-1 and zeros for the other slots,
// written as one uint32 [1, max_batch] row-major page to each of up to two tensors (the persistent next-token
// buffer and the decode trace's token input).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kTileBytesU32 = 4096;
constexpr uint32_t kFaceBytesU32 = 1024;
}  // namespace

void kernel_main() {
    const uint32_t gathered_addr = get_arg_val<uint32_t>(0);
    const uint32_t gathered_is_dram = get_arg_val<uint32_t>(1);
    const uint32_t out0_addr = get_arg_val<uint32_t>(2);
    const uint32_t out0_is_dram = get_arg_val<uint32_t>(3);
    const uint32_t out1_addr = get_arg_val<uint32_t>(4);  // 0: none
    const uint32_t out1_is_dram = get_arg_val<uint32_t>(5);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t num_cores = get_compile_time_arg_val(1);
    constexpr uint32_t tp = get_compile_time_arg_val(2);
    constexpr uint32_t vocab_local = get_compile_time_arg_val(3);
    constexpr uint32_t max_batch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(5);
    static_assert(num_cores <= 32 && users <= 16 && max_batch * 4 <= 1024);
    constexpr uint32_t halves = users > 8 ? 2 : 1;  // left faces (0, 2), then with more than 8 users the right (1, 3)
    constexpr uint32_t chip_bytes = halves * 2 * kFaceBytesU32;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // tp x (face 0 | face 2 [| face 1 | face 3]), the token row
    const uint32_t tokens_l1 = scratch + tp * chip_bytes;

    for (uint32_t ch = 0; ch < tp; ++ch) {
        uint64_t tile;
        if (gathered_is_dram) {
            const InterleavedAddrGen<true> g = {.bank_base_address = gathered_addr, .page_size = kTileBytesU32};
            tile = get_noc_addr(ch, g);
        } else {
            const InterleavedAddrGen<false> g = {.bank_base_address = gathered_addr, .page_size = kTileBytesU32};
            tile = get_noc_addr(ch, g);
        }
        const uint32_t dst = scratch + ch * chip_bytes;
        noc_async_read(tile, dst, kFaceBytesU32);  // rows 0-15, columns 0-15
        noc_async_read(tile + 2 * kFaceBytesU32, dst + kFaceBytesU32, kFaceBytesU32);
        if constexpr (halves == 2) {
            noc_async_read(tile + kFaceBytesU32, dst + 2 * kFaceBytesU32, kFaceBytesU32);  // columns 16-31
            noc_async_read(tile + 3 * kFaceBytesU32, dst + 3 * kFaceBytesU32, kFaceBytesU32);
        }
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tokens_l1);
    for (uint32_t u = 0; u < max_batch; ++u) {
        tokens[u] = 0;
    }
    // One pass over the (chip, core) rows in order, every user's running maximum at once (strict: the first maximum
    // wins, the lowest global id on ties).
    uint32_t best_key[users];
    uint32_t best_id[users];
    bool found[users];
    for (uint32_t u = 0; u < users; ++u) {
        best_key[u] = 0;
        best_id[u] = 0;
        found[u] = false;
    }
    for (uint32_t ch = 0; ch < tp; ++ch) {
        for (uint32_t c = 0; c < num_cores; ++c) {
            // row c lives in face 0 (c < 16) or face 2 (copied right after face 0): 16 uint32 per row; users 8-15 in
            // the same row of face 1 or 3, 2 KB further
            const uint32_t at = scratch + ch * chip_bytes + (c < 16 ? 0 : kFaceBytesU32) + (c % 16) * 64;
            const volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at);
            uint32_t vals[2 * users];
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 2 * users; ++i) {
                vals[i] = i < 16 ? row[i] : row[2 * kFaceBytesU32 / 4 + i - 16];
            }
#pragma GCC unroll 8
            for (uint32_t u = 0; u < users; ++u) {
                const uint32_t key = vals[u * 2];
                if (!found[u] || key > best_key[u]) {
                    found[u] = true;
                    best_key[u] = key;
                    best_id[u] = ch * vocab_local + vals[u * 2 + 1];
                }
            }
        }
    }
    for (uint32_t u = 0; u < users; ++u) {
        tokens[u] = best_id[u];
    }

    const uint32_t bytes = max_batch * 4;
    if (out0_is_dram) {
        const InterleavedAddrGen<true> g = {.bank_base_address = out0_addr, .page_size = bytes};
        noc_async_write(tokens_l1, get_noc_addr(0, g), bytes);
    } else {
        const InterleavedAddrGen<false> g = {.bank_base_address = out0_addr, .page_size = bytes};
        noc_async_write(tokens_l1, get_noc_addr(0, g), bytes);
    }
    if (out1_addr != 0) {
        if (out1_is_dram) {
            const InterleavedAddrGen<true> g = {.bank_base_address = out1_addr, .page_size = bytes};
            noc_async_write(tokens_l1, get_noc_addr(0, g), bytes);
        } else {
            const InterleavedAddrGen<false> g = {.bank_base_address = out1_addr, .page_size = bytes};
            noc_async_write(tokens_l1, get_noc_addr(0, g), bytes);
        }
    }
    noc_async_write_barrier();
}
