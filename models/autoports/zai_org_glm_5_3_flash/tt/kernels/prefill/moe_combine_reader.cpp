// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 EP combine — reader (moe_combine_common.hpp). Compile-time: C (tokens), K (slots), CTC (column tiles of
// H). Runtime: out, dest, w addresses, first unit, unit count. dest [1, K C] uint32 and w [1, K C] fp32 are single
// ROW_MAJOR pages (pair p = j C + t); out [CAP, H] bf16 TILE. A local pair's row d of column tile c is two 16-element
// face rows (32 B each) of out's tile (d / 32, c); each is read as its 64-B aligned DRAM block and copied into the
// slot tile's row.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_combine_common.hpp"

using namespace mcomb;

void kernel_main() {
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t CTC = get_compile_time_arg_val(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t dest_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(4);
    if (n == 0) {
        return;
    }
    const InterleavedAddrGen<true> og{.bank_base_address = out_addr, .page_size = BF16_TILE};
    const InterleavedAddrGen<true> dg{.bank_base_address = dest_addr, .page_size = K * C * 4};
    const InterleavedAddrGen<true> wg{.bank_base_address = w_addr, .page_size = K * C * 4};
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    const uint32_t idx = get_write_ptr(CB_IDX);  // [K][32] dest | [K][32] w | 64 x 64-B staging blocks
    volatile tt_l1_ptr uint32_t* dest = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx);
    volatile tt_l1_ptr uint32_t* wbits = dest + K * 32;
    volatile tt_l1_ptr float* wf = reinterpret_cast<volatile tt_l1_ptr float*>(wbits);
    const uint32_t stage = idx + 2 * K * 32 * 4;  // 64-B aligned (K * 256 bytes before it)

    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t r = u / CTC, c = u % CTC;
        if (r != cur_r) {  // the row tile's dest / weights + its K column-0 weight tiles
            cur_r = r;
            for (uint32_t j = 0; j < K; ++j) {
                noc_async_read(dg.get_noc_addr(0, (j * C + r * 32) * 4), idx + j * 128, 128);
                noc_async_read(wg.get_noc_addr(0, (j * C + r * 32) * 4), idx + (K + j) * 128, 128);
            }
            noc_async_read_barrier();
            cb_reserve_back(CB_W, K);
            const uint32_t wt = get_write_ptr(CB_W);
            // zero the whole tiles first: the column broadcast must never see stale L1 (NaN bits) in columns 1..31
            for (uint32_t off = 0; off < K * FP32_TILE; off += MEM_ZEROS_SIZE) {
                noc_async_read(
                    zeros, wt + off, MEM_ZEROS_SIZE < K * FP32_TILE - off ? MEM_ZEROS_SIZE : K * FP32_TILE - off);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < K; ++j) {
                volatile tt_l1_ptr float* t = reinterpret_cast<volatile tt_l1_ptr float*>(wt + j * FP32_TILE);
                for (uint32_t rr = 0; rr < 32; ++rr) {
                    t[tile_off(rr, 0)] = wf[j * 32 + rr];
                }
            }
            cb_push_back(CB_W, K);
        }
        cb_reserve_back(CB_S, K);
        const uint32_t s0 = get_write_ptr(CB_S);
        for (uint32_t off = 0; off < K * BF16_TILE; off += MEM_ZEROS_SIZE) {
            noc_async_read(
                zeros, s0 + off, MEM_ZEROS_SIZE < K * BF16_TILE - off ? MEM_ZEROS_SIZE : K * BF16_TILE - off);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < K; ++j) {
            // the slot's local rows: their 2 face rows, staged as 64-B aligned blocks (<= 64 reads in flight)
            uint32_t ns = 0;
            for (uint32_t rr = 0; rr < 32; ++rr) {
                if (wbits[j * 32 + rr] == 0) {
                    continue;  // another chip's pair (weight 0): the zero row stays
                }
                const uint32_t d = dest[j * 32 + rr];
                const uint32_t dr = d & 31u;
                const uint32_t base_b = ((((dr >> 4) << 1) * 256u) + (dr & 15u) * 16u) * 2u;  // face row, left half
                const uint64_t tile = og.get_noc_addr((d >> 5) * CTC + c, 0);
                noc_async_read(tile + (base_b & ~63u), stage + ns * 64, 64);
                noc_async_read(tile + ((base_b + 512u) & ~63u), stage + (ns + 1) * 64, 64);
                ns += 2;
            }
            noc_async_read_barrier();
            ns = 0;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s0 + j * BF16_TILE);
            for (uint32_t rr = 0; rr < 32; ++rr) {
                if (wbits[j * 32 + rr] == 0) {
                    continue;
                }
                const uint32_t dr = dest[j * 32 + rr] & 31u;
                const uint32_t src_b = ((((dr >> 4) << 1) * 256u) + (dr & 15u) * 16u) * 2u;
                const uint32_t dst_w = ((((rr >> 4) << 1) * 256u) + (rr & 15u) * 16u) / 2u;  // bf16 pairs -> words
                for (uint32_t h = 0; h < 2; ++h) {  // left / right face row (+256 elements = +512 B)
                    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        stage + (ns + h) * 64 + ((src_b + h * 512u) & 63u));
                    for (uint32_t q = 0; q < 8; ++q) {
                        st[dst_w + h * 128u + q] = src[q];
                    }
                }
                ns += 2;
            }
        }
        cb_push_back(CB_S, K);
    }
}
