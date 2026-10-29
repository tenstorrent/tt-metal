// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Column-sharded decode indexer, final stage: the merge top-k returned for every row r the K positions ``pos[r]``
// (uint32 ROW_MAJOR [R,1,1,K], indices into the W = COLS * K gathered candidates of the row) -> ids[r] = the
// candidates' GLOBAL entry ids ``cand[r][pos[r][i]]`` (uint32 ROW_MAJOR [R,1,1,W]), sorted ASCENDING (a position >= W
// gives the sentinel 0xFFFFFFFF, which sorts last): the gather and the canonical id order (tt/mhc_kernels/ids_sort.cpp:
// 3-pass LSD radix sort over the low 24 bits) in one program, one core per row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_pos = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cand = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_b = get_compile_time_arg_val(3);
    constexpr uint32_t cb_c = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t NW = get_compile_time_arg_val(7);  // bitmap words (ids < NW * 32), 0 = radix sort
    constexpr uint32_t SW = (NW + 31) / 32;
    constexpr auto pos_args = TensorAccessorArgs<8>();
    constexpr auto cand_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<cand_args.next_compile_time_args_offset()>();

    const uint32_t t_first = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t cand_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto pos_acc = TensorAccessor(pos_args, pos_addr, K * 4);
    const auto cand_acc = TensorAccessor(cand_args, cand_addr, W * 4);
    const auto out_acc = TensorAccessor(out_args, out_addr, K * 4);
    experimental::CB cbp(cb_pos), cbd(cb_cand), cba(cb_a), cbb(cb_b), cbc(cb_c);
    cbp.reserve_back(1);
    cbd.reserve_back(1);
    cba.reserve_back(1);
    cbb.reserve_back(1);
    cbc.reserve_back(1);
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp.get_write_ptr());
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbd.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());
    volatile tt_l1_ptr uint32_t* bm = cnt;
    volatile tt_l1_ptr uint32_t* summ = cnt + NW;

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        noc.async_read(pos_acc, cbp, K * 4, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(cand_acc, cbd, W * 4, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        if constexpr (NW > 0) {
            // two-level bitmap sort (ids < NW * 32): summary bit per bitmap word, words initialised on first touch, no
            // radix passes
            for (uint32_t i = 0; i < SW; ++i) {
                summ[i] = 0;
            }
            uint32_t nsent = 0;
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t p = pos[i];
                const uint32_t id = (p < W) ? cand[p] : 0xFFFFFFFFu;
                if (id >= NW * 32u) {
                    ++nsent;
                    continue;
                }
                const uint32_t w = id >> 5;
                const uint32_t sb = 1u << (w & 31u);
                if (!(summ[w >> 5] & sb)) {
                    summ[w >> 5] |= sb;
                    bm[w] = 0;
                }
                bm[w] |= 1u << (id & 31u);
            }
            uint32_t o = 0;
            for (uint32_t s = 0; s < SW; ++s) {
                uint32_t sw = summ[s];
                while (sw) {
                    const uint32_t j = __builtin_ctz(sw);
                    sw &= sw - 1;
                    const uint32_t w = (s << 5) + j;
                    uint32_t b = bm[w];
                    while (b) {
                        const uint32_t t = __builtin_ctz(b);
                        b &= b - 1;
                        pb[o++] = (w << 5) + t;
                    }
                }
            }
            for (; o < K; ++o) {
                pb[o] = 0xFFFFFFFFu;
            }
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t p = pos[i];
                pa[i] = (p < W) ? cand[p] : 0xFFFFFFFFu;
            }
            volatile tt_l1_ptr uint32_t* src = pa;
            volatile tt_l1_ptr uint32_t* dst = pb;
            for (uint32_t pass = 0; pass < 3; ++pass) {
                const uint32_t sh = pass * 8;
                for (uint32_t d = 0; d < 256; ++d) {
                    cnt[d] = 0;
                }
                for (uint32_t i = 0; i < K; ++i) {
                    cnt[(src[i] >> sh) & 255u]++;
                }
                uint32_t run = 0;
                for (uint32_t d = 0; d < 256; ++d) {
                    const uint32_t c = cnt[d];
                    cnt[d] = run;
                    run += c;
                }
                for (uint32_t i = 0; i < K; ++i) {
                    const uint32_t v = src[i];
                    dst[cnt[(v >> sh) & 255u]++] = v;
                }
                volatile tt_l1_ptr uint32_t* tmp = src;
                src = dst;
                dst = tmp;
            }
            // three passes: the sorted keys are in pb
        }
        noc.async_write(cbb, out_acc, K * 4, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
