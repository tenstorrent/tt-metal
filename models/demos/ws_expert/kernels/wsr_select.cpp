// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head, BRISC (NOC0: DRAM reads only): for each row tile of this core's tokens, reads its NET logit tiles
// (fp32 [32 tokens x NE experts]) and picks each token's top K experts by comparing the fp32 bit patterns as ordered
// integers (no float compare: these RISC-Vs have no FPU). Per row tile it pushes one fp32 tile into CB 1: token r's
// row holds its K picks' logits minus its largest (descending: column 0 is 0), the rest -inf (exp -> 0), and one
// page into CB 3: the picks' expert ids (uint16, K a token, 32 tokens).
// RT: 0 logits addr, 1 first row tile, 2 row tiles, 3 T (rows past it get no picks: K ids of 0xFFFF), 4 stamps
// (0: start, 1: the picks done). CT: 0 NET (expert tiles), 1 K, then TensorAccessorArgs(logits).
#include "api/dataflow/dataflow_api.h"

namespace {
// fp32 bits -> an integer with the same order (negatives: magnitude bits flipped)
FORCE_INLINE int32_t key(uint32_t f) {
    return static_cast<int32_t>(f ^ ((static_cast<int32_t>(f) >> 31) & 0x7FFFFFFF));
}

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// a tile's element (r, c), 4-byte elements, faces of 16 x 16
FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
}  // namespace

void kernel_main() {
    constexpr uint32_t NET = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr auto largs = TensorAccessorArgs<2>();
    constexpr uint32_t l_cb = tt::CBIndex::c_0, s_cb = tt::CBIndex::c_1, id_cb = tt::CBIndex::c_3;
    constexpr uint32_t tbytes = 4096;
    static_assert(K <= 32);
    const auto ls = TensorAccessor(largs, get_arg_val<uint32_t>(0), tbytes);
    const uint32_t r0 = get_arg_val<uint32_t>(1), nr = get_arg_val<uint32_t>(2), T = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    stamps[0] = wall();
    for (uint32_t rt = r0; rt < r0 + nr; ++rt) {
        cb_reserve_back(l_cb, NET);
        const uint32_t l1 = get_write_ptr(l_cb);
        for (uint32_t j = 0; j < NET; ++j) {
            noc_async_read_page(rt * NET + j, ls, l1 + j * tbytes);
        }
        noc_async_read_barrier();
        cb_push_back(l_cb, NET);
        cb_wait_front(l_cb, NET);
        cb_reserve_back(s_cb, 1);
        cb_reserve_back(id_cb, 1);
        volatile tt_l1_ptr uint32_t* lg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(l_cb));
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(s_cb));
        volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(id_cb));
        for (uint32_t r = 0; r < 32; ++r) {
            int32_t kv[K];
            uint32_t kb[K];
            uint16_t ke[K];
            uint32_t n = 0;
            if (rt * 32 + r < T) {
                for (uint32_t j = 0; j < NET; ++j) {
                    const volatile tt_l1_ptr uint32_t* tile = lg + j * 1024;
                    for (uint32_t c = 0; c < 32; ++c) {
                        const uint32_t b = tile[at(r, c)];
                        const int32_t v = key(b);
                        if (n == K && v <= kv[K - 1]) {
                            continue;
                        }
                        uint32_t p = n < K ? n++ : K - 1;  // insertion point from the end
                        while (p > 0 && kv[p - 1] < v) {
                            kv[p] = kv[p - 1];
                            kb[p] = kb[p - 1];
                            ke[p] = ke[p - 1];
                            --p;
                        }
                        kv[p] = v;
                        kb[p] = b;
                        ke[p] = static_cast<uint16_t>(j * 32 + c);
                    }
                }
            }
            const float top = n ? __builtin_bit_cast(float, kb[0]) : 0.0f;
            for (uint32_t c = 0; c < 32; ++c) {
                uint32_t out = 0xFF800000u;  // -inf
                if (c < n) {
                    out = __builtin_bit_cast(uint32_t, __builtin_bit_cast(float, kb[c]) - top);
                }
                s[at(r, c)] = out;
            }
            for (uint32_t k = 0; k < K; ++k) {
                ids[r * K + k] = k < n ? ke[k] : 0xFFFF;
            }
        }
        cb_pop_front(l_cb, NET);
        cb_push_back(s_cb, 1);
        cb_push_back(id_cb, 1);
    }
    stamps[1] = wall();
}
