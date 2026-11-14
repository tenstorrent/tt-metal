// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode SDPA (sdpa_dec.py), writer (BRISC). Core (g, c):
// 1. masks the scores past the last position (last tile row), 2. sends its 6 row maxima to the group's NG cores and
// takes the global maxima M (cb_M), 3. sends rows 0 .. 5 of its partial P V (tile d) and of P 1 (row sums) into
// stacked tiles on the group's combiner d (core c = d < 8): partial c at stack tile c / 5, rows 6 (c % 5) ..,
// 4. combiner d: once all NG partials landed, the compute sums and normalises them; rows 0 .. 5 of the result go to
// rows 6g .. 6g + 5 of output tile d (token-major [1, 1, 32, 256]).
#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"

namespace {
// fp32 bits -> an unsigned key in float order
inline uint32_t fkey(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

// rows 0 .. 5 of a bf16 tile at src (L1) -> rows R0 .. R0 + 5 of the bf16 tile at dst (NOC)
inline void put_rows(uint32_t src, uint64_t dst, uint32_t R0) {
    uint32_t r = 0;
    while (r < 6) {
        const uint32_t R = R0 + r;
        const uint32_t run = R < 16 ? (16 - R < 6 - r ? 16 - R : 6 - r) : 6 - r;
        const uint32_t off = R < 16 ? R * 32 : 1024 + (R - 16) * 32;
        noc_async_write(src + r * 32, dst + off, run * 32);
        noc_async_write(src + 512 + r * 32, dst + off + 512, run * 32);
        r += run;
    }
}
}  // namespace

void kernel_main() {
    // common args: out, NG, NC (combiners per group), then NOC xy of the H * NG cores (x << 16 | y), group-major
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t NG = get_common_arg_val<uint32_t>(1), NC = get_common_arg_val<uint32_t>(2);
    const uint32_t g = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1);
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_info = 5, cb_tok = 7, cb_M = 8, cb_stk = 9, cb_x = 10, cb_zero = 11;
    constexpr uint32_t cb_s = 16, cb_m = 17, cb_o = 18, cb_out = 19;
    const uint32_t TB = get_tile_size(cb_o);
    const auto os = TensorAccessor(o_args, out_addr, TB);
    const uint32_t sem_max = get_semaphore(0), sem_part = get_semaphore(1);
    auto xy = [&](uint32_t j) {
        const uint32_t v = get_common_arg_val<uint32_t>(3 + g * NG + j);
        return std::pair<uint32_t, uint32_t>(v >> 16, v & 0xffff);
    };
    const bool comb = c < NC;
    const uint32_t stk = get_write_ptr(cb_stk);
    const uint32_t zero = get_write_ptr(cb_zero);
    {
        // the constant bf16 tiles: ones, the reduce scaler (row 0 of each face), the row selection Sel[r, 6k + r]
        constexpr uint32_t cb_one = 3;
        cb_reserve_back(cb_one, 3);
        volatile tt_l1_ptr uint32_t* t = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_one);
        for (uint32_t i = 0; i < 512; ++i) {
            t[i] = 0x3F803F80u;
            t[512 + i] = (i % 128) < 8 ? 0x3F803F80u : 0;
            t[1024 + i] = 0;
        }
        volatile tt_l1_ptr uint16_t* sel = (volatile tt_l1_ptr uint16_t*)(t + 1024);
        for (uint32_t r = 0; r < 6; ++r) {
            for (uint32_t k = 0; k < 5; ++k) {
                const uint32_t col = 6 * k + r;
                sel[(col < 16 ? 0 : 256) + r * 16 + col % 16] = 0x3F80;
            }
        }
        cb_push_back(cb_one, 3);
    }
    cb_wait_front(cb_info, 1);
    volatile tt_l1_ptr uint32_t* info = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_info);
    const uint32_t nr = info[0], last = info[1], valid = info[2];

    // 1. mask
    uint32_t m[6];
    if (nr != 0) {
        {
#ifdef SDPA_ZONES
            DeviceZoneScopedN("TD-W-waitS");
#endif
            cb_wait_front(cb_s, nr);
        }
        if (last) {
            volatile tt_l1_ptr uint32_t* s = (volatile tt_l1_ptr uint32_t*)(get_read_ptr(cb_s) + (nr - 1) * 4096);
            for (uint32_t r = 0; r < 6; ++r) {
                for (uint32_t j = valid; j < 32; ++j) {
                    s[(j < 16 ? 0 : 256) + r * 16 + (j % 16)] = 0xCE6E6B28u;  // -1e9
                }
            }
        }
        cb_reserve_back(cb_tok, 1);
        cb_push_back(cb_tok, 1);
        // 2. row maxima: fp32 tile, row r column 0
        {
#ifdef SDPA_ZONES
            DeviceZoneScopedN("TD-W-waitm");
#endif
            cb_wait_front(cb_m, 1);
        }
        volatile tt_l1_ptr uint32_t* mt = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_m);
        for (uint32_t r = 0; r < 6; ++r) {
            m[r] = mt[r * 16];
        }
    } else {
        for (uint32_t r = 0; r < 6; ++r) {
            m[r] = 0xFF800000u;  // -inf
        }
    }
    const uint32_t xb = get_write_ptr(cb_x);
    volatile tt_l1_ptr uint32_t* mine = (volatile tt_l1_ptr uint32_t*)(xb + NG * 32);
    for (uint32_t r = 0; r < 6; ++r) {
        mine[r] = m[r];
    }
    for (uint32_t j = 0; j < NG; ++j) {
        const auto [x, y] = xy(j);
        noc_async_write(xb + NG * 32, get_noc_addr(x, y, xb + c * 32), 32);
    }
    noc_async_writes_flushed();
    for (uint32_t j = 0; j < NG; ++j) {
        const auto [x, y] = xy(j);
        noc_semaphore_inc(get_noc_addr(x, y, sem_max), 1);
    }
    {
#ifdef SDPA_ZONES
        DeviceZoneScopedN("TD-W-xchg");
#endif
        noc_semaphore_wait((volatile tt_l1_ptr uint32_t*)sem_max, NG);
    }
    cb_reserve_back(cb_M, 1);
    volatile tt_l1_ptr uint32_t* Mt = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_M);
    for (uint32_t r = 0; r < 6; ++r) {
        uint32_t best = 0xFF800000u;
        for (uint32_t j = 0; j < NG; ++j) {
            const uint32_t v = ((volatile tt_l1_ptr uint32_t*)(xb + j * 32))[r];
            if (fkey(v) > fkey(best)) {
                best = v;
            }
        }
        Mt[r * 16] = best;
    }
    cb_push_back(cb_M, 1);

    // (while the compute takes P V) the zero rows for empty partials and padding, and the unused stack rows
    {
        volatile tt_l1_ptr uint32_t* z = (volatile tt_l1_ptr uint32_t*)zero;
        for (uint32_t i = 0; i < 2048 / 4; ++i) {
            z[i] = 0;
        }
    }
    if (comb) {
        // zero the stack rows no partial lands on (the row-selection matmul reads them)
        const uint32_t nst = (NG + 4) / 5;
        for (uint32_t t = 0; t < 2 * nst; ++t) {
            for (uint32_t R = 0; R < 32; ++R) {
                const uint32_t i = (t % nst) * 5 + R / 6;
                if (R < 30 && i < NG) {
                    continue;
                }
                const uint32_t off = R < 16 ? R * 32 : 1024 + (R - 16) * 32;
                volatile tt_l1_ptr uint32_t* p = (volatile tt_l1_ptr uint32_t*)(stk + t * TB + off);
                for (uint32_t w = 0; w < 8; ++w) {
                    p[w] = 0;
                    p[128 + w] = 0;
                }
            }
        }
    }
    // 3. partials to the combiners
    uint32_t po = zero, pl = zero;
    if (nr != 0) {
#ifdef SDPA_ZONES
        DeviceZoneScopedN("TD-W-waitO");
#endif
        cb_wait_front(cb_o, 9);
        po = get_read_ptr(cb_o);
        pl = po + 8 * TB;
    }
    const uint32_t nst = (NG + 4) / 5;
    for (uint32_t d = 0; d < NC; ++d) {
        const auto [x, y] = xy(d);
        put_rows(nr != 0 ? po + d * TB : zero, get_noc_addr(x, y, stk + (c / 5) * TB), 6 * (c % 5));
        put_rows(pl, get_noc_addr(x, y, stk + (nst + c / 5) * TB), 6 * (c % 5));
    }
    noc_async_writes_flushed();
    for (uint32_t d = 0; d < NC; ++d) {
        const auto [x, y] = xy(d);
        noc_semaphore_inc(get_noc_addr(x, y, sem_part), 1);
    }
    if (comb) {
        // 4. all partials in: the compute combines, rows 0 .. 5 to the output
        {
#ifdef SDPA_ZONES
            DeviceZoneScopedN("TD-W-waitpart");
#endif
            noc_semaphore_wait((volatile tt_l1_ptr uint32_t*)sem_part, NG);
        }
        cb_reserve_back(cb_stk, 2 * nst);
        cb_push_back(cb_stk, 2 * nst);
        {
#ifdef SDPA_ZONES
            DeviceZoneScopedN("TD-W-waitout");
#endif
            cb_wait_front(cb_out, 1);
        }
        put_rows(get_read_ptr(cb_out), os.get_noc_addr(c), 6 * g);
        if (6 * g + 6 == 24) {  // the padding rows 24 .. 31 zero
            put_rows(zero, os.get_noc_addr(c), 24);
            put_rows(zero, os.get_noc_addr(c), 26);
        }
    }
    noc_async_write_barrier();
}
