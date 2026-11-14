// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// fuse-sparse-merge B2 microbench / pass 1 of the exact threshold selection: 256-bin histogram of the HIGH BYTE of the sortable bf16 key of every score of a column, S rows, NC cores.
// Core i reads scores[r][i*SEG .. (i+1)*SEG) (bf16 RM rows), histograms, and the histograms are summed by a 3-level NoC tree (8 -> 1 -> 4 -> 1) of packed u16-pair words; the root writes u32 [S][256] to out.
// sortable key: x = bf16 bits; key = (x & 0x8000) ? ~x : (x | 0x8000)  (monotone in the bf16 value, -inf lowest).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

static FORCE_INLINE uint32_t skey(uint32_t x) { return (x & 0x8000u) ? ((~x) & 0xFFFFu) : (x | 0x8000u); }

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t ROWB = get_compile_time_arg_val(1);  // bytes of one score row (n * 2)
    constexpr uint32_t SEG = get_compile_time_arg_val(2);   // scores per core per row
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t CB_D = get_compile_time_arg_val(4);
    constexpr uint32_t CB_H = get_compile_time_arg_val(5);
    constexpr uint32_t CB_R1 = get_compile_time_arg_val(6);
    constexpr uint32_t CB_R2 = get_compile_time_arg_val(7);
    constexpr uint32_t CB_R3 = get_compile_time_arg_val(8);
    constexpr uint32_t SEM1 = get_compile_time_arg_val(9);
    constexpr auto s_args = TensorAccessorArgs<10, 2>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset(), s_args.next_common_runtime_args_offset()>();
    constexpr uint32_t HW = 128;  // packed words per row (256 bins)
    const uint32_t s_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t i = get_arg_val<uint32_t>(0);
    auto cx = [&](uint32_t j) { return get_arg_val<uint32_t>(1 + 2 * j); };
    auto cy = [&](uint32_t j) { return get_arg_val<uint32_t>(2 + 2 * j); };
    const auto sa = TensorAccessor(s_args, s_addr, ROWB);
    const auto oa = TensorAccessor(o_args, o_addr, 1024);
    CircularBuffer cd(CB_D), ch(CB_H), r1(CB_R1), r2(CB_R2), r3(CB_R3);
    cd.reserve_back(1);
    ch.reserve_back(1);
    r1.reserve_back(1);
    r2.reserve_back(1);
    r3.reserve_back(1);
    const uint32_t d = cd.get_write_ptr();
    for (uint32_t r = 0; r < S; ++r) {
        noc_async_read(sa.get_noc_addr(r, i * SEG * 2), d + r * SEG * 2, SEG * 2);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ch.get_write_ptr());  // S x 128 packed words (u16 pairs)
    for (uint32_t w = 0; w < S * HW; ++w) h[w] = 0;
    for (uint32_t r = 0; r < S; ++r) {
        volatile tt_l1_ptr uint16_t* hh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(h + r * HW);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d + r * SEG * 2);
        // branchless high byte of the sortable key: hb = (x >> 8) ^ (0x80 | (0xFF if sign))  ->  sign ? ~hi : hi | 0x80
        for (uint32_t e = 0; e < SEG / 2; ++e) {
            const uint32_t w2 = p[e];
            const uint32_t a = (w2 >> 8) & 0xFFu, b = w2 >> 24;
            const uint32_t ia = a ^ (0x80u | (0u - (a >> 7)) & 0x7Fu);
            const uint32_t ib = b ^ (0x80u | (0u - (b >> 7)) & 0x7Fu);
            hh[ia] += 1;
            hh[ib] += 1;
        }
    }
    const uint32_t SLOT = S * HW * 4;
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM1));
    volatile tt_l1_ptr uint32_t* sem2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM1 + 1));
    volatile tt_l1_ptr uint32_t* sem3 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM1 + 2));
    // ---- level 1: 8 -> 1
    const uint32_t g0 = i & ~7u, k1 = i & 7u;
    if (k1 != 0) {
        const uint64_t dst = get_noc_addr(cx(g0), cy(g0), r1.get_write_ptr() + (k1 - 1) * SLOT);
        noc_async_write(ch.get_write_ptr(), dst, SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx(g0), cy(g0), get_semaphore(SEM1)), 1);
        return;
    }
    noc_semaphore_wait(sem, 7);
    {
        volatile tt_l1_ptr uint32_t* rx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1.get_write_ptr());
        for (uint32_t s7 = 0; s7 < 7; ++s7)
            for (uint32_t w = 0; w < S * HW; ++w) h[w] += rx[s7 * S * HW + w];
    }
    // ---- level 2: 4 group leaders -> 1 (cores 0, 8, 16, 24 -> 0; 32.. -> 32 ...)
    const uint32_t gi = i >> 3, k2 = gi & 3u, g1 = (gi & ~3u) << 3;
    if (k2 != 0) {
        const uint64_t dst = get_noc_addr(cx(g1), cy(g1), r2.get_write_ptr() + (k2 - 1) * SLOT);
        noc_async_write(ch.get_write_ptr(), dst, SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx(g1), cy(g1), get_semaphore(SEM1 + 1)), 1);
        return;
    }
    noc_semaphore_wait(sem2, 3);
    {
        volatile tt_l1_ptr uint32_t* rx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2.get_write_ptr());
        for (uint32_t s3 = 0; s3 < 3; ++s3)
            for (uint32_t w = 0; w < S * HW; ++w) h[w] += rx[s3 * S * HW + w];
    }
    // ---- level 3: 4 super leaders (0, 32, 64, 96) -> 0
    const uint32_t k3 = i >> 5;
    if (k3 != 0) {
        const uint64_t dst = get_noc_addr(cx(0), cy(0), r3.get_write_ptr() + (k3 - 1) * SLOT);
        noc_async_write(ch.get_write_ptr(), dst, SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx(0), cy(0), get_semaphore(SEM1 + 2)), 1);
        return;
    }
    noc_semaphore_wait(sem3, NC / 32 - 1);
    // root: unpack to u32 and write
    volatile tt_l1_ptr uint32_t* rx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r3.get_write_ptr());
    cd.reserve_back(0);
    volatile tt_l1_ptr uint32_t* o32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);  // reuse the data buffer (S*SEG*2 >= S*1024 when SEG >= 512)
    for (uint32_t r = 0; r < S; ++r) {
        for (uint32_t w = 0; w < HW; ++w) {
            uint32_t a = h[r * HW + w], lo = a & 0xFFFFu, hi = a >> 16;
            for (uint32_t q = 0; q < NC / 32 - 1; ++q) {
                const uint32_t b = rx[q * S * HW + r * HW + w];
                lo += b & 0xFFFFu;
                hi += b >> 16;
            }
            o32[r * 256 + 2 * w] = lo;
            o32[r * 256 + 2 * w + 1] = hi;
        }
        noc_async_write(d + r * 1024, oa.get_noc_addr(r), 1024);
    }
    noc_async_write_barrier();
}
