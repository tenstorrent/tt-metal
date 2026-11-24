// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill ring write of a long chunk (data movement only). Ring slot q (< 1024) takes row j of the chunk's
// k / v [1, H, T, 128] bfp8 TILE, j = the last real row (j < n) with j % 1024 == q. n comes from DRAM (the
// chunk's nreal buffer), so one program serves any n in (1024, T].
// With bh = (n - 1) / 1024 and m = n - 1024 bh: slots q < m read block bh, the others block bh - 1. 1024 rows are
// whole tiles, so a ring tile is a source tile, or (the one tile row that holds slot m, if m % 32 != 0) rows of
// two source tiles. A bfp8 tile row is its own bytes: 2 exponents + 32 mantissas (per 16-wide face row).
// Runtime args: nreal addr, n_runs, per run: src addr, ring addr, first tile, end tile (ring tile index
// (h * 32 + R) * 4 + c).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T32 = get_named_compile_time_arg_val("t32");  // chunk tile rows (T / 32)
    constexpr uint32_t NB = get_named_compile_time_arg_val("nb");    // tiles per batch
    constexpr uint32_t TB = 1088;                                    // bfp8 tile bytes
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_n = 2;
    const uint32_t la = get_write_ptr(cb_a), lb = get_write_ptr(cb_b), ln = get_write_ptr(cb_n);
    const InterleavedAddrGen<true> na = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = 64};
    noc_async_read(na.get_noc_addr(0), ln, 64);
    noc_async_read_barrier();
    const uint32_t n = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ln);
    const uint32_t bh = (n - 1) >> 10, m = n - (bh << 10);
    const uint32_t n_runs = get_arg_val<uint32_t>(1);
    for (uint32_t u = 0; u < n_runs; ++u) {
        const InterleavedAddrGen<true> sa = {.bank_base_address = get_arg_val<uint32_t>(2 + 4 * u), .page_size = TB};
        const InterleavedAddrGen<true> ra = {.bank_base_address = get_arg_val<uint32_t>(3 + 4 * u), .page_size = TB};
        const uint32_t p1 = get_arg_val<uint32_t>(5 + 4 * u);
        for (uint32_t p0 = get_arg_val<uint32_t>(4 + 4 * u); p0 < p1; p0 += NB) {
            const uint32_t cnt = p1 - p0 < NB ? p1 - p0 : NB;
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t p = p0 + i, c = p & 3, R = (p >> 2) & 31, h = p >> 7, q0 = 32 * R;
                const uint32_t base = (h * T32 + R) * 4 + c;  // + 128 per block
                noc_async_read(sa.get_noc_addr(base + 128 * (q0 < m ? bh : bh - 1)), la + i * TB, TB);
                if (q0 < m && m < q0 + 32) {  // rows >= m - q0 come from block bh - 1
                    noc_async_read(sa.get_noc_addr(base + 128 * (bh - 1)), lb + i * TB, TB);
                }
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t q0 = 32 * (((p0 + i) >> 2) & 31);
                if (q0 < m && m < q0 + 32) {
                    volatile tt_l1_ptr uint8_t* a = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(la + i * TB);
                    const volatile tt_l1_ptr uint8_t* b =
                        reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(lb + i * TB);
                    for (uint32_t r = m - q0; r < 32; ++r) {
                        for (uint32_t f = 2 * (r >> 4); f < 2 * (r >> 4) + 2; ++f) {
                            const uint32_t e = f * 16 + (r & 15), o = 64 + f * 256 + (r & 15) * 16;
                            a[e] = b[e];
                            for (uint32_t x = 0; x < 16; ++x) {
                                a[o + x] = b[o + x];
                            }
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < cnt; ++i) {
                noc_async_write(la + i * TB, ra.get_noc_addr(p0 + i), TB);
            }
            noc_async_write_barrier();
        }
    }
}
