// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill -> decode handoff of one cache set per core (data movement only). Builds in L1:
//   x_hist: [32, C] bf16 TILE, row d = a source row (rows without a source stay zero);
//   NV ring vectors: dense fp32 [C] (3 pages of 1024 floats, zero padded), each = a source row widened to fp32;
// then writes them to DRAM. Sources are bf16 TILE tensors read one tile row (WT tiles) at a time in blocks.
// Runtime args: x_hist addr, NV, NV ring addrs, n_groups, per group: src addr, tile row, n_rows, then per row:
//   (row in tile, kind, index): kind 0 = x_hist row index, kind 1 = ring vector index.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t BLK = get_named_compile_time_arg_val("blk");  // tiles per read block
    constexpr uint32_t VEC_PAGES = get_named_compile_time_arg_val("vec_pages");
    constexpr uint32_t cb_xh = 0, cb_vec = 1, cb_stage = 2;
    uint32_t a = 0;
    const uint32_t xh = get_arg_val<uint32_t>(a++);
    const uint32_t nv = get_arg_val<uint32_t>(a++);
    const uint32_t vec0 = a;
    a += nv;
    const uint32_t n_groups = get_arg_val<uint32_t>(a++);

    const uint32_t xh_l1 = get_write_ptr(cb_xh), vec_l1 = get_write_ptr(cb_vec), st_l1 = get_write_ptr(cb_stage);
    {  // zero x_hist and the vectors
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(xh_l1);
        for (uint32_t i = 0; i < WT * 2048 / 4; ++i) {
            p[i] = 0;
        }
        p = reinterpret_cast<volatile uint32_t*>(vec_l1);
        for (uint32_t i = 0; i < nv * VEC_PAGES * 1024; ++i) {
            p[i] = 0;
        }
    }
    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t src = get_arg_val<uint32_t>(a), trow = get_arg_val<uint32_t>(a + 1);
        const uint32_t nr = get_arg_val<uint32_t>(a + 2);
        const uint32_t rows_at = a + 3;
        a += 3 + 3 * nr;
        const InterleavedAddrGen<true> sa = {.bank_base_address = src, .page_size = 2048};
        for (uint32_t j0 = 0; j0 < WT; j0 += BLK) {
            const uint32_t nb = (WT - j0) < BLK ? (WT - j0) : BLK;
            for (uint32_t b = 0; b < nb; ++b) {
                noc_async_read(sa.get_noc_addr(trow * WT + j0 + b), st_l1 + b * 2048, 2048);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t r = get_arg_val<uint32_t>(rows_at + 3 * i);
                const uint32_t kind = get_arg_val<uint32_t>(rows_at + 3 * i + 1);
                const uint32_t idx = get_arg_val<uint32_t>(rows_at + 3 * i + 2);
                for (uint32_t b = 0; b < nb; ++b) {
                    const uint32_t j = j0 + b;
                    for (uint32_t h = 0; h < 2; ++h) {  // the row's two 16-element face rows
                        const volatile uint16_t* s = reinterpret_cast<const volatile uint16_t*>(
                            st_l1 + b * 2048 + ((r >> 4) * 2 + h) * 512 + (r & 15) * 32);
                        if (kind == 0) {
                            volatile uint16_t* d = reinterpret_cast<volatile uint16_t*>(
                                xh_l1 + j * 2048 + ((idx >> 4) * 2 + h) * 512 + (idx & 15) * 32);
                            for (uint32_t e = 0; e < 16; ++e) {
                                d[e] = s[e];
                            }
                        } else {
                            volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(
                                vec_l1 + idx * VEC_PAGES * 4096 + (j * 32 + h * 16) * 4);
                            for (uint32_t e = 0; e < 16; ++e) {
                                d[e] = static_cast<uint32_t>(s[e]) << 16;
                            }
                        }
                    }
                }
            }
        }
    }
    const InterleavedAddrGen<true> xa = {.bank_base_address = xh, .page_size = 2048};
    for (uint32_t j = 0; j < WT; ++j) {
        noc_async_write(xh_l1 + j * 2048, xa.get_noc_addr(j), 2048);
    }
    for (uint32_t v = 0; v < nv; ++v) {
        const InterleavedAddrGen<true> va = {.bank_base_address = get_arg_val<uint32_t>(vec0 + v), .page_size = 4096};
        for (uint32_t p = 0; p < VEC_PAGES; ++p) {
            noc_async_write(vec_l1 + (v * VEC_PAGES + p) * 4096, va.get_noc_addr(p), 4096);
        }
    }
    noc_async_write_barrier();
}
