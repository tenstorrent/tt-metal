// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC expand + next projection, writer.  (1) builds the projection constants (ONES, SEL_i, SEL_all; see
// mhc_proj2_writer.cpp); (2) writes the new streams: X'_j row 4*tl + j' -> rows 0..3 of the zero-padded token page
// (tok0 + tl, column tile j) of x_new; (3) writes the TG valid rows of the projection partial into rows [row0, row0 +
// TG) of its packed partial page.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_ones = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stg = get_compile_time_arg_val(3);  // NSTG staging pages (rows 4..31 stay zero)
    constexpr uint32_t cb_p = get_compile_time_arg_val(4);
    constexpr uint32_t NCOL = get_compile_time_arg_val(5);
    constexpr uint32_t TG = get_compile_time_arg_val(6);
    constexpr uint32_t TPJ = get_compile_time_arg_val(7);
    constexpr uint32_t NT = get_compile_time_arg_val(8);
    constexpr uint32_t NSTG = get_compile_time_arg_val(9);
    constexpr auto xo_args = TensorAccessorArgs<10>();
    constexpr auto p_args = TensorAccessorArgs<xo_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t tok0 = get_arg_val<uint32_t>(1);
    const uint32_t pg = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
    const uint32_t xo_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto xo_acc = TensorAccessor(xo_args, xo_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cbo(cb_ones), cbs(cb_sel), cbx(cb_xn), cbg(cb_stg), cbp(cb_p);

    cbo.reserve_back(1);
    cbs.reserve_back(5);
    cbg.reserve_back(NSTG);
    noc.async_write_zeros(cbo, TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbs, 5 * TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbg, NSTG * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t rr = 0; rr < 32; ++rr) {
            const uint32_t face = (rr >= 16 ? 2 : 0) + (NCOL >= 16 ? 1 : 0);
            o[face * 256 + (rr % 16) * 16 + (NCOL % 16)] = 0x3F800000u;
        }
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t tl = 0; tl < TG; ++tl) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t c = 4 * tl + i;
                const uint32_t face = (tl >= 16 ? 2 : 0) + (c >= 16 ? 1 : 0);
                const uint32_t w = face * 256 + (tl % 16) * 16 + (c % 16);
                s[i * 1024 + w] = 0x3F800000u;
                s[4 * 1024 + w] = 0x3F800000u;
            }
        }
    }
    cbo.push_back(1);
    cbs.push_back(5);

    // ---- new streams ----
    volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbg.get_write_ptr());
    cbx.wait_front(TPJ);
    volatile tt_l1_ptr uint32_t* xn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbx.get_read_ptr());
    uint32_t used = 0;
    uint32_t pages[NSTG];
    for (uint32_t jj = 0; jj < TPJ; ++jj) {
        for (uint32_t tl = 0; tl < TG; ++tl) {
            volatile tt_l1_ptr uint32_t* dst = stg + used * (TILE / 4);
            volatile tt_l1_ptr uint32_t* src =
                xn + jj * (TILE / 4) + (RO(4 * tl) >> 2);  // 4 rows of 16 words, face 0 (| face 2)
            for (uint32_t w = 0; w < 64; ++w) {
                dst[w] = src[w];
                dst[256 + w] = src[256 + w];
            }
            pages[used] = (tok0 + tl) * NT + r * TPJ + jj;
            if (++used == NSTG) {
                for (uint32_t u = 0; u < used; ++u) {
                    noc.async_write(
                        cbg, xo_acc, TILE, {.offset_bytes = u * TILE}, {.page_id = pages[u], .offset_bytes = 0});
                }
                noc.async_write_barrier();
                used = 0;
            }
        }
    }
    if (used) {
        for (uint32_t u = 0; u < used; ++u) {
            noc.async_write(cbg, xo_acc, TILE, {.offset_bytes = u * TILE}, {.page_id = pages[u], .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
    cbx.pop_front(TPJ);

    cbp.wait_front(1);
    const uint32_t dst_off = ((row0 >> 4) << 11) + ((row0 & 15) << 6);
    noc.async_write(cbp, p_acc, TG * 64, {.offset_bytes = 0}, {.page_id = pg, .offset_bytes = dst_off});
    noc.async_write(cbp, p_acc, TG * 64, {.offset_bytes = 1024}, {.page_id = pg, .offset_bytes = dst_off + 1024});
    noc.async_write_barrier();
    cbp.pop_front(1);
}
