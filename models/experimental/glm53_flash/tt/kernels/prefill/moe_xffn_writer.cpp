// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 XSL expert FFN (moe_xffn_common.hpp) -- writer (NCRISC, NoC1): per M group (control page: tile rows mt,
// valid rows, first dispatch row d0; mt = 0 = end), per N chunk and tile row, the compute's ROW_MAJOR block (32 rows x
// NC tiles) goes to out_rm rows d0 + 32 r + row at column bytes [nc NC 64, +NC 64) -- only the valid rows (the
// padding rows of a region are never read by the combine). Compile-time: H_T, NC (+ KC, MG with XFFN_X_ON_NOC1).
// Runtime: out_rm address (+ x_ext, disp addresses, disp words with XFFN_X_ON_NOC1).
// XFFN_X_ON_NOC1: this kernel (NCRISC, NoC1) also gathers each group's x rows for the gate|up phase (CB_XRM, through
// the plan's disp rows), so the weight stream (reader, NoC0) and the activation gather use different NoCs. With
// XFFN_X_SPLIT only the ODD K chunks (into CB_XRM1; the reader gathers the even ones into CB_XRM): the gather -- the
// critical path of the gate|up phase (in-program probe, 2026-10-22) -- runs on both RISCs / NoCs. XFFN_VALID_ROWS (and
// XFFN_X_SPLIT): only the group's valid rows are gathered (x_ext then needs no zero row).
// XFFN_PROBE: runtime arg 4 = the probe tensor (moe_xffn_probe.hpp), row P + core (P, core: args 5, 6).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_xffn_common.hpp"
#include "moe_xffn_probe.hpp"

using namespace xffn;
using namespace xffn_probe_slot;

void kernel_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t ROW_BYTES = H_T * 32 * 2;
    constexpr uint32_t SEG = NC * 32 * 2;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    XP_BEGIN();
#ifdef XFFN_X_ON_NOC1
    constexpr uint32_t KC = get_compile_time_arg_val(2);
    constexpr uint32_t MG = get_compile_time_arg_val(3);
    constexpr uint32_t XSEG = KC * 32 * 2;
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t disp_addr = get_arg_val<uint32_t>(2);
    const uint32_t cap = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<true> x_g{.bank_base_address = x_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = cap * 4};
    volatile tt_l1_ptr uint32_t* dl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WTAB));
#endif

    while (true) {
        XP_TIC();
        cb_wait_front(CB_WCTL, 1);
        XP_TOC(W_CTL_WAIT);
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_WCTL));
        const uint32_t mt = c[0];
        const uint32_t rows = c[1];
        const uint32_t d0 = c[2];
        cb_pop_front(CB_WCTL, 1);
        if (mt == 0) {
            break;
        }
#ifdef XFFN_X_ON_NOC1
        XP_TIC();
        noc_async_read(disp_g.get_noc_addr(0, d0 * 4), reinterpret_cast<uint32_t>(dl), mt * 32 * 4);
        noc_async_read_barrier();
        XP_TOC(W_DISP_READ);
#if defined(XFFN_X_SPLIT) || defined(XFFN_VALID_ROWS)
        const uint32_t nr = rows;  // only the valid rows (a padding row's output is never written)
#else
        const uint32_t nr = mt * 32;  // every row of the group (padding rows read x_ext's zero row C)
#endif
        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
#ifdef XFFN_X_SPLIT
            if ((kc & 1) == 0) {
                continue;  // the reader gathers the even K chunks (NoC0) into CB_XRM
            }
            constexpr uint32_t CBX = CB_XRM1;
#else
            constexpr uint32_t CBX = CB_XRM;
#endif
            XP_TIC();
            cb_reserve_back(CBX, MG * KC);  // whole MG blocks (pushes land exactly on the CB end)
            XP_TOC(W_X_WAIT);
            XP_TIC();
            const uint32_t xp = get_write_ptr(CBX);
            for (uint32_t r = 0; r < nr; ++r) {
                noc_async_read(x_g.get_noc_addr(dl[r], kc * XSEG), xp + r * XSEG, XSEG);
            }
            noc_async_read_barrier();
            XP_TOC(W_X_READ);
            XP_ADD(W_X_KB, nr * XSEG / 1024);
            cb_push_back(CBX, MG * KC);
        }
#endif
#ifdef XFFN_NO_DOWN
        continue;
#endif
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            for (uint32_t r = 0; r < mt; ++r) {
                XP_TIC();
                cb_wait_front(CB_YRM, NC);
                XP_TOC(W_Y_WAIT);
                XP_TIC();
                const uint32_t rp = get_read_ptr(CB_YRM);
                for (uint32_t t = 0; t < 32; ++t) {
                    const uint32_t row = r * 32 + t;
                    if (row < rows) {
                        noc_async_write(rp + t * SEG, out_g.get_noc_addr(d0 + row, nc * SEG), SEG);
                    }
                }
                noc_async_write_barrier();
                XP_TOC(W_Y_WRITE);
                XP_ADD(W_Y_KB, (rows - r * 32 < 32 ? rows - r * 32 : 32) * SEG / 1024);
                cb_pop_front(CB_YRM, NC);
            }
        }
    }
#ifdef XFFN_PROBE
    // the writer's L1 scratch: CB_WTAB (the disp rows, done)
    XP_FLUSH(
        (get_write_ptr(CB_WTAB) + 63) & ~63u,
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5) + get_arg_val<uint32_t>(6));
#endif
}
