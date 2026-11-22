// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// M1L-0 microbenchmark (STEP3_PLAN_v3.1 N3): one full-attention layer's K and V reads for one user, the way the
// chunked M1L workers will read them. This core takes position tiles [t0, t0 + n) of the layer, in chunks of `chunk`
// tiles: per chunk the page-table entries it needs (64 B aligned reads of 16 entries, kept until used up), then the
// chunk's K tiles (16 per position tile, head_dim 512) and V tiles into cb 0 / cb 1, then a barrier. No compute: the
// achieved bandwidth is the reads' ceiling under the paged, interleaved BF8 layout of the aliased full view
// ([pages, 1, 128, 512]: page p holds tiles [64 p, 64 p + 64)). kv_split: K over this kernel's NOC, V over the other
// kernel (a second instance on the other RISC / NOC takes V); 0: this kernel reads both.
//
// Compile-time args: 0 chunk (position tiles), 1 kv_split, 2 which (0: K and V or K only, 1: V only)
// Runtime args: 0 K address, 1 V address, 2 table address (DRAM row of P int32), 3 t0, 4 n, 5 iterations,
// 6 timing word (L1, 0 = off: wall clock at start and end), 7 table row bytes (4 P)

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t chunk = get_compile_time_arg_val(0);
constexpr bool kv_split = get_compile_time_arg_val(1) != 0;
constexpr uint32_t which = get_compile_time_arg_val(2);
constexpr uint32_t kBf8 = 1088, kDt = 16;

void kernel_main() {
    const uint32_t k_addr = get_arg_val<uint32_t>(0), v_addr = get_arg_val<uint32_t>(1);
    const uint32_t tab_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3), n = get_arg_val<uint32_t>(4);
    const uint32_t iters = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* tw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const InterleavedAddrGen<true> tg{.bank_base_address = tab_addr, .page_size = get_arg_val<uint32_t>(7)};
    const InterleavedAddrGenFast<true> kg{.bank_base_address = k_addr, .page_size = kBf8, .data_format = DataFormat::Bfp8_b};
    const InterleavedAddrGenFast<true> vg{.bank_base_address = v_addr, .page_size = kBf8, .data_format = DataFormat::Bfp8_b};
    auto& ck = get_local_cb_interface(0);
    auto& cv = get_local_cb_interface(1);
    auto& ct = get_local_cb_interface(2 + which);  // each instance its own table block
    const uint32_t kbuf = ck.fifo_limit - ck.fifo_size, vbuf = cv.fifo_limit - cv.fifo_size;
    const uint32_t tbuf = ct.fifo_limit - ct.fifo_size;  // 64 B: 16 entries
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbuf);
    const bool do_k = !kv_split || which == 0, do_v = !kv_split || which == 1;
    if (tw) {
        tw[0] = *clk;
    }
    for (uint32_t it = 0; it < iters; it++) {
        uint32_t cached = 0xffffffff;  // the 16-entry block in tbuf
        for (uint32_t c0 = t0; c0 < t0 + n; c0 += chunk) {
            const uint32_t c1 = c0 + chunk < t0 + n ? c0 + chunk : t0 + n;
            for (uint32_t t = c0; t < c1; t++) {
                const uint32_t col = t / 4, blk = col / 16;
                if (blk != cached) {
                    // the table row: 64 B aligned 16-entry reads (DRAM read alignment)
                    noc_async_read(tg.get_noc_addr(0, blk * 64), tbuf, 64);
                    noc_async_read_barrier();
                    cached = blk;
                }
                const uint32_t page = tab[col % 16];
                const uint32_t base = page * 64 + (t % 4) * kDt;
                const uint32_t slot = (t - c0) * kDt;
                for (uint32_t d = 0; d < kDt; d++) {
                    if (do_k) {
                        noc_async_read_tile(base + d, kg, kbuf + (slot + d) * kBf8);
                    }
                    if (do_v) {
                        noc_async_read_tile(base + d, vg, vbuf + (slot + d) * kBf8);
                    }
                }
            }
            noc_async_read_barrier();
        }
    }
    if (tw) {
        tw[1] = *clk;
    }
}
