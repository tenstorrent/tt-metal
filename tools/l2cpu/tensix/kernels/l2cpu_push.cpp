// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Push: pages of a ROW_MAJOR interleaved DRAM tensor (one page per row; row b in bank (first_bank + b) % num_banks
// at bank_addr + (b / num_banks) * page_stride) -> L2CPU memory, row b at dst + b * row_bytes, through the L2CPU
// tile: dst is either the coherent Memory Port alias (x280 may cache the rows) or the uncached System Port alias
// (the x280 must then read those rows uncached only; avoids write-allocating many rows into its 2 MiB L3).
// The source table is read from the channel (L2CPU_LINK_OFF_PUSH_SRC, host-written), not from runtime args, so the
// program does not depend on where the source tensor was allocated (an eager warm-up and a trace capture share one
// cached program). Table: u32 row_bytes, page_stride, num_banks, first_bank, then 8 x {u32 x, u32 y, u64 addr}.
// Modes: notify_first = 0: rows in order, all writes acked before the kernel ends (a following notify program
//                          publishes complete data).
//        notify_first = 1: streamed: notify (req_seq + doorbell) FIRST, then rows in the consumer's order
//                          r(p) = (p % groups) * group_rows + p / groups for p = 0 .. groups * group_rows - 1
//                          (rows >= n_rows skipped; groups = 1 is plain order; e.g. 4 x 8 when 4 consumer harts
//                          own 8 rows each and should all start early); after each row a write barrier and
//                          landed = (req & 0xFFFF) << 16 | rows_complete (one 4-byte write, coherent alias).
// This core handles every row_step-th row starting at first_row (non-streamed mode; streamed mode is 1 core).
// Runtime args: l2_x, l2_y, base_hi, base_lo, dst_hi, dst_lo, n_rows, first_row, row_step, notify_first, groups,
//               group_rows, diag (streamed: wall clock after the doorbell into DIAG line 0, as l2cpu_notify.cpp).
//               The alias (coherent or uncached zone) is the caller's choice of dst.
// CB 0: 2 * row_bytes + 2 KiB.
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_noc.h"

static inline void read_row(uint32_t b, volatile tt_l1_ptr uint32_t* t, uint32_t dst) {
    uint32_t row_bytes = t[0], stride = t[1], nb = t[2], fb = t[3];
    volatile tt_l1_ptr uint32_t* e = t + 4 + 4 * ((fb + b) % nb);
    uint64_t src = get_noc_addr(e[0], e[1], e[2] + (b / nb) * stride);  // bank addresses < 4 GiB
    for (uint32_t o = 0; o < row_bytes; o += 8192) {
        noc_async_read(src + o, dst + o, row_bytes - o < 8192 ? row_bytes - o : 8192);
    }
}

void kernel_main() {
    l2cpu_link c{
        noc_index,
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        ((uint64_t)get_arg_val<uint32_t>(2) << 32) | get_arg_val<uint32_t>(3),
        get_write_ptr(0)};
    uint64_t dst = ((uint64_t)get_arg_val<uint32_t>(4) << 32) | get_arg_val<uint32_t>(5);
    uint32_t n_rows = get_arg_val<uint32_t>(6), first = get_arg_val<uint32_t>(7), step = get_arg_val<uint32_t>(8);
    uint32_t streamed = get_arg_val<uint32_t>(9), groups = get_arg_val<uint32_t>(10),
             group_rows = get_arg_val<uint32_t>(11);
    if (groups == 0) {
        groups = 1;
    }
    if (group_rows == 0) {
        group_rows = n_rows;
    }
    uint32_t tab = c.l1 + 1024;
    l2cpu_noc_read(c.noc, tab, c.x, c.y, c.base + L2CPU_LINK_OFF_PUSH_SRC, 192);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    uint32_t row_bytes = t[0];
    uint32_t buf[2] = {c.l1 + 2048, c.l1 + 2048 + ((row_bytes + 63) & ~63u)};
    uint32_t order[64], n = 0;
    for (uint32_t p = 0; streamed && p < groups * group_rows && n < 64; p++) {
        uint32_t r = (p % groups) * group_rows + p / groups;
        if (r < n_rows) {
            order[n++] = r;
        }
    }
    for (uint32_t r = first; r < n_rows && !streamed && n < 64; r += step) {
        order[n++] = r;
    }
    uint32_t req = streamed ? l2cpu_link_notify(c) : 0;
    if (streamed && get_arg_val<uint32_t>(12)) {
        l2cpu_link_write32(c, L2CPU_LINK_OFF_DIAG, get_timestamp_32b(), 3);
    }
    volatile tt_l1_ptr uint32_t* lv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.l1 + 64 * 6);
    if (n) {
        read_row(order[0], t, buf[0]);
    }
    for (uint32_t k = 0; k < n; k++) {
        noc_async_read_barrier();  // row k is in buf[k & 1]
        if (k + 1 < n) {
            noc_async_writes_flushed();                   // buf[(k+1) & 1] has left L1
            read_row(order[k + 1], t, buf[(k + 1) & 1]);  // overlaps this row's write
        }
        l2cpu_noc_write_bulk(c.noc, buf[k & 1], c.x, c.y, dst + (uint64_t)order[k] * row_bytes, row_bytes);
        if (streamed) {
            noc_async_write_barrier();  // row acked before it is announced
            lv[0] = ((req & 0xFFFFu) << 16) | ((k + 1) & 0xFFFFu);
            l2cpu_noc_write(c.noc, c.l1 + 64 * 6, c.x, c.y, c.base + L2CPU_LINK_OFF_LANDED, 4);
        }
    }
    noc_async_write_barrier();
}
