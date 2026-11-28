// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Wait: poll done_seq == req_seq for at most timeout_us (wall clock; then 0xDEAD0000 | req in the wait status
// word and return: a stuck responder never hangs the device; the reply is not copied). Optionally copy n_words u32 of
// the channel's reply line into an output page (bank NoC coordinates + address) after the release. Runtime args: l2_x,
// l2_y, base_hi, base_lo, timeout_us, n_words, out_x, out_y, out_addr, diag diag != 0: accumulate into DIAG line 1
// (u32): [0] count, [1] sum(release - notify_end), [2] max, [3] sum(release - entry), [4] first notify_end, [5] last
// release, in wall-clock ticks.
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_noc.h"

void kernel_main() {
    l2cpu_link c{
        noc_index,
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        ((uint64_t)get_arg_val<uint32_t>(2) << 32) | get_arg_val<uint32_t>(3),
        get_write_ptr(0)};
    uint32_t timeout_us = get_arg_val<uint32_t>(4), n_words = get_arg_val<uint32_t>(5);
    uint64_t out = get_noc_addr(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8));
    uint32_t diag = get_arg_val<uint32_t>(9);
    uint32_t t_entry = get_timestamp_32b();
    uint32_t req = l2cpu_link_wait(c, timeout_us);
    uint32_t t_rel = get_timestamp_32b();
    if (req && n_words) {
        uint32_t s = c.l1 + 256;
        l2cpu_noc_read(c.noc, s, c.x, c.y, c.base + L2CPU_LINK_OFF_REPLY, 64);
        noc_async_read_barrier();
        noc_async_write(s, out, 4 * n_words);
        noc_async_write_barrier();
    }
    if (diag) {
        uint32_t t_notify = l2cpu_link_read32(c, L2CPU_LINK_OFF_DIAG, 8);
        uint32_t l = c.l1 + 64 * 9;
        l2cpu_noc_read(c.noc, l, c.x, c.y, c.base + L2CPU_LINK_OFF_DIAG + 64, 64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l);
        uint32_t gap = t_rel - t_notify;
        if (d[0] == 0) {
            d[4] = t_notify;
        }
        d[0] += 1;
        d[1] += gap;
        if (gap > d[2]) {
            d[2] = gap;
        }
        d[3] += t_rel - t_entry;
        d[5] = t_rel;
        l2cpu_noc_write(c.noc, l, c.x, c.y, c.base + L2CPU_LINK_OFF_DIAG + 64, 64);
        noc_async_write_barrier();
    }
}
