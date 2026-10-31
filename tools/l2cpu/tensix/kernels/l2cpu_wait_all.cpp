// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Wait for several channels at once (one per L2CPU tile, e.g. a batch split over tiles): reads every channel's
// req_seq, then polls the done_seq words round-robin until each equals its req_seq, all under ONE wall-clock bound of
// timeout_us. At the bound every channel that is not done gets 0xDEAD0000 | (req & 0xFFFF) in its own wait status
// word (the done ones are left alone) and the kernel returns: a dead tile never hangs the device, and the host sees
// which tile failed. One poll is one 64 B NoC read (~1 us); a done channel is not polled again.
// Runtime args: n (1..8), timeout_us, diag, then n x {l2_x, l2_y, base_hi, base_lo}.
// diag != 0: as l2cpu_wait.cpp, accumulated in channel 0's DIAG line 1 against channel 0's notify timestamp
// (DIAG line 0 word 0): [0] count, [1] sum(release - notify_end), [2] max, [3] sum(release - entry), [4] first
// notify_end, [5] last release, in wall-clock ticks. CB 0: >= 1 KiB L1 scratch.
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_noc.h"

constexpr uint32_t L2CPU_WAIT_ALL_MAX = 8;

void kernel_main() {
    uint32_t n = get_arg_val<uint32_t>(0), timeout_us = get_arg_val<uint32_t>(1), diag = get_arg_val<uint32_t>(2);
    if (n > L2CPU_WAIT_ALL_MAX) {
        n = L2CPU_WAIT_ALL_MAX;
    }
    uint32_t l1 = get_write_ptr(0);
    l2cpu_link c[L2CPU_WAIT_ALL_MAX];
    uint32_t req[L2CPU_WAIT_ALL_MAX];
    for (uint32_t i = 0; i < n; i++) {
        c[i] = l2cpu_link{
            noc_index,
            get_arg_val<uint32_t>(3 + 4 * i),
            get_arg_val<uint32_t>(4 + 4 * i),
            ((uint64_t)get_arg_val<uint32_t>(5 + 4 * i) << 32) | get_arg_val<uint32_t>(6 + 4 * i),
            l1};
    }
    uint32_t t_entry = get_timestamp_32b();
    for (uint32_t i = 0; i < n; i++) {
        req[i] = l2cpu_link_read32(c[i], L2CPU_LINK_OFF_REQ_SEQ, 0);
    }
    uint32_t t0 = get_timestamp_32b();
    uint32_t budget = timeout_us * L2CPU_WAIT_TICKS_PER_US;  // timeout_us <= 3,000,000 (32-bit tick wrap)
    uint32_t pending = (1u << n) - 1u;
    while (pending) {
        for (uint32_t i = 0; i < n; i++) {
            if ((pending >> i & 1u) && l2cpu_link_read32(c[i], L2CPU_LINK_OFF_DONE_SEQ, 1) == req[i]) {
                pending &= ~(1u << i);
            }
        }
        if (pending && get_timestamp_32b() - t0 > budget) {
            break;
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        if (pending >> i & 1u) {
            l2cpu_link_write32(c[i], L2CPU_LINK_OFF_WAIT_STATUS, L2CPU_WAIT_STATUS_TIMEOUT | (req[i] & 0xFFFFu), 3);
        }
    }
    uint32_t t_rel = get_timestamp_32b();
    if (diag) {
        uint32_t t_notify = l2cpu_link_read32(c[0], L2CPU_LINK_OFF_DIAG, 8);
        uint32_t l = l1 + 64 * 9;
        l2cpu_noc_read(c[0].noc, l, c[0].x, c[0].y, c[0].base + L2CPU_LINK_OFF_DIAG + 64, 64);
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
        l2cpu_noc_write(c[0].noc, l, c[0].x, c[0].y, c[0].base + L2CPU_LINK_OFF_DIAG + 64, 64);
        noc_async_write_barrier();
    }
}
