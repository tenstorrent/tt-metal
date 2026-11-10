// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of one stream (see test_dram_read_after_flag_cross_noc.py). Runs on BRISC; the host picks its NoC.
// For k = 0..n-1 (seq = base + k), page = page0 / page1 alternately (P = 2):
//   1. k >= 1: wait until the reader has acknowledged k pages (credit >= k), so a page is never
//      rewritten before it was read;
//   2. fill the payload with seq (w0 = seq, w1 = ~seq, w2 = the page index, every other word a function of
//      seq and its index);
//   3. noc_async_write of the page to DRAM (non-posted; address resolved for this kernel's NoC);
//   4. noc_async_write_barrier: the write is ACKED;
//   5. noc_semaphore_inc of the reader's flag (+ atomic barrier).
// Result row (uint32, accumulated over launches) in this core's L1: [0] pages written lo [1] hi [2] timeouts
// [12..15] page0 / page1 NoC address as written (lo, hi).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t payload_cb = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t words = page_bytes / 4;

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    const uint32_t page1 = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t recv_x = get_arg_val<uint32_t>(a++);
    const uint32_t recv_y = get_arg_val<uint32_t>(a++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    const uint32_t timeout = get_arg_val<uint32_t>(a++);
    const uint32_t flag_delay = get_arg_val<uint32_t>(a++);  // experiment: cycles between the ack and the flag
    const uint32_t fence = get_arg_val<uint32_t>(a++);       // experiment: read the page back (own NoC) first

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t payload = get_write_ptr(payload_cb);
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload);
    const uint64_t flag_noc = get_noc_addr(recv_x, recv_y, flag_addr);
    const uint32_t base = *state;

    // credit counts this launch's acknowledgements (0 at launch start)
    auto wait_credit = [&](uint32_t need) -> bool {
        uint32_t prev = get_timestamp_32b(), elapsed = 0;  // low clock word only: wrap-safe
        while ((invalidate_l1_cache(), *credit) < need) {
            const uint32_t now = get_timestamp_32b();
            elapsed += now - prev;
            prev = now;
            if (elapsed > timeout) {
                return false;
            }
        }
        return true;
    };

    uint32_t sent = 0, timed_out = 0;
    for (uint32_t k = 0; k < n; ++k) {
        if (k >= 1 && !wait_credit(k)) {
            timed_out = 1;
            break;
        }
        const uint32_t seq = base + k;
        const uint32_t page = (seq & 1) ? page1 : page0;
        for (uint32_t i = 0; i < words; ++i) {
            pw[i] = i == 0 ? seq : i == 1 ? ~seq : i == 2 ? page : (seq * 0x9E3779B1u) ^ (i * 0x85EBCA77u);
        }
        noc_async_write(payload, dst.get_noc_addr(page), page_bytes);
        noc_async_write_barrier();  // acked
        if (fence) {  // read-back through the writer's own NoC / DRAM endpoint into the second payload slot
            noc_async_read(dst.get_noc_addr(page), payload + page_bytes, page_bytes);
            noc_async_read_barrier();
        }
        if (flag_delay) {
            for (uint32_t s = get_timestamp_32b(); get_timestamp_32b() - s < flag_delay;) {
            }
        }
        noc_semaphore_inc(flag_noc, 1);
        noc_async_atomic_barrier();
        ++sent;
    }
    if (!timed_out && !wait_credit(n)) {
        timed_out = 1;
    }
    if (!timed_out) {
        *credit = 0;
        *state = base + n;
    }
    const uint64_t t = (static_cast<uint64_t>(res[1]) << 32 | res[0]) + sent;
    res[0] = static_cast<uint32_t>(t);
    res[1] = static_cast<uint32_t>(t >> 32);
    res[2] += timed_out;
    const uint64_t e0 = dst.get_noc_addr(page0), e1 = dst.get_noc_addr(page1);
    res[12] = static_cast<uint32_t>(e0), res[13] = static_cast<uint32_t>(e0 >> 32);
    res[14] = static_cast<uint32_t>(e1), res[15] = static_cast<uint32_t>(e1 >> 32);
}
