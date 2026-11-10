// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of one stream (see test_dram_read_after_flag_cross_noc.py). Runs on NCRISC; the host picks its NoC.
// For k = 0..n-1 (seq = base + k):
//   1. wait until the flag in this core's L1 is >= 1 (the writer incremented it after its DRAM write was acked);
//   2. noc_async_read the page through this kernel's NoC (address resolved for that NoC), read barrier,
//      invalidate_l1_cache;
//   3. EXPECTED: every word of the page belongs to seq. If not, record the read and re-read (short delay)
//      until it does, counting re-reads and cycles;
//   4. flag -= 1, credit += 1 on the writer core.
// Result row (uint32, accumulated over launches) in this core's L1:
//   [0] txns lo [1] txns hi [2] mismatches [3] whole page stale [4] torn / other [5] unresolved
//   [6] max re-reads [7] max cycles [8] timeouts [9] flag > 1 at wait [10] final flag != 0 [11] logged
//   [12..15] page0 / page1 NoC address as read (lo, hi)
//   [16 + 8 i] mismatch i (< 8): seq, page, w0, w1, flag at wait, re-reads, cycles, words not of seq

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t read_cb = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr uint32_t words = page_bytes / 4;
    constexpr uint32_t kMaxRereads = 100000, kDelay = 64, kLog = 8;

    size_t a = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    const uint32_t page1 = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t send_x = get_arg_val<uint32_t>(a++);
    const uint32_t send_y = get_arg_val<uint32_t>(a++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(a++);
    const uint32_t credit_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    const uint32_t timeout = get_arg_val<uint32_t>(a++);
    const uint32_t num_stop = get_arg_val<uint32_t>(a++);  // hammer cores to release: (x, y)
    const uint32_t stop_w = get_arg_val<uint32_t>(a++);
    const uint32_t stop_r = get_arg_val<uint32_t>(a++);
    const uint32_t stop_args = a;

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    const uint32_t buf = get_write_ptr(read_cb);
    volatile tt_l1_ptr uint32_t* rw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint64_t credit_noc = get_noc_addr(send_x, send_y, credit_addr);
    const uint32_t base = *state;

    auto pat = [](uint32_t seq, uint32_t i) {
        return i == 0 ? seq : i == 1 ? ~seq : (seq * 0x9E3779B1u) ^ (i * 0x85EBCA77u);
    };
    auto read_page = [&](uint64_t addr) {
        noc_async_read(addr, buf, page_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
    };
    auto bad_words = [&](uint32_t seq) {
        uint32_t b = 0;
        for (uint32_t i = 0; i < words; ++i) {
            b += rw[i] != pat(seq, i) ? 1 : 0;
        }
        return b;
    };
    auto max_to = [&](uint32_t i, uint32_t v) { res[i] = v > res[i] ? v : res[i]; };

    uint32_t txn = 0, timed_out = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t seq = base + k;
        uint32_t fv, prev = get_timestamp_32b(), elapsed = 0;
        while ((invalidate_l1_cache(), fv = *flag) == 0) {
            const uint32_t now = get_timestamp_32b();
            elapsed += now - prev;
            prev = now;
            if (elapsed > timeout) {
                timed_out = 1;
                break;
            }
        }
        if (timed_out) {
            break;
        }
        res[9] += fv > 1 ? 1 : 0;

        const uint32_t page = (seq & 1) ? page1 : page0;
        const uint64_t addr = src.get_noc_addr(page);  // for this kernel's NoC
        read_page(addr);
        const uint32_t bad = bad_words(seq);
        if (bad != 0) {  // flag seen, data not visible
            const uint32_t w0 = rw[0], w1 = rw[1];
            const bool whole_old = bad_words(seq - 2) == 0;  // the previous write to this page
            const uint32_t t0 = get_timestamp_32b();
            uint32_t rr = 0;
            while (rr < kMaxRereads && bad_words(seq) != 0) {
                for (uint32_t s = get_timestamp_32b(); get_timestamp_32b() - s < kDelay;) {
                }
                read_page(addr);
                ++rr;
            }
            const uint32_t cyc = get_timestamp_32b() - t0;
            res[2] += 1;
            res[whole_old ? 3 : 4] += 1;
            res[5] += rr == kMaxRereads ? 1 : 0;
            max_to(6, rr);
            max_to(7, cyc);
            if (res[11] < kLog) {
                volatile tt_l1_ptr uint32_t* r = res + 16 + 8 * res[11];
                r[0] = seq, r[1] = page, r[2] = w0, r[3] = w1, r[4] = fv, r[5] = rr, r[6] = cyc, r[7] = bad;
                res[11] += 1;
            }
        }
        ++txn;
        noc_semaphore_inc(get_noc_addr(flag_addr), 0xFFFFFFFFu);  // flag -= 1
        noc_semaphore_inc(credit_noc, 1);
        noc_async_atomic_barrier();
    }
    invalidate_l1_cache();
    if (!timed_out) {
        if (*flag != 0) {
            res[10] += 1;
            *flag = 0;
        }
        *state = base + n;
    }
    for (uint32_t h = 0; h < num_stop; ++h) {  // release the hammer cores
        const uint32_t x = get_arg_val<uint32_t>(stop_args + 2 * h), y = get_arg_val<uint32_t>(stop_args + 2 * h + 1);
        noc_semaphore_inc(get_noc_addr(x, y, stop_w), 1);
        noc_semaphore_inc(get_noc_addr(x, y, stop_r), 1);
    }
    noc_async_atomic_barrier();
    const uint64_t t = (static_cast<uint64_t>(res[1]) << 32 | res[0]) + txn;
    res[0] = static_cast<uint32_t>(t);
    res[1] = static_cast<uint32_t>(t >> 32);
    res[8] += timed_out;
    const uint64_t e0 = src.get_noc_addr(page0), e1 = src.get_noc_addr(page1);
    res[12] = static_cast<uint32_t>(e0), res[13] = static_cast<uint32_t>(e0 >> 32);
    res[14] = static_cast<uint32_t>(e1), res[15] = static_cast<uint32_t>(e1 >> 32);
}
