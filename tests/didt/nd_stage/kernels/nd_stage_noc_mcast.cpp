// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NoC probe in the matmul mcast style: the first core of each row multicasts a buffer to the rest of the
// row, alternating a pattern and its complement so every bit toggles each round, and the receivers compare
// what arrived against their own local copy of the same pattern. Handshake as in the matmul readers:
// receivers bump the sender's ready semaphore, the sender multicasts data then the round number.
//
// Runtime args: is_sender, n_rounds, buf_bytes, pattern_addr, rx_addr, num_receivers, mcast_x0, mcast_y0,
//               mcast_x1, mcast_y1, sender_x, sender_y, check_every, result_addr, core_id

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "nd_stage_result.h"

void kernel_main() {
    const uint32_t is_sender = get_arg_val<uint32_t>(0);
    const uint32_t n_rounds = get_arg_val<uint32_t>(1);
    const uint32_t buf_bytes = get_arg_val<uint32_t>(2);
    const uint32_t pattern_addr = get_arg_val<uint32_t>(3);
    const uint32_t rx_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_receivers = get_arg_val<uint32_t>(5);
    const uint32_t mcast_x0 = get_arg_val<uint32_t>(6);
    const uint32_t mcast_y0 = get_arg_val<uint32_t>(7);
    const uint32_t mcast_x1 = get_arg_val<uint32_t>(8);
    const uint32_t mcast_y1 = get_arg_val<uint32_t>(9);
    const uint32_t sender_x = get_arg_val<uint32_t>(10);
    const uint32_t sender_y = get_arg_val<uint32_t>(11);
    const uint32_t check_every = get_arg_val<uint32_t>(12);
    const uint32_t result_addr = get_arg_val<uint32_t>(13);
    const uint32_t core_id = get_arg_val<uint32_t>(14);

    const uint32_t ready_sem = get_semaphore(0);
    const uint32_t valid_sem = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_sem);

    nd_stage::Result res(result_addr, nd_stage::MAGIC_NOC, core_id);

    if (is_sender) {
        const uint64_t data_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, rx_addr);
        const uint64_t valid_mcast = get_noc_multicast_addr(mcast_x0, mcast_y0, mcast_x1, mcast_y1, valid_sem);
        for (uint32_t r = 0; r < n_rounds; ++r) {
            noc_semaphore_wait(ready, num_receivers);
            noc_semaphore_set(ready, 0);
            noc_async_write_multicast(pattern_addr + (r & 1) * buf_bytes, data_mcast, buf_bytes, num_receivers, true);
            // The round number travels behind the data on the same path, so it lands after it.
            *valid = r + 1;
            noc_semaphore_set_multicast(valid_sem, valid_mcast, num_receivers);
            res.end_iteration(false, false, r);
        }
        noc_async_write_barrier();
    } else {
        const uint64_t sender_ready = get_noc_addr(sender_x, sender_y, ready_sem);
        const uint32_t words = buf_bytes / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* rx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rx_addr);
        for (uint32_t r = 0; r < n_rounds; ++r) {
            noc_semaphore_inc(sender_ready, 1);
            noc_semaphore_wait(valid, r + 1);
            const bool check_it = r == 0 || r + 1 == n_rounds || r % check_every == 0;
            bool bad = false;
            if (check_it) {
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* want =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pattern_addr + (r & 1) * buf_bytes);
                for (uint32_t w = 0; w < words; ++w) {
                    const uint32_t got = rx[w];
                    if (got != want[w]) {
                        bad = true;
                        res.record_mismatch<0>(r, w, got, want[w], 0);
                    }
                }
            }
            res.end_iteration(check_it, bad, r);
        }
        noc_async_atomic_barrier();
    }
    res.finish();
}
