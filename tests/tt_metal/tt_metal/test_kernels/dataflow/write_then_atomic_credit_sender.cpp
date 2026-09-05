// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender half of the write->atomic-credit ordering probe (see the receiver for the
// hazard description). Publishes a payload with noc_async_writes_flushed() only --
// i.e. "request has left this NIU", NOT "committed at the destination" -- and then
// sends the credit as a remote atomic increment.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t recv_x = get_arg_val<uint32_t>(0);
    const uint32_t recv_y = get_arg_val<uint32_t>(1);
    const uint32_t payload_addr = get_arg_val<uint32_t>(2);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(3);
    const uint32_t data_sem_addr = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t ack_sem_addr = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t iters = get_arg_val<uint32_t>(6);
    const uint32_t src_addr = get_arg_val<uint32_t>(7);
    const uint32_t use_barrier = get_arg_val<uint32_t>(8);
    const uint32_t credit_vc = get_arg_val<uint32_t>(9);
    const uint32_t credit_first = get_arg_val<uint32_t>(10);  // detector self-test

    volatile tt_l1_ptr uint32_t* ack_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_sem_addr);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    const uint32_t words = payload_bytes / sizeof(uint32_t);

    const uint64_t dst_noc_addr = get_noc_addr(recv_x, recv_y, payload_addr);
    const uint64_t data_sem_noc_addr = get_noc_addr(recv_x, recv_y, data_sem_addr);

    for (uint32_t i = 1; i <= iters; ++i) {
        // Stamp the whole payload with this iteration's tag so a not-yet-committed
        // tail still holds iteration i-1 and is therefore detectable.
        for (uint32_t w = 0; w < words; ++w) {
            src[w] = i;
        }

        if (credit_first) {
            // Detector self-test: credit BEFORE the payload is even issued. The
            // receiver must see a stale tail essentially every iteration; if it does
            // not, the probe is not measuring anything.
            noc_semaphore_inc(data_sem_noc_addr, 1, noc_index, credit_vc);
            noc_async_write(src_addr, dst_noc_addr, payload_bytes);
            noc_async_write_barrier();
        } else {
            noc_async_write(src_addr, dst_noc_addr, payload_bytes);
            if (use_barrier) {
                noc_async_write_barrier();  // waits for ACKs: payload committed
            } else {
                noc_async_writes_flushed();  // waits only for requests SENT
            }
            noc_semaphore_inc(data_sem_noc_addr, 1, noc_index, credit_vc);
        }

        // Serialize with the receiver so the next iteration cannot overwrite the
        // payload while it is still being checked.
        noc_semaphore_wait(ack_sem, i);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
