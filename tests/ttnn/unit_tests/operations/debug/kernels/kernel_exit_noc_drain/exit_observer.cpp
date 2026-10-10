// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Observer of test_kernel_exit_noc_drain.py, second program, core A (BRISC), enqueued directly behind the producer.
// Its first statements read the live atomic-response counter and the software snapshot its start-up code took (the
// value noc_async_atomic_barrier() compares against for equality). It then polls the counter: any increase is a
// response that arrived after the snapshot although this kernel has issued nothing. It evaluates the barrier
// predicate once with nothing issued, then after `own_count` atomics of its own, with a cap instead of the API's
// unbounded loop. The observation waits are capped; an outer process timeout bounds the test as a whole.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t live_at_start = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);
    const uint32_t snapshot = noc_nonposted_atomics_acked[noc_index];

    const uint32_t polls = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t result_offset = get_arg_val<uint32_t>(2);
    const uint32_t note_offset = get_arg_val<uint32_t>(3);
    const uint32_t own_count = get_arg_val<uint32_t>(4);
    const uint32_t cap = get_arg_val<uint32_t>(5);
    const uint32_t target_x = get_arg_val<uint32_t>(6);
    const uint32_t target_y = get_arg_val<uint32_t>(7);
    const uint32_t semaphore_offset = get_arg_val<uint32_t>(8);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, 2048);

    const uint32_t page = get_write_ptr(tt::CBIndex::c_0);
    volatile tt_l1_ptr uint32_t* note = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + note_offset);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + result_offset);

    uint32_t live_last = live_at_start;
    for (uint32_t p = 0; p < polls; ++p) {
        live_last = NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED);
    }
    // The predicate noc_async_atomic_barrier() and noc_async_full_barrier() spin on, with nothing issued here.
    const uint32_t idle_predicate = ncrisc_noc_nonposted_atomics_flushed(noc_index) ? 1 : 0;

    uint32_t own_barrier_polls = 0xFFFFFFFF;  // polls until the predicate held after this kernel's own atomics
    if (own_count) {
        const uint64_t semaphore = get_noc_addr(target_x, target_y, page + semaphore_offset);
        for (uint32_t i = 0; i < own_count; ++i) {
            noc_semaphore_inc(semaphore, 1);
        }
        for (uint32_t p = 0; p < cap; ++p) {
            if (ncrisc_noc_nonposted_atomics_flushed(noc_index)) {
                own_barrier_polls = p;
                break;
            }
        }
        // Let this kernel's own responses come back (capped) so it leaves nothing in flight.
        const uint32_t want = live_last + own_count;
        for (uint32_t p = 0; p < cap; ++p) {
            if (static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, NIU_MST_ATOMIC_RESP_RECEIVED) - want) >= 0) {
                break;
            }
        }
    }

    for (uint32_t i = 0; i < 16; ++i) {
        result[i] = 0;
    }
    result[0] = 0xE0170004;
    result[1] = note[0];  // producer magic
    result[2] = note[1];  // producer saw this program's launch message preloaded
    result[3] = note[2];  // producer mode
    result[4] = note[3];  // producer atomic count
    result[5] = snapshot;
    result[6] = live_at_start;
    result[7] = live_last;
    result[8] = idle_predicate;
    result[9] = own_barrier_polls;
    result[10] = noc_index;
    result[11] = 0xD0E5;
    {
        // Complete the result stores before the NoC reads the block: load the last word written and consume it.
        uint32_t sink;
        asm volatile("fence\n\tlw %0, 44(%1)\n\tand x0, x0, %0" : "=&r"(sink) : "r"(result) : "memory");
    }
    noc_async_write(page + result_offset, out.get_noc_addr(0), 2048);
    noc_async_write_barrier();
}
