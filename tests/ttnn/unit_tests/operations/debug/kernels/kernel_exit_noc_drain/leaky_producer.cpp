// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Producer of test_kernel_exit_noc_drain.py, first program, core A (BRISC).
// It waits, with a cap, until the launch message of the next program has been preloaded into this core's launch ring:
// fast dispatch preloads it, and the next kernel's start-up code snapshots the NoC counters (noc_local_state_init)
// before it waits for its go message. It then issues `count` non-posted semaphore increments to a word on core B and
// returns: in mode 0 at once, with the responses still in flight (the pattern under test); in mode 1 only after
// noc_async_atomic_barrier() (control). It leaves notes for the kernel the next program starts on this core.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t target_x = get_arg_val<uint32_t>(0);
    const uint32_t target_y = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t mode = get_arg_val<uint32_t>(3);
    const uint32_t wait_cap = get_arg_val<uint32_t>(4);
    const uint32_t note_offset = get_arg_val<uint32_t>(5);
    const uint32_t semaphore_offset = get_arg_val<uint32_t>(6);

    // The CB page sits at the same L1 address on every core of both programs.
    const uint32_t page = get_write_ptr(tt::CBIndex::c_0);
    const uint64_t semaphore = get_noc_addr(target_x, target_y, page + semaphore_offset);
    volatile tt_l1_ptr uint32_t* note = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + note_offset);

    tt_l1_ptr mailboxes_t* const mailboxes = reinterpret_cast<tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
    const uint32_t next = (mailboxes->launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
    uint32_t preloaded = 0;
    for (uint32_t polls = 0; polls < wait_cap; ++polls) {
        invalidate_l1_cache();
        if (mailboxes->launch[next].kernel_config.preload & DISPATCH_ENABLE_FLAG_PRELOAD) {
            preloaded = 1;
            break;
        }
    }
    note[0] = 0xE0170003;
    note[1] = preloaded;
    note[2] = mode;
    note[3] = count;

    for (uint32_t i = 0; i < count; ++i) {
        noc_semaphore_inc(semaphore, 1);
    }
    if (mode == 1) {
        noc_async_atomic_barrier();
    }
}
