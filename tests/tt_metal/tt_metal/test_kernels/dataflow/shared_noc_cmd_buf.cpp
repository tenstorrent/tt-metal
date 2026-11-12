// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One of the two data movement kernels of a program whose kernels share one dedicated NOC
// (DataMovement1XXConfig::dedicated_noc_sharing). Copies `size` bytes between two L1 buffers on this core over
// the NOC, as a read (do_write = 0, the read command buffer) or a write (do_write = 1, the write command
// buffer). With both kernels writing, both program the write command buffer and the watcher's shared-NOC
// check trips.
//
// The two kernels hand-shake through flags in L1, never through the NOC:
//   - wait_for_other_first: issue only after the other kernel's transfer is done, so when both write, the
//     watcher error lands on this kernel.
//   - Before returning, each kernel re-syncs its NoC counters once both transfers are done: the NOC's
//     hardware counters count both kernels' traffic, which the end-of-kernel checks compare with each RISC's
//     own count.
// signal_done_early: report the core done to dispatch up front, for a test in which this kernel hangs on the
// watcher error afterwards.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/watcher_test_helpers.h"

void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);
    constexpr uint32_t size = get_arg(args::size);
    constexpr uint32_t my_flag_addr = get_arg(args::my_flag_addr);
    constexpr uint32_t other_flag_addr = get_arg(args::other_flag_addr);
    constexpr bool do_write = get_arg(args::do_write) != 0;
    constexpr bool wait_for_other_first = get_arg(args::wait_for_other_first) != 0;
    constexpr bool signal_done_early = get_arg(args::signal_done_early) != 0;

    volatile tt_l1_ptr uint32_t* my_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_flag_addr);
    volatile tt_l1_ptr uint32_t* other_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(other_flag_addr);

    if constexpr (signal_done_early) {
#if defined(WATCHER_KERNEL_SLOW_DISPATCH)
        // Slow dispatch has no dispatcher to notify; the host polls this signal.
        GET_MAILBOX_ADDRESS_DEV(go_messages[0].slot[0])->signal = RUN_MSG_DONE;
#else
        signal_completion_before_hang();
#endif
    }

    if constexpr (wait_for_other_first) {
        while (*other_flag == 0);
    }

    if constexpr (do_write) {
        noc_async_write(src_addr, get_noc_addr(dst_addr), size);
        noc_async_write_barrier();
    } else {
        noc_async_read(get_noc_addr(src_addr), dst_addr, size);
        noc_async_read_barrier();
    }

    *my_flag = 1;
    while (*other_flag == 0);
    ncrisc_noc_counters_init();
}
