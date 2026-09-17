// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "internal/firmware_common.h"
#include "internal/hw_thread.h"
#include "hostdev/dev_msgs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/debug/stack_usage.h"
#include <kernel_includes.hpp>
#include "api/kernel_thread_globals.h"
#include "api/debug/waypoint.h"

// Per-processor kernel thread info for Quasar (set from kernel_config before kernel runs)
thread_local uint32_t num_sw_threads __attribute__((used));
thread_local uint32_t my_thread_id __attribute__((used));

extern "C" [[gnu::section(".start")]]
uint32_t _start() {
#if defined(DEBUG_NULL_KERNELS)
    mark_stack_usage();
    while ((*GET_MAILBOX_ADDRESS_DEV(go_messages))[(*GET_MAILBOX_ADDRESS_DEV(go_message_index))].signal != RUN_MSG_GO) {
    }
#else
    // Raw read: hw_thread_idx has not been filled yet, and do_thread_crt1() below zeroes the .tbss
    // it lives in, so caching it any earlier would just be discarded.
    uint32_t hartid = internal_::read_hw_thread_idx();

    // Obtain launch message from mailbox and derive thread 0 (lowest hartid with same kernel).
    uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    launch_msg_t tt_l1_ptr* launch_msg = &(*GET_MAILBOX_ADDRESS_DEV(launch))[launch_idx];
    uint32_t my_kt = launch_msg->kernel_config.kernel_text_offset[hartid];
    uint32_t thread_0_hartid = hartid;
    if (launch_msg->kernel_config.enables & (1u << hartid)) {
        for (uint32_t j = 0; j < MaxDMProcessorsPerCoreType; j++) {
            if ((launch_msg->kernel_config.enables & (1u << j)) &&
                launch_msg->kernel_config.kernel_text_offset[j] == my_kt) {
                thread_0_hartid = j;
                break;
            }
        }
    }

    extern uint32_t __tdata_lma[];
    extern uint32_t __ldm_tdata_start[];
    extern uint32_t __ldm_tdata_end[];

    // Materialize __tdata_lma's address exactly once. The two references below (do_crt1 in the
    // thread-0 branch and do_thread_crt1) otherwise emit two R_RISCV_HI20 relocations for
    // __tdata_lma. On Quasar the XIP relocation pass (ElfFile::Impl::XIPify) pairs HI20<->LO12
    // heuristically (each LO12 to the nearest preceding HI20); when the second HI20 lands in a loop
    // tail with its LO12 reached via a back-edge it is left orphaned, throwing "R_RISCV_HI20
    // relocation ... has no matching R_RISCV_LO12". The asm barrier makes the pointer opaque so the
    // compiler keeps a single materialization (one HI20) and reuses the register/spill instead of
    // re-emitting a lui/addi pair.
    uint32_t* tdata_lma = __tdata_lma;
    asm volatile("" : "+r"(tdata_lma));

    if (hartid == thread_0_hartid) {
        do_crt1(&tdata_lma[__ldm_tdata_end - __ldm_tdata_start]);
        (*GET_MAILBOX_ADDRESS_DEV(shared_globals_ready))[hartid] = SHARED_GLOBALS_READY_GO;
    }

    do_thread_crt1(tdata_lma);
    // .tbss has been zeroed: cache this thread's hw index.
    internal_::init_hw_thread_idx();

    // Wait until first thread in the group has set its slot to GO.
    while ((*GET_MAILBOX_ADDRESS_DEV(shared_globals_ready))[thread_0_hartid] != SHARED_GLOBALS_READY_GO) {
    }

    // Setup after CRT so thread_local writes land in this hart's TLS.
    num_sw_threads = launch_msg->kernel_config.num_sw_threads[hartid];
    my_thread_id = launch_msg->kernel_config.kernel_thread_id[hartid];

    // Paint stack after all thread_local writes and CRT init are done.
    mark_stack_usage();

    {
        DeviceZoneScopedMainChildN("DRISC-KERNEL");
        WAYPOINT("K");
        kernel_main();
        WAYPOINT("KD");
    }
#endif
    return measure_stack_usage();
}
