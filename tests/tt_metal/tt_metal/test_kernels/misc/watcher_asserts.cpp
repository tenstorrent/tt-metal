// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/ring_buffer.h"
#include "internal/firmware_common.h"
#include "api/compile_time_args.h"

/*
 * A test for the assert feature.
*/

void kernel_main() {
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);
    uint32_t assert_type = get_arg_val<uint32_t>(2);

    WATCHER_RING_BUFFER_PUSH(a);
    WATCHER_RING_BUFFER_PUSH(b);
    //For Erisc do a dummy increment since there is no worker kernel that would increment dispatch message addr to signal compute kernel completion.
    if (a == b) {
        //We will assert later. This kernel will hang.
        //Need to signal completion to dispatcher before hanging so that
        //Dispatcher Kernel is able to finish.
        //Device Close () requires fast dispatch kernels to finish.
        volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);

        // Signal completion to dispatcher before assert hangs the kernel
        // SD signaling: IDLE_ERISC, DRISC (SD only) require RUN_MSG_DONE
#if defined(COMPILE_FOR_IDLE_ERISC) or defined(COMPILE_FOR_DRISC)
        go_message_in->signal = RUN_MSG_DONE;
#else
        // FD: ACTIVE_ETH notifies dispatcher via NOC
        uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
        notify_dispatch_core_done(dispatch_addr, noc_index);
#endif
    }
    if (assert_type == DebugAssertHwFault && a==b) {
        uint32_t hw_assert_cause = get_arg_val<uint32_t>(3);
        volatile int32_t* p = (int32_t*)0xffffffffff000000;
        uint32_t tmp;
        switch (hw_assert_cause) {
            case 2: asm volatile(".word 0x00000000"); break; // illegal instruction
            case 4: asm volatile("lw %0, 0x2(x0)" : "=r"(tmp)); break; // load not aligned
            case 5: tmp = *p; break; // load access fault
            case 6: asm volatile("sw %0, 0x2(x0)" : "=r"(tmp)); break; // store not aligned
            case 7: *p = 0; break; // store access fault
            default: ASSERT(0, DebugAssertHwFault);
        }
    } else {
        ASSERT(a != b, static_cast<debug_assert_type_t>(assert_type));
    }
}
