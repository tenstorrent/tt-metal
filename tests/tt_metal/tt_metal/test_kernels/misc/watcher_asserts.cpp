// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

void kernel_main() {
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);
    uint32_t assert_type = get_arg_val<uint32_t>(2);

#if defined(COMPILE_FOR_DM)
    constexpr uint32_t dm_id = get_compile_time_arg_val(0);
    uint64_t cpu_index = 0;
    asm volatile("csrr %0, mhartid" : "=r"(cpu_index));
    // On Quasar since all 8 kernels are launched: execute only the processor matching dm_id ; skip others
    if(dm_id != cpu_index)
        return;
#endif
    // Conditionally enable using defines for each trisc
#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC) or defined(COMPILE_FOR_ERISC) or defined(COMPILE_FOR_IDLE_ERISC) or defined(COMPILE_FOR_DM))
    WATCHER_RING_BUFFER_PUSH(a);
    WATCHER_RING_BUFFER_PUSH(b);
#if defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC) or defined(COMPILE_FOR_ERISC) or defined(COMPILE_FOR_IDLE_ERISC) or defined(COMPILE_FOR_DM)
    //For Erisc do a dummy increment since there is no worker kernel that would increment dispatch message addr to signal compute kernel completion.
    if (a == b) {
        //We will assert later. This kernel will hang.
        //Need to signal completion to dispatcher before hanging so that
        //Dispatcher Kernel is able to finish.
        //Device Close () requires fast dispatch kernels to finish.
        volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);

        // Signal completion to dispatcher before assert hangs the kernel
        // SD signaling: IDLE_ERISC (all archs) and Quasar DM require RUN_MSG_DONE
        // TODO: Remove COMPILE_FOR_DM once FD is enabled on Quasar
#if defined(COMPILE_FOR_IDLE_ERISC) or defined(COMPILE_FOR_DM)
        go_message_in->signal = RUN_MSG_DONE;
#else
        // FD: ACTIVE_ETH, BRISC, NCRISC notify dispatcher via NOC
        uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
        notify_dispatch_core_done(dispatch_addr, noc_index);
#endif
    }
#else
#if defined(COMPILE_FOR_TRISC)
    volatile tt_l1_ptr uint8_t * const trisc_run = &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))
        ->subordinate_sync.map[COMPILE_FOR_TRISC + 1];  // first entry is for NCRISC
    *trisc_run = RUN_SYNC_MSG_DONE;
#endif
#endif

    ASSERT(a != b, static_cast<debug_assert_type_t>(assert_type));
#endif
}
