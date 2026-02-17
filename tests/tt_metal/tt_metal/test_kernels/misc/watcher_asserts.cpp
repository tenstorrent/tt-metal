// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/ring_buffer.h"
#include "internal/firmware_common.h"

/*
 * A test for the assert feature.
*/
#if !defined(COMPILE_FOR_BRISC) && !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_DM)
#include "api/compute/common.h"
#endif

void kernel_main() {
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);
    uint32_t assert_type = get_arg_val<uint32_t>(2);

    // Conditionally enable using defines for each trisc
#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_DM))
    WATCHER_RING_BUFFER_PUSH(a);
    WATCHER_RING_BUFFER_PUSH(b);
#if defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC) or defined(COMPILE_FOR_ERISC)
    //For Erisc do a dummy increment since there is no worker kernel that would increment dispatch message addr to signal compute kernel completion.
    if (a == b) {
        //We will assert later. This kernel will hang.
        //Need to signal completion to dispatcher before hanging so that
        //Dispatcher Kernel is able to finish.
        //Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
        tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
        tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
        volatile go_msg_t* go_message_in = reinterpret_cast<volatile go_msg_t*>(&mailboxes->go_messages[mailboxes->go_message_index]);
        uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
        notify_dispatch_core_done(dispatch_addr, noc_index);
    }
#else
#if defined(TRISC0) or defined(TRISC1) or defined(TRISC2)
    volatile tt_l1_ptr uint8_t * const trisc_run = &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))
        ->subordinate_sync.map[COMPILE_FOR_TRISC + 1];  // first entry is for NCRISC
    *trisc_run = RUN_SYNC_MSG_DONE;
#endif
#endif

// For slow dispatch
#if !defined(COMPILE_FOR_ERISC) && defined(COMPILE_FOR_DM)
    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;
#endif
    ASSERT(a != b, static_cast<debug_assert_type_t>(assert_type));

#endif
}
