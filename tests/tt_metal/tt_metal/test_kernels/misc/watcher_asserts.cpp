// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"
#include "debug/ring_buffer.h"

/*
 * A test for the assert feature.
*/
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif

    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);

    // Conditionally enable using defines for each trisc
#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or \
    (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or \
    (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC))
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
        uint64_t dispatch_addr =
            NOC_XY_ADDR(NOC_X(mailboxes->go_message.master_x),
                        NOC_Y(mailboxes->go_message.master_y), DISPATCH_MESSAGE_ADDR);
        noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
    }
#else
#if defined(TRISC0) or defined(TRISC1) or defined(TRISC2)
#define GET_TRISC_RUN_EVAL(x, t) x##t
#define GET_TRISC_RUN(x, t) GET_TRISC_RUN_EVAL(x, t)
    volatile tt_l1_ptr uint8_t * const trisc_run = &GET_TRISC_RUN(((tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE))->slave_sync.trisc, COMPILE_FOR_TRISC);
    *trisc_run = RUN_SYNC_MSG_DONE;
#endif
#endif

    ASSERT(a != b);
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
}
#else
}
}
#endif
