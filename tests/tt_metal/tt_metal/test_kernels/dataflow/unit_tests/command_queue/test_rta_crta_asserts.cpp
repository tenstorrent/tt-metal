// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel for testing watcher RTA/CRTA bounds checking:
// 1. Writes rta_count, crta_count, and all arg values to L1 for validation
// 2. If MAX_RTA_IDX/MAX_CRTA_IDX defined: accesses that index to test bounds checking
// Supports both DM (BRISC/NCRISC) and compute (TRISC0) kernels

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/common.h"
#endif

extern uint32_t rta_count;
extern uint32_t crta_count;

// Helper to write RTA/CRTA metadata and values to L1
static FORCE_INLINE void write_args_to_l1(uint32_t l1_write_addr) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_write_addr);
    ptr[0] = rta_count;
    ptr[1] = crta_count;

    for (size_t i = 0; i < rta_count; i++) {
        ptr[i + 2] = get_arg_val<uint32_t>(i);
    }
    for (size_t i = 0; i < crta_count; i++) {
        ptr[i + rta_count + 2] = get_common_arg_val<uint32_t>(i);
    }
}

#ifndef COMPILE_FOR_TRISC
// Signal dispatcher completion before triggering assert (prevents Finish() hang)
static FORCE_INLINE void signal_dispatcher_completion() {
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
    uint64_t dispatch_addr = NOC_XY_ADDR(
        NOC_X(mailboxes->go_messages[mailboxes->go_message_index].master_x),
        NOC_Y(mailboxes->go_messages[mailboxes->go_message_index].master_y),
        DISPATCH_MESSAGE_ADDR +
            NOC_STREAM_REG_SPACE_SIZE * mailboxes->go_messages[mailboxes->go_message_index].dispatch_message_offset);
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,
        NOC_UNICAST_WRITE_VC,
        false,
        true);
}

void kernel_main() {
    write_args_to_l1(get_write_ptr(tt::CBIndex::c_0));

#if defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
    // Signal completion to BRISC before triggering assert
    signal_dispatcher_completion();
#endif

#ifdef MAX_RTA_IDX
    // Access RTA: this should have a watcher assert when MAX_RTA_IDX >= rta_count
    uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif
#ifdef MAX_CRTA_IDX
    // Access CRTA: this should have a watcher assert when MAX_CRTA_IDX >= crta_count
    uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif
}

#else  // Compute Kernel

namespace NAMESPACE {
void MAIN {
    UNPACK({
        // Pass the CB base address as a compile time arg
        write_args_to_l1(get_compile_time_arg_val(0));

#if defined(MAX_RTA_IDX) || defined(MAX_CRTA_IDX)
        // Signal completion to TRISC before triggering assert
        volatile tt_l1_ptr mailboxes_t* mailbox = reinterpret_cast<volatile tt_l1_ptr mailboxes_t*>(MEM_MAILBOX_BASE);
        volatile tt_l1_ptr subordinate_map_t* sync =
            reinterpret_cast<volatile tt_l1_ptr subordinate_map_t*>(&mailbox->subordinate_sync);
        sync->trisc0 = RUN_SYNC_MSG_DONE;
#endif

#ifdef MAX_RTA_IDX
        // Access RTA: this should have a watcher assert when MAX_RTA_IDX >= rta_count
        uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif
#ifdef MAX_CRTA_IDX
        // Access CRTA: this should have a watcher assert when MAX_CRTA_IDX >= crta_count
        uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif
    })
}
}  // namespace NAMESPACE
#endif
