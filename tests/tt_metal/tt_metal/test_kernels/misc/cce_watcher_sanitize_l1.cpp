// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/compile_time_args.h"
#include "internal/firmware_common.h"

// Hits debug_sanitize_l1_access on a CCE hart: an address past both the cached SRAM view and the
// uncached alias. Signal SD completion first so the hang is only the sanitizer trap.
void kernel_main() {
    constexpr uint32_t overflow_addr = get_compile_time_arg_val(0);

    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_in->signal = RUN_MSG_DONE;

    CoreLocalMem<uint32_t> l1_overflow_buffer(overflow_addr);
    l1_overflow_buffer[0] = 0xDEADBEEF;
}
