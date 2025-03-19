// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "dprint_buffer.h"

// Add option to skip noc logging for certain cores via a define.
#if defined(NOC_LOGGING_ENABLED) && !defined(SKIP_NOC_LOGGING)
void log_noc_xfer(uint32_t len) {
    // Hijack print buffer for noc logging data.
    volatile tt_l1_ptr uint32_t* buf_ptr =
        (volatile tt_l1_ptr uint32_t*)(reinterpret_cast<DebugPrintMemLayout*>(get_debug_print_buffer())->data);

    int highest_bit_position = 0;
    while (len >>= 1) {
        highest_bit_position++;
    }

    buf_ptr[highest_bit_position]++;
}

#define LOG_LEN(l) log_noc_xfer(l);
#define LOG_READ_LEN_FROM_STATE(noc_id) LOG_LEN(NOC_CMD_BUF_READ_REG(noc_id, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE));
#define LOG_WRITE_LEN_FROM_STATE(noc_id) LOG_LEN(NOC_CMD_BUF_READ_REG(noc_id, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE));

#else

#define LOG_LEN(l)
#define LOG_READ_LEN_FROM_STATE(noc_id)
#define LOG_WRITE_LEN_FROM_STATE(noc_id)
#endif
