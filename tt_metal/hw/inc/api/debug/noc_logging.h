// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/debug/dprint_buffer.h"
#include "internal/hw_thread.h"

// Add option to skip noc logging for certain cores via a define.
#if defined(NOC_LOGGING_ENABLED) && !defined(SKIP_NOC_LOGGING)
// NOC transfer length histogram: 32 buckets (one per bit in the length field) per RISC, laid
// out as a flat uint32_t array starting at the base of the device print buffer. NOC logging is
// mutually exclusive with DPRINT (enforced host-side), so the buffer is fully available here.
constexpr uint32_t NOC_LOG_BUCKETS_PER_RISC = 32;

void log_noc_xfer(uint32_t len) {
    volatile tt_l1_ptr uint32_t* buf_base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_device_print_buffer());
    volatile tt_l1_ptr uint32_t* buf_ptr = buf_base + internal_::get_hw_thread_idx() * NOC_LOG_BUCKETS_PER_RISC;

    int highest_bit_position = 0;
    while (len >>= 1) {
        highest_bit_position++;
    }

    buf_ptr[highest_bit_position]++;
}

#define LOG_LEN(l) log_noc_xfer(l);
#define LOG_READ_LEN_FROM_STATE(noc_id) LOG_LEN(noc_debug_read_at_len_be(noc_id, NCRISC_RD_CMD_BUF));
#define LOG_WRITE_LEN_FROM_STATE(noc_id) LOG_LEN(noc_debug_read_at_len_be(noc_id, NCRISC_WR_CMD_BUF));

#else

#define LOG_LEN(l)
#define LOG_READ_LEN_FROM_STATE(noc_id)
#define LOG_WRITE_LEN_FROM_STATE(noc_id)
#endif
