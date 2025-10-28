// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensix.h"

enum ThreadId {
    BRISCThreadId = 0,
    UnpackThreadId = 1,
    MathThreadId = 2,
    PackThreadId = 3,
};

volatile uint32_t tt_reg_ptr* mailbox_base[4] = {
    reinterpret_cast<volatile uint32_t tt_reg_ptr*>(TENSIX_MAILBOX0_BASE),
    reinterpret_cast<volatile uint32_t tt_reg_ptr*>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint32_t tt_reg_ptr*>(TENSIX_MAILBOX2_BASE),
    reinterpret_cast<volatile uint32_t tt_reg_ptr*>(TENSIX_MAILBOX3_BASE)};

inline void mailbox_write(const ThreadId thread, const uint32_t data) { mailbox_base[thread][0] = data; }

// Blocking read
inline uint32_t mailbox_read(const ThreadId thread) { return mailbox_base[thread][0]; }

inline bool mailbox_not_empty(const ThreadId thread) { return mailbox_base[thread][1] > 0; }
