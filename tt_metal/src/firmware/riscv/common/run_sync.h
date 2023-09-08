// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// run_sync.h
//
// Contains the mechanism to sync the "run" and "done" messages across
// brisc/ncrisc/trisc

#pragma once

// 0x80808000 is a micro-optimization, calculated with 1 riscv insn
constexpr uint32_t RUN_SYNC_MESSAGE_INIT = 0x40;
constexpr uint32_t RUN_SYNC_MESSAGE_GO   = 0x80;
constexpr uint32_t RUN_SYNC_MESSAGE_DONE = 0;
constexpr uint32_t RUN_SYNC_MESSAGE_ALL_TRISCS_GO = 0x80808000;
constexpr uint32_t RUN_SYNC_MESSAGE_ALL_GO = 0x80808080;
constexpr uint32_t RUN_SYNC_MESSAGE_ALL_SLAVES_DONE = 0;

struct run_sync_message_t {
    union {
        volatile uint32_t all;
        struct {
            volatile uint8_t ncrisc; // ncrisc must come first, see ncrisc-halt.S
            volatile uint8_t trisc0;
            volatile uint8_t trisc1;
            volatile uint8_t trisc2;
        };
    };
};
//static_assert((uint32_t)(&(((struct slave_run_t *)0)->ncrisc)) == 0);
