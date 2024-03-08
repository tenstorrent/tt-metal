// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common_values.hpp"
#include "dev_mem_map.h"

/*
* This file contains addresses that are visible to both host and device compiled code.
*/

// TODO: these could be moved to even lower addresses -- 5 RISC-V hexes combined don't need 100 KB
constexpr static std::uint32_t BRISC_L1_ARG_BASE = 101 * 1024;
constexpr static std::uint32_t BRISC_L1_RESULT_BASE = 102 * 1024;
constexpr static std::uint32_t NCRISC_L1_ARG_BASE = 103 * 1024;
constexpr static std::uint32_t NCRISC_L1_RESULT_BASE = 104 * 1024;
constexpr static std::uint32_t TRISC_L1_ARG_BASE = 105 * 1024;
constexpr static std::uint32_t L1_ALIGNMENT = 16;

// config for 32 L1 buffers is at addr BUFFER_CONFIG_BASE
// 12 bytes for each buffer: (addr, size, size_in_tiles)
// addr and size are in 16B words (byte address >> 4)
// this is a total of 32 * 3 * 4 = 384B
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_BASE = 106 * 1024;
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_SIZE = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

// 4 uint32_t semaphores per core aligned to 16B
constexpr static std::uint32_t SEMAPHORE_BASE = CIRCULAR_BUFFER_CONFIG_BASE + CIRCULAR_BUFFER_CONFIG_SIZE;
constexpr static std::uint32_t NUM_SEMAPHORES = 4;
constexpr static std::uint32_t SEMAPHORE_SIZE = NUM_SEMAPHORES * L1_ALIGNMENT;

// Debug printer buffers - A total of 5*PRINT_BUFFER_SIZE starting at PRINT_BUFFER_NC address
constexpr static std::uint32_t PRINT_BUFFER_SIZE = 204; // per thread
constexpr static std::uint32_t PRINT_BUFFERS_COUNT = 5; // one for each thread
constexpr static std::uint32_t PRINT_BUFFER_NC = 107 * 1024; // NCRISC, address in bytes
constexpr static std::uint32_t PRINT_BUFFER_T0 = PRINT_BUFFER_NC + PRINT_BUFFER_SIZE; // TRISC0
constexpr static std::uint32_t PRINT_BUFFER_T1 = PRINT_BUFFER_T0 + PRINT_BUFFER_SIZE; // TRISC1
constexpr static std::uint32_t PRINT_BUFFER_T2 = PRINT_BUFFER_T1 + PRINT_BUFFER_SIZE; // TRISC2
constexpr static std::uint32_t PRINT_BUFFER_BR = PRINT_BUFFER_T2 + PRINT_BUFFER_SIZE; // BRISC

constexpr static std::uint32_t L1_UNRESERVED_BASE = 120 * 1024; // Start of unreserved space

// Breakpoint regions
constexpr static std::uint32_t NCRISC_BREAKPOINT = 109568;
constexpr static std::uint32_t TRISC0_BREAKPOINT = 109572;
constexpr static std::uint32_t TRISC1_BREAKPOINT = 109576;
constexpr static std::uint32_t TRISC2_BREAKPOINT = 109580;
constexpr static std::uint32_t BRISC_BREAKPOINT  = 109584;

// Frame pointer addresses for breakpoint... first create macros
// since I need them for inline assembly
#define NCRISC_SP_MACRO 109588
#define TRISC0_SP_MACRO 109592
#define TRISC1_SP_MACRO 109596
#define TRISC2_SP_MACRO 109600
#define BRISC_SP_MACRO  109604

// Brekapoint call line number macros
#define NCRISC_BP_LNUM_MACRO 109608
#define TRISC0_BP_LNUM_MACRO 109612
#define TRISC1_BP_LNUM_MACRO 109616
#define TRISC2_BP_LNUM_MACRO 109620
#define BRISC_BP_LNUM_MACRO  109624

constexpr static std::uint32_t NCRISC_SP = NCRISC_SP_MACRO;
constexpr static std::uint32_t TRISC0_SP = TRISC0_SP_MACRO;
constexpr static std::uint32_t TRISC1_SP = TRISC1_SP_MACRO;
constexpr static std::uint32_t TRISC2_SP = TRISC2_SP_MACRO;
constexpr static std::uint32_t BRISC_SP  = BRISC_SP_MACRO;

constexpr static std::uint32_t NCRISC_BP_LNUM = NCRISC_BP_LNUM_MACRO;
constexpr static std::uint32_t TRISC0_BP_LNUM = TRISC0_BP_LNUM_MACRO;
constexpr static std::uint32_t TRISC1_BP_LNUM = TRISC1_BP_LNUM_MACRO;
constexpr static std::uint32_t TRISC2_BP_LNUM = TRISC2_BP_LNUM_MACRO;
constexpr static std::uint32_t BRISC_BP_LNUM = BRISC_BP_LNUM_MACRO;

// Dispatch message address
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR = 110816;
constexpr static std::uint64_t DISPATCH_MESSAGE_REMOTE_SENDER_ADDR = 110848;

constexpr static std::uint32_t COMMAND_PTR_SHARD_IDX = 8;

// Command queue pointers
constexpr static uint32_t CQ_ISSUE_READ_PTR = 110944;
constexpr static uint32_t CQ_ISSUE_WRITE_PTR = CQ_ISSUE_READ_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_WRITE_PTR = CQ_ISSUE_WRITE_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_READ_PTR = CQ_COMPLETION_WRITE_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_LAST_EVENT = CQ_COMPLETION_READ_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_16B_SCRATCH = CQ_COMPLETION_LAST_EVENT + L1_ALIGNMENT;
constexpr static uint32_t EVENT_PTR = CQ_COMPLETION_16B_SCRATCH + L1_ALIGNMENT;

// Host addresses for dispatch
static constexpr uint32_t HOST_CQ_ISSUE_READ_PTR = 0;
static constexpr uint32_t HOST_CQ_COMPLETION_WRITE_PTR = 32;
static constexpr uint32_t HOST_CQ_FINISH_PTR = 64;
static constexpr uint32_t CQ_START = 96;

static constexpr uint32_t CQ_CONSUMER_CB_BASE = 111056;
// CB0
static constexpr uint32_t CQ_CONSUMER_CB0_ACK = CQ_CONSUMER_CB_BASE;
static constexpr uint32_t CQ_CONSUMER_CB0_RECV = CQ_CONSUMER_CB0_ACK + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB0_NUM_PAGES_BASE = CQ_CONSUMER_CB0_RECV + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB0_PAGE_SIZE = CQ_CONSUMER_CB0_NUM_PAGES_BASE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB0_TOTAL_SIZE = CQ_CONSUMER_CB0_PAGE_SIZE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB0_READ_PTR = CQ_CONSUMER_CB0_TOTAL_SIZE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB0_WRITE_PTR = CQ_CONSUMER_CB0_READ_PTR + L1_ALIGNMENT;

// CB1
static constexpr uint32_t CQ_CONSUMER_CB1_ACK = CQ_CONSUMER_CB0_WRITE_PTR + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_RECV = CQ_CONSUMER_CB1_ACK + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_NUM_PAGES_BASE = CQ_CONSUMER_CB1_RECV + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_PAGE_SIZE = CQ_CONSUMER_CB1_NUM_PAGES_BASE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_TOTAL_SIZE = CQ_CONSUMER_CB1_PAGE_SIZE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_READ_PTR = CQ_CONSUMER_CB1_TOTAL_SIZE + L1_ALIGNMENT;
static constexpr uint32_t CQ_CONSUMER_CB1_WRITE_PTR = CQ_CONSUMER_CB1_READ_PTR + L1_ALIGNMENT;

// DRAM write barrier
// Host writes (4B value) to and reads from this address across all L1s to ensure previous writes have been committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_ALIGNMENT = 32;
constexpr static std::uint32_t DRAM_BARRIER_SIZE = ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

constexpr static std::uint32_t DRAM_UNRESERVED_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE; // Start of unreserved space
