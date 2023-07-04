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
constexpr static std::uint32_t PROFILER_L1_MARKER_UINT32_SIZE = 2;
constexpr static std::uint32_t PROFILER_L1_MARKER_BYTES_SIZE = PROFILER_L1_MARKER_UINT32_SIZE * sizeof(uint32_t);

constexpr static std::uint32_t PROFILER_L1_PROGRAM_ID_COUNT = 2;
constexpr static std::uint32_t PROFILER_L1_GUARANTEED_MARKER_COUNT = 4;

constexpr static std::uint32_t PROFILER_L1_OPTIONAL_MARKER_COUNT = 250;
constexpr static std::uint32_t PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT = 2;

constexpr static std::uint32_t PROFILER_L1_VECTOR_SIZE = (PROFILER_L1_OPTIONAL_MARKER_COUNT + PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_PROGRAM_ID_COUNT) * PROFILER_L1_MARKER_UINT32_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_SIZE = PROFILER_L1_VECTOR_SIZE  * sizeof(uint32_t);

constexpr static std::uint32_t PROFILER_L1_BUFFER_BR = 88 * 1024;
constexpr static std::uint32_t PROFILER_L1_BUFFER_NC = PROFILER_L1_BUFFER_BR + PROFILER_L1_BUFFER_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_T0 = PROFILER_L1_BUFFER_NC + PROFILER_L1_BUFFER_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_T1 = PROFILER_L1_BUFFER_T0 + PROFILER_L1_BUFFER_SIZE;
constexpr static std::uint32_t PROFILER_L1_BUFFER_T2 = PROFILER_L1_BUFFER_T1 + PROFILER_L1_BUFFER_SIZE;

constexpr static std::uint32_t PROFILER_L1_END_ADDRESS = PROFILER_L1_BUFFER_T2 + PROFILER_L1_BUFFER_SIZE;

constexpr static std::uint32_t PROFILER_OP_SUPPORT_COUNT = 1000;
constexpr static std::uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC = PROFILER_L1_MARKER_UINT32_SIZE * (PROFILER_L1_PROGRAM_ID_COUNT +  PROFILER_L1_GUARANTEED_MARKER_COUNT + PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) * PROFILER_OP_SUPPORT_COUNT;
constexpr static std::uint32_t PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC * sizeof(uint32_t);
constexpr static std::uint32_t PROFILER_RISC_COUNT = 5;

static_assert (PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC > PROFILER_L1_BUFFER_SIZE);

constexpr static std::uint32_t BRISC_L1_ARG_BASE = 98 * 1024;
constexpr static std::uint32_t BRISC_L1_RESULT_BASE = 99 * 1024;
constexpr static std::uint32_t NCRISC_L1_ARG_BASE = 100 * 1024;
constexpr static std::uint32_t NCRISC_L1_RESULT_BASE = 101 * 1024;
constexpr static std::uint32_t TRISC_L1_ARG_BASE = 102 * 1024;
constexpr static std::uint32_t IDLE_ERISC_L1_ARG_BASE = 32 * 1024;
constexpr static std::uint32_t IDLE_ERISC_L1_RESULT_BASE = 33 * 1024;
constexpr static std::uint32_t L1_ALIGNMENT = 16;

// config for 32 L1 buffers is at addr BUFFER_CONFIG_BASE
// 12 bytes for each buffer: (addr, size, size_in_tiles)
// addr and size are in 16B words (byte address >> 4)
// this is a total of 32 * 3 * 4 = 384B
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_BASE = 103 * 1024;
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_SIZE = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

constexpr static std::uint32_t PROFILER_L1_CONTROL_VECTOR_SIZE = 32;
constexpr static std::uint32_t PROFILER_L1_CONTROL_BUFFER_SIZE = PROFILER_L1_CONTROL_VECTOR_SIZE * sizeof(uint32_t);
constexpr static std::uint32_t PROFILER_L1_BUFFER_CONTROL = CIRCULAR_BUFFER_CONFIG_BASE + CIRCULAR_BUFFER_CONFIG_SIZE;

// 4 uint32_t semaphores per core aligned to 16B
constexpr static std::uint32_t SEMAPHORE_BASE = PROFILER_L1_BUFFER_CONTROL + PROFILER_L1_CONTROL_BUFFER_SIZE;
constexpr static std::uint32_t NUM_SEMAPHORES = 4;
constexpr static std::uint32_t SEMAPHORE_SIZE = NUM_SEMAPHORES * L1_ALIGNMENT;

// Debug printer buffers - A total of 5*PRINT_BUFFER_SIZE starting at PRINT_BUFFER_NC address
constexpr static std::uint32_t PRINT_BUFFER_START = SEMAPHORE_BASE + SEMAPHORE_SIZE; // per thread
constexpr static std::uint32_t PRINT_BUFFER_MAX_SIZE = 1024; // per thread

constexpr static std::uint32_t PRINT_BUFFER_SIZE = 204; // per thread
constexpr static std::uint32_t PRINT_BUFFERS_COUNT = 5; // one for each thread
constexpr static std::uint32_t PRINT_BUFFER_NC = PRINT_BUFFER_START; // NCRISC, address in bytes
constexpr static std::uint32_t PRINT_BUFFER_T0 = PRINT_BUFFER_NC + PRINT_BUFFER_SIZE; // TRISC0
constexpr static std::uint32_t PRINT_BUFFER_T1 = PRINT_BUFFER_T0 + PRINT_BUFFER_SIZE; // TRISC1
constexpr static std::uint32_t PRINT_BUFFER_T2 = PRINT_BUFFER_T1 + PRINT_BUFFER_SIZE; // TRISC2
constexpr static std::uint32_t PRINT_BUFFER_BR = PRINT_BUFFER_T2 + PRINT_BUFFER_SIZE; // BRISC
constexpr static std::uint32_t PRINT_BUFFER_IDLE_ER = PRINT_BUFFER_START; // Idle ERISC

// Debug ring buffer, shared between all cores
constexpr static std::uint32_t RING_BUFFER_ADDR = PRINT_BUFFER_START + PRINT_BUFFER_MAX_SIZE;
constexpr static std::uint32_t RING_BUFFER_SIZE = 128;

// Breakpoint regions
constexpr static std::uint32_t NCRISC_BREAKPOINT = PRINT_BUFFER_START;
constexpr static std::uint32_t TRISC0_BREAKPOINT = NCRISC_BREAKPOINT + 4;
constexpr static std::uint32_t TRISC1_BREAKPOINT = TRISC0_BREAKPOINT + 4;
constexpr static std::uint32_t TRISC2_BREAKPOINT = TRISC1_BREAKPOINT + 4;
constexpr static std::uint32_t BRISC_BREAKPOINT  = TRISC2_BREAKPOINT + 4;

// Frame pointer addresses for breakpoint... first create macros
// since I need them for inline assembly
#define NCRISC_SP_MACRO (BRISC_BREAKPOINT + 4)
#define TRISC0_SP_MACRO (NCRISC_SP_MACRO + 4)
#define TRISC1_SP_MACRO (TRISC0_SP_MACRO + 4)
#define TRISC2_SP_MACRO (TRISC1_SP_MACRO + 4)
#define BRISC_SP_MACRO  (TRISC2_SP_MACRO + 4)

// Brekapoint call line number macros
#define NCRISC_BP_LNUM_MACRO (BRISC_SP_MACRO + 4)
#define TRISC0_BP_LNUM_MACRO (NCRISC_BP_LNUM_MACRO + 4)
#define TRISC1_BP_LNUM_MACRO (TRISC0_BP_LNUM_MACRO + 4)
#define TRISC2_BP_LNUM_MACRO (TRISC1_BP_LNUM_MACRO + 4)
#define BRISC_BP_LNUM_MACRO  (TRISC2_BP_LNUM_MACRO + 4)

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
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR = RING_BUFFER_ADDR + RING_BUFFER_SIZE;
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR_RESERVATION = 32;
constexpr static std::uint64_t DISPATCH_MESSAGE_REMOTE_SENDER_ADDR = DISPATCH_MESSAGE_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION;

constexpr static std::uint32_t COMMAND_PTR_SHARD_IDX = 8;

// Command queue pointers
constexpr static uint32_t CQ_ISSUE_READ_PTR = DISPATCH_MESSAGE_REMOTE_SENDER_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION;
constexpr static uint32_t CQ_ADDR_RESERVATION = 96;
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

static constexpr uint32_t CQ_CONSUMER_CB_BASE = CQ_ISSUE_READ_PTR + CQ_ADDR_RESERVATION;
static constexpr uint32_t CQ_CONSUMER_CB_ADDR_RESERVATION = 256;
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

constexpr static std::uint32_t L1_UNRESERVED_BASE = CQ_CONSUMER_CB_BASE + CQ_CONSUMER_CB_ADDR_RESERVATION; //CQ_CONSUMER_CB_BASE + 16 * L1_ALIGNMENT; // Start of unreserved space
constexpr static std::uint32_t ERISC_L1_UNRESERVED_BASE = L1_UNRESERVED_BASE; // Start of unreserved space

// DRAM write barrier
// Host writes (4B value) to and reads from this address across all L1s to ensure previous writes have been committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_ALIGNMENT = 32;
constexpr static std::uint32_t DRAM_BARRIER_SIZE = ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

constexpr static std::uint32_t DRAM_UNRESERVED_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE; // Start of unreserved space
