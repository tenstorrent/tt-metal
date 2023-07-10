#pragma once

#include <cstdint>
#include "common_values.hpp"

/*
* This file contains addresses that are visible to both host and device compiled code.
*/

// TODO: these could be moved to even lower addresses -- 5 RISC-V hexes combined don't need 100 KB
constexpr static std::uint32_t RUNTIME_CONFIG_BASE = 100 * 1024;
constexpr static std::uint32_t BRISC_L1_ARG_BASE = 101 * 1024;
constexpr static std::uint32_t BRISC_L1_RESULT_BASE = 102 * 1024;
constexpr static std::uint32_t NCRISC_L1_ARG_BASE = 103 * 1024;
constexpr static std::uint32_t NCRISC_L1_RESULT_BASE = 104 * 1024;

// config for 32 L1 buffers is at addr BUFFER_CONFIG_BASE
// 12 bytes for each buffer: (addr, size, size_in_tiles)
// addr and size are in 16B words (byte address >> 4)
// this is a total of 32 * 3 * 4 = 384B
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_BASE = 105 * 1024;
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t CIRCULAR_BUFFER_CONFIG_SIZE = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

// 4 semaphores per core aligned to 16B
constexpr static std::uint32_t SEMAPHORE_BASE = CIRCULAR_BUFFER_CONFIG_BASE + CIRCULAR_BUFFER_CONFIG_SIZE;
constexpr static std::uint32_t NUM_SEMAPHORES = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_SEMAPHORE = 1;
constexpr static std::uint32_t SEMAPHORE_ALIGNMENT = 16;
constexpr static std::uint32_t ALIGNED_SIZE_PER_SEMAPHORE = (((UINT32_WORDS_PER_SEMAPHORE * sizeof(uint32_t)) + SEMAPHORE_ALIGNMENT - 1) / SEMAPHORE_ALIGNMENT) * SEMAPHORE_ALIGNMENT;
constexpr static std::uint32_t SEMAPHORE_SIZE = NUM_SEMAPHORES * ALIGNED_SIZE_PER_SEMAPHORE;

// Debug printer buffers - A total of 5*PRINT_BUFFER_SIZE starting at PRINT_BUFFER_NC address
constexpr static std::uint32_t PRINT_BUFFER_SIZE = 204; // per thread
constexpr static std::uint32_t PRINT_BUFFERS_COUNT = 5; // one for each thread
constexpr static std::uint32_t PRINT_BUFFER_NC = 106 * 1024; // NCRISC, address in bytes
constexpr static std::uint32_t PRINT_BUFFER_T0 = PRINT_BUFFER_NC + PRINT_BUFFER_SIZE; // TRISC0
constexpr static std::uint32_t PRINT_BUFFER_T1 = PRINT_BUFFER_T0 + PRINT_BUFFER_SIZE; // TRISC1
constexpr static std::uint32_t PRINT_BUFFER_T2 = PRINT_BUFFER_T1 + PRINT_BUFFER_SIZE; // TRISC2
constexpr static std::uint32_t PRINT_BUFFER_BR = PRINT_BUFFER_T2 + PRINT_BUFFER_SIZE; // BRISC
constexpr static std::uint32_t CONSTANT_REGISTER_VALUE = PRINT_BUFFER_BR + PRINT_BUFFER_SIZE + 4; // Producer/consumer sync scratch address

constexpr static std::uint32_t UNRESERVED_BASE = 120 * 1024; // Start of unreserved space

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

// Space allocated for op info, used in graph interpreter
constexpr static std::uint32_t OP_INFO_BASE_ADDR = 109628;
constexpr static std::uint32_t OP_INFO_SIZE      = 280; // So far, holds up to 10 ops

// Deassert reset from kernel dispatch
constexpr static std::uint32_t DEASSERT_RESET_SRC_L1_ADDR = 110752;
constexpr static std::uint32_t ASSERT_RESET_SRC_L1_ADDR   = 110784;

// Dispatch message address
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR = 110816;
constexpr static std::uint64_t DISPATCH_MESSAGE_REMOTE_SENDER_ADDR = 110848;
constexpr static std::uint32_t NOTIFY_HOST_KERNEL_COMPLETE_ADDR = 110912;

// Command queue pointers
constexpr static u32 CQ_READ_PTR = 110944;
constexpr static u32 CQ_WRITE_PTR = 110976;
constexpr static u32 CQ_READ_TOGGLE = 111008;
constexpr static u32 CQ_WRITE_TOGGLE = 111040;


// Information for deassert/assert reset
enum class TensixSoftResetOptions: std::uint32_t {
    NONE = 0,
    BRISC = ((std::uint32_t) 1 << 11),
    TRISC0 = ((std::uint32_t) 1 << 12),
    TRISC1 = ((std::uint32_t) 1 << 13),
    TRISC2 = ((std::uint32_t) 1 << 14),
    NCRISC = ((std::uint32_t) 1 << 18),
    STAGGERED_START = ((std::uint32_t) 1 << 31)
};

constexpr TensixSoftResetOptions operator|(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs) {
    return static_cast<TensixSoftResetOptions>(
        static_cast<uint32_t>(lhs) |
        static_cast<uint32_t>(rhs)
    );
}

constexpr TensixSoftResetOptions operator&(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs) {
    return static_cast<TensixSoftResetOptions>(
        static_cast<uint32_t>(lhs) &
        static_cast<uint32_t>(rhs)
    );
}

constexpr bool operator!=(TensixSoftResetOptions lhs, TensixSoftResetOptions rhs) {
    return
        static_cast<uint32_t>(lhs) !=
        static_cast<uint32_t>(rhs);
}

static constexpr TensixSoftResetOptions ALL_TRISC_SOFT_RESET = TensixSoftResetOptions::TRISC0 |
                                                               TensixSoftResetOptions::TRISC1 |
                                                               TensixSoftResetOptions::TRISC2;

static constexpr uint32_t TENSIX_SOFT_RESET_ADDR = 0xFFB121B0;

static constexpr TensixSoftResetOptions TENSIX_DEASSERT_SOFT_RESET = TensixSoftResetOptions::NCRISC |
                                                                     ALL_TRISC_SOFT_RESET |
                                                                     TensixSoftResetOptions::STAGGERED_START;

static constexpr TensixSoftResetOptions TENSIX_ASSERT_SOFT_RESET = TensixSoftResetOptions::BRISC |
                                                                   TensixSoftResetOptions::NCRISC |
                                                                   ALL_TRISC_SOFT_RESET;


static constexpr TensixSoftResetOptions ALL_TENSIX_SOFT_RESET = TensixSoftResetOptions::BRISC |
                                                                TensixSoftResetOptions::NCRISC |
                                                                TensixSoftResetOptions::STAGGERED_START |
                                                                ALL_TRISC_SOFT_RESET;

static constexpr TensixSoftResetOptions TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER = TensixSoftResetOptions::NCRISC |
                                                                                ALL_TRISC_SOFT_RESET;


// Host addresses for dispatch
static constexpr u32 HOST_CQ_READ_PTR = 0;
static constexpr u32 HOST_CQ_READ_TOGGLE_PTR = 32;
static constexpr u32 HOST_CQ_FINISH_PTR = 64;
static constexpr u32 CQ_START = 96;
