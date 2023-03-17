#pragma once

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
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 3;

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
