// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN };
enum PerfCounterType : uint8_t {
    UNDEF = 0,
    // FPU Group (3 counters)
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group (11 counters)
    DATA_HAZARD_STALLS_MOVD2A,
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_AVAILABLE,
    SRCA_WRITE_AVAILABLE,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    SRCB_WRITE,
    SRCA_WRITE,
    // TDMA_PACK Group (3 counters)
    PACKER_DEST_READ_AVAILABLE,
    PACKER_BUSY,
    AVAILABLE_MATH,
    // INSTRN_THREAD Group (61 counters)
    CFG_INSTRN_AVAILABLE_0,
    CFG_INSTRN_AVAILABLE_1,
    CFG_INSTRN_AVAILABLE_2,
    SYNC_INSTRN_AVAILABLE_0,
    SYNC_INSTRN_AVAILABLE_1,
    SYNC_INSTRN_AVAILABLE_2,
    THCON_INSTRN_AVAILABLE_0,
    THCON_INSTRN_AVAILABLE_1,
    THCON_INSTRN_AVAILABLE_2,
    XSEARCH_INSTRN_AVAILABLE_0,
    XSEARCH_INSTRN_AVAILABLE_1,
    XSEARCH_INSTRN_AVAILABLE_2,
    MOVE_INSTRN_AVAILABLE_0,
    MOVE_INSTRN_AVAILABLE_1,
    MOVE_INSTRN_AVAILABLE_2,
    FPU_INSTRN_AVAILABLE_0,
    FPU_INSTRN_AVAILABLE_1,
    FPU_INSTRN_AVAILABLE_2,
    UNPACK_INSTRN_AVAILABLE_0,
    UNPACK_INSTRN_AVAILABLE_1,
    UNPACK_INSTRN_AVAILABLE_2,
    PACK_INSTRN_AVAILABLE_0,
    PACK_INSTRN_AVAILABLE_1,
    PACK_INSTRN_AVAILABLE_2,
    THREAD_STALLS_0,
    THREAD_STALLS_1,
    THREAD_STALLS_2,
    WAITING_FOR_SRCA_CLEAR,
    WAITING_FOR_SRCB_CLEAR,
    WAITING_FOR_SRCA_VALID,
    WAITING_FOR_SRCB_VALID,
    WAITING_FOR_THCON_IDLE_0,
    WAITING_FOR_THCON_IDLE_1,
    WAITING_FOR_THCON_IDLE_2,
    WAITING_FOR_UNPACK_IDLE_0,
    WAITING_FOR_UNPACK_IDLE_1,
    WAITING_FOR_UNPACK_IDLE_2,
    WAITING_FOR_PACK_IDLE_0,
    WAITING_FOR_PACK_IDLE_1,
    WAITING_FOR_PACK_IDLE_2,
    WAITING_FOR_MATH_IDLE_0,
    WAITING_FOR_MATH_IDLE_1,
    WAITING_FOR_MATH_IDLE_2,
    WAITING_FOR_NONZERO_SEM_0,
    WAITING_FOR_NONZERO_SEM_1,
    WAITING_FOR_NONZERO_SEM_2,
    WAITING_FOR_NONFULL_SEM_0,
    WAITING_FOR_NONFULL_SEM_1,
    WAITING_FOR_NONFULL_SEM_2,
    WAITING_FOR_MOVE_IDLE_0,
    WAITING_FOR_MOVE_IDLE_1,
    WAITING_FOR_MOVE_IDLE_2,
    WAITING_FOR_MMIO_IDLE_0,
    WAITING_FOR_MMIO_IDLE_1,
    WAITING_FOR_MMIO_IDLE_2,
    WAITING_FOR_SFPU_IDLE_0,
    WAITING_FOR_SFPU_IDLE_1,
    WAITING_FOR_SFPU_IDLE_2,
    THREAD_INSTRUCTIONS_0,
    THREAD_INSTRUCTIONS_1,
    THREAD_INSTRUCTIONS_2,
    // L1 Bank 0 (MUX_CTRL bit 4 = 0, monitors L1 ports 0-7)
    L1_0_UNPACKER_0,            // Port 0: Unpacker #0
    L1_0_UNPACKER_1_ECC_PACK1,  // Port 1: Unpacker #1 / ECC / Packer #1
    L1_0_TDMA_BUNDLE_0_RISC,    // Port 2: TDMA Bundle 0 / RISC / TRISC0
    L1_0_TDMA_BUNDLE_1_TRISC,   // Port 3: TDMA Bundle 1 / TRISC1 / TRISC2
    L1_0_NOC_RING0_OUTGOING_0,  // Port 4: NOC Ring 0 Outgoing channel 0
    L1_0_NOC_RING0_OUTGOING_1,  // Port 5: NOC Ring 0 Outgoing channel 1
    L1_0_NOC_RING0_INCOMING_0,  // Port 6: NOC Ring 0 Incoming channel 0
    L1_0_NOC_RING0_INCOMING_1,  // Port 7: NOC Ring 0 Incoming channel 1
    // L1 Bank 1 (MUX_CTRL bit 4 = 1, monitors L1 ports 8-15)
    L1_1_TDMA_PACKER_2,         // Port 8: TDMA Packer 2 write
    L1_1_EXT_UNPACKER_1,        // Port 9: Extended Unpacker interface 1
    L1_1_EXT_UNPACKER_2,        // Port 10: Extended Unpacker interface 2
    L1_1_EXT_UNPACKER_3,        // Port 11: Extended Unpacker interface 3
    L1_1_NOC_RING1_OUTGOING_0,  // Port 12: NOC Ring 1 Outgoing channel 0
    L1_1_NOC_RING1_OUTGOING_1,  // Port 13: NOC Ring 1 Outgoing channel 1
    L1_1_NOC_RING1_INCOMING_0,  // Port 14: NOC Ring 1 Incoming channel 0
    L1_1_NOC_RING1_INCOMING_1,  // Port 15: NOC Ring 1 Incoming channel 1
    // Blackhole-specific L1 ports (differ from Wormhole at ports 1 and 8)
    L1_0_UNIFIED_PACKER,  // BH Port 1, mux 0: Unified Packer (WH has Unpacker#1/ECC/Pack1)
    L1_1_RISC_CORE,       // BH Port 8, mux 1: RISC Core L1 access (WH has TDMA Packer 2)
    // === Grant counters (accessed via out_fmt bit 16 = 1) ===
    // INSTRN_THREAD grant counters: actual instruction issue counts (8 types x 3 threads = 24)
    CFG_INSTRN_ISSUED_0,
    CFG_INSTRN_ISSUED_1,
    CFG_INSTRN_ISSUED_2,
    SYNC_INSTRN_ISSUED_0,
    SYNC_INSTRN_ISSUED_1,
    SYNC_INSTRN_ISSUED_2,
    THCON_INSTRN_ISSUED_0,
    THCON_INSTRN_ISSUED_1,
    THCON_INSTRN_ISSUED_2,
    XSEARCH_INSTRN_ISSUED_0,
    XSEARCH_INSTRN_ISSUED_1,
    XSEARCH_INSTRN_ISSUED_2,
    MOVE_INSTRN_ISSUED_0,
    MOVE_INSTRN_ISSUED_1,
    MOVE_INSTRN_ISSUED_2,
    FPU_INSTRN_ISSUED_0,
    FPU_INSTRN_ISSUED_1,
    FPU_INSTRN_ISSUED_2,
    UNPACK_INSTRN_ISSUED_0,
    UNPACK_INSTRN_ISSUED_1,
    UNPACK_INSTRN_ISSUED_2,
    PACK_INSTRN_ISSUED_0,
    PACK_INSTRN_ISSUED_1,
    PACK_INSTRN_ISSUED_2,
    // TDMA_UNPACK grant counters (11): detailed write/HF info
    INSTRN_2_HF_CYCLES,           // Math instrns that took 2 HF cycles (grant 257)
    INSTRN_1_HF_CYCLE,            // Math instrns that took 1 HF cycle (grant 258)
    SRCB_WRITE_ACTUAL,            // srcB writes not blocked by overwrite (grant 259)
    SRCA_WRITE_NOT_BLOCKED_OVR,   // srcA writes not blocked by overwrite (grant 260)
    SRCA_WRITE_ACTUAL,            // srcA writes not blocked by port (grant 261)
    SRCB_WRITE_NOT_BLOCKED_PORT,  // srcB writes not blocked by port (grant 262)
    SRCA_WRITE_THREAD0,           // srcA writes from thread 0 (grant 263)
    SRCB_WRITE_THREAD0,           // srcB writes from thread 0 (grant 264)
    SRCA_WRITE_THREAD1,           // srcA writes from thread 1 (grant 265)
    SRCB_WRITE_THREAD1,           // srcB writes from thread 1 (grant 266)
    MATH_INSTRN_NOT_BLOCKED_SRC,  // Math not blocked by src_data_ready (grant 256, same req as 0)
    // TDMA_PACK additional req counters (WH only, BH has these tied to 0)
    PACKER_DEST_READ_1,  // Dest accumulator register 1 read request (req 12)
    PACKER_DEST_READ_2,  // Dest accumulator register 2 read request (req 13)
    PACKER_DEST_READ_3,  // Dest accumulator register 3 read request (req 14)
    PACKER_BUSY_0,       // Packer engine 0 busy (req 15)
    PACKER_BUSY_1,       // Packer engine 1 busy (req 16)
    PACKER_BUSY_2,       // Packer engine 2 busy (req 17)
    // TDMA_PACK grant counters: dest read grants and scoreboard stalls
    DEST_READ_GRANTED_0,           // Dest register 0 read granted (grant 267)
    DEST_READ_GRANTED_1,           // Dest register 1 read granted (grant 268)
    DEST_READ_GRANTED_2,           // Dest register 2 read granted (grant 269)
    DEST_READ_GRANTED_3,           // Dest register 3 read granted (grant 270)
    MATH_NOT_STALLED_DEST_WR_PORT  // Math not stalled by dest write port (grant 271)
    // Note: AVAILABLE_MATH (existing, counter_sel 272) = math not stalled by scoreboard (grant 272)
};

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt : 24;
        uint32_t counter_type : 8;
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter() = delete;
    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(static_cast<uint32_t>(counter_type)) {}

    PerfCounter(uint64_t raw_data) : raw_data(raw_data) {}
};
static_assert(sizeof(PerfCounter) == sizeof(uint64_t), "PerfCounter must be 64-bit");

#if defined(PROFILE_PERF_COUNTERS) && COMPILE_FOR_TRISC == 1

#include "kernel_profiler.hpp"
#include "api/debug/assert.h"

namespace kernel_profiler {

constexpr PerfCounterGroup counter_groups[] = {
    PerfCounterGroup::FPU,
    PerfCounterGroup::PACK,
    PerfCounterGroup::UNPACK,
    PerfCounterGroup::L1_0,
    PerfCounterGroup::L1_1,
    PerfCounterGroup::INSTRN};
constexpr size_t NUM_COUNTER_GROUPS = sizeof(counter_groups) / sizeof(counter_groups[0]);
constexpr size_t MAX_NUM_COUNTERS_PER_GROUP = 85;  // INSTRN: 61 req + 24 grant
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> fpu_counters = {
    {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257}}};
constexpr size_t NUM_FPU_COUNTERS = 3;
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> unpack_counters = {
    {{PerfCounterType::SRCA_WRITE, 261},
     {PerfCounterType::SRCB_WRITE, 259},
     {PerfCounterType::UNPACK0_BUSY_THREAD0, 7},
     {PerfCounterType::UNPACK1_BUSY_THREAD0, 8},
     {PerfCounterType::SRCA_WRITE_AVAILABLE, 6},
     {PerfCounterType::SRCB_WRITE_AVAILABLE, 5},
     {PerfCounterType::MATH_INSTRN_STARTED, 3},
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4},
     {PerfCounterType::DATA_HAZARD_STALLS_MOVD2A, 1},
     {PerfCounterType::UNPACK0_BUSY_THREAD1, 9},
     {PerfCounterType::UNPACK1_BUSY_THREAD1, 10},
     // Additional grant counters (counter_sel with bit 16 set = out_fmt grant mode)
     // Note: SRCA_WRITE(261) and SRCB_WRITE(259) above are already grant counters
     {PerfCounterType::MATH_INSTRN_NOT_BLOCKED_SRC, 256},
     {PerfCounterType::INSTRN_2_HF_CYCLES, 257},
     {PerfCounterType::INSTRN_1_HF_CYCLE, 258},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 260},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 262},
     {PerfCounterType::SRCA_WRITE_THREAD0, 263},
     {PerfCounterType::SRCB_WRITE_THREAD0, 264},
     {PerfCounterType::SRCA_WRITE_THREAD1, 265},
     {PerfCounterType::SRCB_WRITE_THREAD1, 266}}};
constexpr size_t NUM_UNPACK_COUNTERS = 20;
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> pack_counters = {
    {{PerfCounterType::PACKER_DEST_READ_AVAILABLE, 11},
     {PerfCounterType::PACKER_BUSY, 18},
     // Additional req counters (WH: active, BH: packer engines 1-3 tied to 0)
     {PerfCounterType::PACKER_DEST_READ_1, 12},
     {PerfCounterType::PACKER_DEST_READ_2, 13},
     {PerfCounterType::PACKER_DEST_READ_3, 14},
     {PerfCounterType::PACKER_BUSY_0, 15},
     {PerfCounterType::PACKER_BUSY_1, 16},
     {PerfCounterType::PACKER_BUSY_2, 17},
     // Grant counters (counter_sel with bit 16 set)
     {PerfCounterType::DEST_READ_GRANTED_0, 267},
     {PerfCounterType::DEST_READ_GRANTED_1, 268},
     {PerfCounterType::DEST_READ_GRANTED_2, 269},
     {PerfCounterType::DEST_READ_GRANTED_3, 270},
     {PerfCounterType::MATH_NOT_STALLED_DEST_WR_PORT, 271},
     {PerfCounterType::AVAILABLE_MATH, 272}}};  // AVAILABLE_MATH = math not stalled by scoreboard
constexpr size_t NUM_PACK_COUNTERS = 14;

// L1 bank 0 counters (MUX_CTRL[6:4] = 0): unpacker, TDMA bundles, ring0 NOC
// Port 1 differs between architectures: BH has unified packer, WH has unpacker#1/ECC/pack1
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> l1_0_counters = {
    {{PerfCounterType::L1_0_UNPACKER_0, 0},
#if defined(ARCH_BLACKHOLE)
     {PerfCounterType::L1_0_UNIFIED_PACKER, 1},
#else
     {PerfCounterType::L1_0_UNPACKER_1_ECC_PACK1, 1},
#endif
     {PerfCounterType::L1_0_TDMA_BUNDLE_0_RISC, 2},
     {PerfCounterType::L1_0_TDMA_BUNDLE_1_TRISC, 3},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_0, 4},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_1, 5},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_0, 6},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_1, 7}}};
constexpr size_t NUM_L1_0_COUNTERS = 8;

// L1 bank 1 counters (MUX_CTRL[6:4] = 1): packer/risc, ext unpacker, ring1 NOC
// Port 8 differs between architectures: BH has RISC core, WH has TDMA packer 2
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> l1_1_counters = {
#if defined(ARCH_BLACKHOLE)
    {{PerfCounterType::L1_1_RISC_CORE, 0},
#else
    {{PerfCounterType::L1_1_TDMA_PACKER_2, 0},
#endif
     {PerfCounterType::L1_1_EXT_UNPACKER_1, 1},
     {PerfCounterType::L1_1_EXT_UNPACKER_2, 2},
     {PerfCounterType::L1_1_EXT_UNPACKER_3, 3},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_0, 4},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_1, 5},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_0, 6},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_1, 7}}};
constexpr size_t NUM_L1_1_COUNTERS = 8;

// INSTRN counters (61 counters)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> instrn_counters = {
    {{PerfCounterType::CFG_INSTRN_AVAILABLE_0, 0},
     {PerfCounterType::CFG_INSTRN_AVAILABLE_1, 1},
     {PerfCounterType::CFG_INSTRN_AVAILABLE_2, 2},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_0, 3},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_1, 4},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_2, 5},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_0, 6},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_1, 7},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_2, 8},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_0, 9},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_1, 10},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_2, 11},
     {PerfCounterType::MOVE_INSTRN_AVAILABLE_0, 12},
     {PerfCounterType::MOVE_INSTRN_AVAILABLE_1, 13},
     {PerfCounterType::MOVE_INSTRN_AVAILABLE_2, 14},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_0, 15},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_1, 16},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_2, 17},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_0, 18},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_1, 19},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_2, 20},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_0, 21},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_1, 22},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_2, 23},
     {PerfCounterType::THREAD_INSTRUCTIONS_0, 24},
     {PerfCounterType::THREAD_INSTRUCTIONS_1, 25},
     {PerfCounterType::THREAD_INSTRUCTIONS_2, 26},
     {PerfCounterType::THREAD_STALLS_0, 27},
     {PerfCounterType::THREAD_STALLS_1, 28},
     {PerfCounterType::THREAD_STALLS_2, 29},
     {PerfCounterType::WAITING_FOR_SRCA_VALID, 30},
     {PerfCounterType::WAITING_FOR_SRCB_VALID, 31},
     {PerfCounterType::WAITING_FOR_SRCA_CLEAR, 32},
     {PerfCounterType::WAITING_FOR_SRCB_CLEAR, 33},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_0, 34},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_1, 35},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_2, 36},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_0, 37},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_1, 38},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_2, 39},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_0, 40},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_1, 41},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_2, 42},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_0, 43},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_1, 44},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_2, 45},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_0, 46},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_1, 47},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_2, 48},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_0, 49},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_1, 50},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_2, 51},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_0, 52},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_1, 53},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_2, 54},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_0, 55},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_1, 56},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_2, 57},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_0, 58},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_1, 59},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_2, 60},
     // Grant counters: actual instruction issue counts (counter_sel + 256)
     {PerfCounterType::CFG_INSTRN_ISSUED_0, 256},
     {PerfCounterType::CFG_INSTRN_ISSUED_1, 257},
     {PerfCounterType::CFG_INSTRN_ISSUED_2, 258},
     {PerfCounterType::SYNC_INSTRN_ISSUED_0, 259},
     {PerfCounterType::SYNC_INSTRN_ISSUED_1, 260},
     {PerfCounterType::SYNC_INSTRN_ISSUED_2, 261},
     {PerfCounterType::THCON_INSTRN_ISSUED_0, 262},
     {PerfCounterType::THCON_INSTRN_ISSUED_1, 263},
     {PerfCounterType::THCON_INSTRN_ISSUED_2, 264},
     {PerfCounterType::XSEARCH_INSTRN_ISSUED_0, 265},
     {PerfCounterType::XSEARCH_INSTRN_ISSUED_1, 266},
     {PerfCounterType::XSEARCH_INSTRN_ISSUED_2, 267},
     {PerfCounterType::MOVE_INSTRN_ISSUED_0, 268},
     {PerfCounterType::MOVE_INSTRN_ISSUED_1, 269},
     {PerfCounterType::MOVE_INSTRN_ISSUED_2, 270},
     {PerfCounterType::FPU_INSTRN_ISSUED_0, 271},
     {PerfCounterType::FPU_INSTRN_ISSUED_1, 272},
     {PerfCounterType::FPU_INSTRN_ISSUED_2, 273},
     {PerfCounterType::UNPACK_INSTRN_ISSUED_0, 274},
     {PerfCounterType::UNPACK_INSTRN_ISSUED_1, 275},
     {PerfCounterType::UNPACK_INSTRN_ISSUED_2, 276},
     {PerfCounterType::PACK_INSTRN_ISSUED_0, 277},
     {PerfCounterType::PACK_INSTRN_ISSUED_1, 278},
     {PerfCounterType::PACK_INSTRN_ISSUED_2, 279}}};
constexpr size_t NUM_INSTRN_COUNTERS = 85;

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK (1 << 2)
#define PROFILE_PERF_COUNTERS_L1_0 (1 << 3)
#define PROFILE_PERF_COUNTERS_L1_1 (1 << 4)
#define PROFILE_PERF_COUNTERS_INSTRN (1 << 5)

/*
Performance Counter registers (to understand programming sequence in start_perf_counter/stop_perf_counter)

Control Registers:

The control registers are RISCV_DEBUG_REG_PERF_CNT_<X>0/1/2 (where X is FPU/TDMA_PACK/TDMA_UNPACK/L1/INSTRN_THREAD)
- RISCV_DEBUG_REG_PERF_CNT_<X>0: Reference period (in cycles)
- RISCV_DEBUG_REG_PERF_CNT_<X>1: Mode register
    - Bits [7:0]: Mode (0=manual, 1=auto-stop, 2=wrap)
        0 = Continuous
        1 = Count until refclk count number of cycles have been hit
        2 = Continuous and don’t maintain refclk count (pretty much same as 0)
    - Bits [12:8]: Bank select (i.e. which counter to read; see PerfCounterType enum for available counters and their
bank select values)
    - Bit [16]: Selects whether to output req or grant count onto RISCV_DEBUG_REG_PERF_CNT_OUT_H_<X>
        Format (0=req_cnt, 1=grant_cnt)
- RISCV_DEBUG_REG_PERF_CNT_<X>2: Control (bit[0]=start, bit[1]=stop)
    - note that these take effect on rising edge only
    - also transitioning the start bit from 0 to 1 clears the counters

Data registers:

- RISCV_DEBUG_REG_PERF_CNT_OUT_L_<X>: contains ref_cnt (i.e. cycles between start and stop)
- RISCV_DEBUG_REG_PERF_CNT_OUT_H_<X>: contains req count if RISCV_DEBUG_REG_PERF_CNT_<X>1[16] is 0
    and grant count if it is 1

Note: Currently only support for obtaining FPU counters (and deriving util metrics from them) is present
but this can be extended easily
*/

#define PERF_CNT_CONTINUOUS_MODE 0
#define PERF_CNT_BANK_SELECT_SHIFT 8
#define PERF_CNT_START_VALUE 1
#define PERF_CNT_STOP_VALUE 2

uint32_t get_cntl_register_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t reg_addr = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: reg_addr = RISCV_DEBUG_REG_PERF_CNT_FPU0; break;
        case PerfCounterGroup::PACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0; break;
        case PerfCounterGroup::UNPACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0; break;
        case PerfCounterGroup::L1_0:
        case PerfCounterGroup::L1_1: reg_addr = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
        case PerfCounterGroup::INSTRN: reg_addr = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return reg_addr;
}

uint32_t get_read_register_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t reg_addr = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU; break;
        case PerfCounterGroup::PACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK; break;
        case PerfCounterGroup::UNPACK: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK; break;
        case PerfCounterGroup::L1_0:
        case PerfCounterGroup::L1_1: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1; break;
        case PerfCounterGroup::INSTRN: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return reg_addr;
}

uint32_t get_flag_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t flag = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: flag = PROFILE_PERF_COUNTERS_FPU; break;
        case PerfCounterGroup::PACK: flag = PROFILE_PERF_COUNTERS_PACK; break;
        case PerfCounterGroup::UNPACK: flag = PROFILE_PERF_COUNTERS_UNPACK; break;
        case PerfCounterGroup::L1_0: flag = PROFILE_PERF_COUNTERS_L1_0; break;
        case PerfCounterGroup::L1_1: flag = PROFILE_PERF_COUNTERS_L1_1; break;
        case PerfCounterGroup::INSTRN: flag = PROFILE_PERF_COUNTERS_INSTRN; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return flag;
}

uint32_t get_num_counters_for_counter_group(PerfCounterGroup counter_group) {
    uint32_t num_counters = 0;
    switch (counter_group) {
        case PerfCounterGroup::FPU: num_counters = NUM_FPU_COUNTERS; break;
        case PerfCounterGroup::UNPACK: num_counters = NUM_UNPACK_COUNTERS; break;
        case PerfCounterGroup::PACK: num_counters = NUM_PACK_COUNTERS; break;
        case PerfCounterGroup::L1_0: num_counters = NUM_L1_0_COUNTERS; break;
        case PerfCounterGroup::L1_1: num_counters = NUM_L1_1_COUNTERS; break;
        case PerfCounterGroup::INSTRN: num_counters = NUM_INSTRN_COUNTERS; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return num_counters;
}

FORCE_INLINE const std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP>&
get_counters_for_counter_group(PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU: return fpu_counters;
        case PerfCounterGroup::UNPACK: return unpack_counters;
        case PerfCounterGroup::PACK: return pack_counters;
        case PerfCounterGroup::L1_0: return l1_0_counters;
        case PerfCounterGroup::L1_1: return l1_1_counters;
        case PerfCounterGroup::INSTRN: return instrn_counters;
        default: {
            ASSERT(false);
            return fpu_counters;
        }
    }
}

void set_l1_mux_ctrl(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* mux_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
    uint32_t mux_val = *mux_reg;
    // Blackhole: 3-bit L1 mux at MUX_CTRL[6:4], values 0-4
    // Wormhole:  1-bit L1 mux at MUX_CTRL[4], values 0-1
#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t L1_MUX_MASK = 0x7 << 4;  // 3 bits [6:4]
#else
    constexpr uint32_t L1_MUX_MASK = 0x1 << 4;  // 1 bit [4]
#endif
    uint32_t mux_sel = (counter_group == PerfCounterGroup::L1_1) ? 1 : 0;
    mux_val = (mux_val & ~L1_MUX_MASK) | (mux_sel << 4);
    *mux_reg = mux_val;
}

void start_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[0] = 0xFFFFFFFF;
    cntl_reg[1] = PERF_CNT_CONTINUOUS_MODE;
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

void stop_single_group(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_STOP_VALUE;
}

void read_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    volatile tt_reg_ptr uint32_t* read_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_read_register_for_counter_group(counter_group));
    const auto& counters = get_counters_for_counter_group(counter_group);
    const uint32_t counters_size = get_num_counters_for_counter_group(counter_group);
    for (unsigned int i = 0; i < counters_size; i++) {
        uint32_t counter_sel = counters[i].second;
        cntl_reg[1] = counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
        (void)read_reg[0];
        (void)read_reg[1];
        uint32_t ref_cnt_val = read_reg[0];
        uint32_t counter_val = read_reg[1];
        PerfCounter counter(counter_val, ref_cnt_val, counters[i].first);
        timeStampedData<PERF_COUNTER_PROFILER_ID>(counter.raw_data);
    }
}

// Use preprocessor #if to select groups at compile time, avoiding
// runtime loop optimization issues with the RISC-V compiler.
void start_perf_counter() {
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    start_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    start_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    start_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    start_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    start_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    start_single_group(PerfCounterGroup::INSTRN);
#endif
}

void stop_perf_counter() {
    // Stop all enabled groups first
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    stop_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    stop_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    stop_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    stop_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    stop_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    stop_single_group(PerfCounterGroup::INSTRN);
#endif

    // Read all enabled groups
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_FPU)
    read_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_PACK)
    read_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_UNPACK)
    read_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_0)
    read_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_L1_1)
    read_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS & PROFILE_PERF_COUNTERS_INSTRN)
    read_single_group(PerfCounterGroup::INSTRN);
#endif
};

// Wrapper struct for starting and stopping performance counters using scope of PerfCounterWrapper object
struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

}  // namespace kernel_profiler

#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()

#endif
