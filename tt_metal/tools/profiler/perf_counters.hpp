// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0 (RISCV_DEBUG_REGS_START_ADDR | 0xC)
#endif

#ifndef RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK (RISCV_DEBUG_REGS_START_ADDR | 0x110)
#endif

#ifndef RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK (RISCV_DEBUG_REGS_START_ADDR | 0x108)
#endif

#ifndef RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1 (RISCV_DEBUG_REGS_START_ADDR | 0x118)
#endif

#ifndef RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD (RISCV_DEBUG_REGS_START_ADDR | 0x100)
#endif

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1, INSTRN };
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
    // L1 Group (16 counters, mux-dependent)
    NOC_RING0_INCOMING_1,
    NOC_RING0_INCOMING_0,
    NOC_RING0_OUTGOING_1,
    NOC_RING0_OUTGOING_0,
    L1_ARB_TDMA_BUNDLE_1,
    L1_ARB_TDMA_BUNDLE_0,
    L1_ARB_UNPACKER,
    L1_NO_ARB_UNPACKER,
    NOC_RING1_INCOMING_1,
    NOC_RING1_INCOMING_0,
    NOC_RING1_OUTGOING_1,
    NOC_RING1_OUTGOING_0,
    TDMA_BUNDLE_1_ARB,
    TDMA_BUNDLE_0_ARB,
    TDMA_EXT_UNPACK_9_10,
    TDMA_PACKER_2_WR
};

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt : 25;
        PerfCounterType counter_type : 7;
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter() = delete;
    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(counter_type) {}

    PerfCounter(uint64_t raw_data) : raw_data(raw_data) {}
};

#if defined(PROFILE_PERF_COUNTERS) && COMPILE_FOR_TRISC == 1

#include "kernel_profiler.hpp"
#include "api/debug/assert.h"

namespace kernel_profiler {

const PerfCounterGroup counter_groups[] = {
    PerfCounterGroup::FPU,
    PerfCounterGroup::PACK,
    PerfCounterGroup::UNPACK,
    PerfCounterGroup::L1,
    PerfCounterGroup::INSTRN};
constexpr size_t MAX_NUM_COUNTERS_PER_GROUP = 61;
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
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4}}};
constexpr size_t NUM_UNPACK_COUNTERS = 8;
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> pack_counters = {
    {{PerfCounterType::PACKER_DEST_READ_AVAILABLE, 11},
     {PerfCounterType::PACKER_BUSY, 18},
     {PerfCounterType::AVAILABLE_MATH, 272}}};
constexpr size_t NUM_PACK_COUNTERS = 3;

// L1 counters (16 counters) - Note: Some are mux-dependent
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> l1_counters = {
    {{PerfCounterType::NOC_RING0_INCOMING_1, 0},
     {PerfCounterType::NOC_RING0_INCOMING_0, 1},
     {PerfCounterType::NOC_RING0_OUTGOING_1, 2},
     {PerfCounterType::NOC_RING0_OUTGOING_0, 3},
     {PerfCounterType::L1_ARB_TDMA_BUNDLE_1, 4},
     {PerfCounterType::L1_ARB_TDMA_BUNDLE_0, 5},
     {PerfCounterType::L1_ARB_UNPACKER, 6},
     {PerfCounterType::L1_NO_ARB_UNPACKER, 7},
     {PerfCounterType::NOC_RING1_INCOMING_1, 8},
     {PerfCounterType::NOC_RING1_INCOMING_0, 9},
     {PerfCounterType::NOC_RING1_OUTGOING_1, 10},
     {PerfCounterType::NOC_RING1_OUTGOING_0, 11},
     {PerfCounterType::TDMA_BUNDLE_1_ARB, 12},
     {PerfCounterType::TDMA_BUNDLE_0_ARB, 13},
     {PerfCounterType::TDMA_EXT_UNPACK_9_10, 14},
     {PerfCounterType::TDMA_PACKER_2_WR, 15}}};
constexpr size_t NUM_L1_COUNTERS = 16;

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
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_2, 60}}};
constexpr size_t NUM_INSTRN_COUNTERS = 61;

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK (1 << 2)
#define PROFILE_PERF_COUNTERS_L1 (1 << 3)
#define PROFILE_PERF_COUNTERS_INSTRN (1 << 4)

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
        case PerfCounterGroup::L1: reg_addr = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
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
        case PerfCounterGroup::L1: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1; break;
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
        case PerfCounterGroup::L1: flag = PROFILE_PERF_COUNTERS_L1; break;
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
        case PerfCounterGroup::L1: num_counters = NUM_L1_COUNTERS; break;
        case PerfCounterGroup::INSTRN: num_counters = NUM_INSTRN_COUNTERS; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return num_counters;
}

FORCE_INLINE std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP>
get_counters_for_counter_group(PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU: return fpu_counters;
        case PerfCounterGroup::UNPACK: return unpack_counters;
        case PerfCounterGroup::PACK: return pack_counters;
        case PerfCounterGroup::L1: return l1_counters;
        case PerfCounterGroup::INSTRN: return instrn_counters;
        default: {
            ASSERT(false);
            break;
        }
    }
    return std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP>();
}

void start_perf_counter() {
    // start counters for selected groups
    for (auto counter_group : counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg =
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            uint32_t counter_select = 0;  // individual counters selected later when reading
            // Set continuous mode then start bit in control registers to start counters
            cntl_reg[0] = 0;
            cntl_reg[1] = counter_select << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
            cntl_reg[2] = PERF_CNT_START_VALUE;
            cntl_reg[2] = 0;
        }
    }
};

void stop_perf_counter() {
    // Stop all counters first (set stop bit)
    for (auto counter_group : counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg =
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            cntl_reg[2] = PERF_CNT_STOP_VALUE;
            cntl_reg[2] = 0;
        }
    }

    // Read data from all counters in all enabled groups
    for (auto counter_group : counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg =
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            volatile tt_reg_ptr uint32_t* read_reg =
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_read_register_for_counter_group(counter_group));
            const std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> counters =
                get_counters_for_counter_group(counter_group);
            const uint32_t counters_size = get_num_counters_for_counter_group(counter_group);
            for (unsigned int i = 0; i < counters_size; i++) {
                uint32_t counter_sel = counters[i].second;
                cntl_reg[1] = counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
                // Wait for registers to update
                while (cntl_reg[1] != (counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE));
                // Extra wait
                for (int wait_count = 0; wait_count < 50; wait_count++) {
                    asm("nop");
                }
                PerfCounter counter(read_reg[1], read_reg[0], counters[i].first);
                timeStampedData<PERF_COUNTER_PROFILER_ID>(counter.raw_data);
            }
        }
    }
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
