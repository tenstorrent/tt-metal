// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1, INSTRN };
enum PerfCounterType : uint8_t {
    UNDEF = 0,
    // FPU Group
    SFPU_COUNTER,
    FPU_COUNTER,
    MATH_COUNTER
};

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt : 28;
        PerfCounterType counter_type : 4;
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter() = delete;
    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(counter_type) {}

    PerfCounter(uint64_t raw_data) : raw_data(raw_data) {}
};

#if defined(PROFILE_PERF_COUNTERS) && COMPILE_FOR_TRISC == 1

#include "kernel_profiler.hpp"
#include "debug/assert.h"

namespace kernel_profiler {

const PerfCounterGroup counter_groups[] = {
    PerfCounterGroup::FPU,
    PerfCounterGroup::PACK,
    PerfCounterGroup::UNPACK,
    PerfCounterGroup::L1,
    PerfCounterGroup::INSTRN};
constexpr size_t MAX_NUM_COUNTERS_PER_GROUP = 5;
constexpr std::array<std::pair<PerfCounterType, uint16_t>, MAX_NUM_COUNTERS_PER_GROUP> fpu_counters = {
    {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257}}};
constexpr size_t NUM_FPU_COUNTERS = 3;

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
