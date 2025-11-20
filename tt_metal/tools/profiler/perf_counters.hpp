// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1, INSTRN };
const PerfCounterGroup counter_groups[] = { PerfCounterGroup::FPU, PerfCounterGroup::PACK, 
    PerfCounterGroup::UNPACK, PerfCounterGroup::L1, PerfCounterGroup::INSTRN };
enum PerfCounterType : uint8_t { SFPU_COUNTER = 0, FPU_COUNTER, MATH_COUNTER };

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt : 28;
        PerfCounterType counter_type : 4;    
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type): 
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(counter_type) {} 
    
    PerfCounter(uint64_t raw_data): raw_data(raw_data) {} 
};

#if defined(PROFILE_PERF_COUNTERS) && COMPILE_FOR_TRISC == 1

#include "kernel_profiler.hpp"

namespace kernel_profiler {

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU        (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK       (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK     (1 << 2)
#define PROFILE_PERF_COUNTERS_L1         (1 << 3)
#define PROFILE_PERF_COUNTERS_INSTRN     (1 << 4)

#define PERF_CNT_CONTINUOUS_MODE 0
#define PERF_CNT_SELECT_SHIFT 8
#define PERF_CNT_START_VALUE 1
#define PERF_CNT_STOP_VALUE 2

uint32_t get_cntl_register_for_counter_group(PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU:
            return RISCV_DEBUG_REG_PERF_CNT_FPU0;
        case PerfCounterGroup::PACK:
            return RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0;
        //case PerfCounterGroup::UNPACK:
        //    return RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0;
        case PerfCounterGroup::L1:
            return RISCV_DEBUG_REG_PERF_CNT_L1_0;
        case PerfCounterGroup::INSTRN:
            return RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0;
        default: {
            //ASSERT(false);
            break;
        }
    }
    return 0; // fix!!
}

uint32_t get_read_register_for_counter_group(PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU:
            return RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU;
        default: {
            //ASSERT(false);
            break;
        }
    }
    return 0; // fix!!
}

uint32_t get_flag_for_counter_group(PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU:
            return PROFILE_PERF_COUNTERS_FPU;
        default: {
            //ASSERT(false);
            break;
        }
    }
    return 0; // fix!!
}

void start_perf_counter() {
    // start counters for selected groups
    for (auto counter_group: counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            uint32_t counter_select = 0; // can select individual counters later when reading
            cntl_reg[0] = 0;
            cntl_reg[1] = counter_select << PERF_CNT_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
            cntl_reg[2] = PERF_CNT_START_VALUE;
            cntl_reg[2] = 0;
        }
    }
};

void stop_perf_counter() {
    // stop all counters first
    for (auto counter_group: counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            cntl_reg[2] = PERF_CNT_STOP_VALUE;
            cntl_reg[2] = 0;
        }
    }

    // read data from all counters in all enabled groups  
    for (auto counter_group: counter_groups) {
        if (PROFILE_PERF_COUNTERS & get_flag_for_counter_group(counter_group)) {
            volatile tt_reg_ptr uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
            volatile tt_reg_ptr uint32_t* read_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_read_register_for_counter_group(counter_group));
            const std::pair<PerfCounterType, uint16_t> counters[] = { {PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257} };
            uint32_t counters_size = sizeof(counters) / sizeof(counters[0]);
            for (unsigned int i = 0; i < 3; i++) {
                uint32_t counter_sel = counters[i].second;
                cntl_reg[1] = counter_sel << PERF_CNT_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
                // wait for registers to update
                while (cntl_reg[1] != counter_sel << PERF_CNT_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE);
                for (int i = 0; i < 50; i++) {
                    asm("nop");
                }
                PerfCounter counter(read_reg[1], read_reg[0], counters[i].first);
                timeStampedData<12345>(counter.raw_data);
            }
        }
    }
};

struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

}  // namespace kernel_profiler

#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define PerfCountersRaii() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define PerfCountersRaii()

#endif