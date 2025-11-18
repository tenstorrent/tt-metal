// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(PROFILE_KERNEL) && COMPILE_FOR_TRISC == 1

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

// figure how to use register: can we just read before and after?
//      define list of counters
//      define a read function, when to reset, then read
// define mechanism to send to host: profiler timesrampedData
// host side processing, 

//uint32_t counters = [
//    // FPU Counter Group
//    RISCV_DEBUG_REG_PERF_CNT_FPU0
//    RISCV_DEBUG_REG_PERF_CNT_FPU1
//    RISCV_DEBUG_REG_PERF_CNT_FPU2
//    // L1 Counter Group
//    RISCV_DEBUG_REG_PERF_CNT_L1_0
//    RISCV_DEBUG_REG_PERF_CNT_L1_1
//    RISCV_DEBUG_REG_PERF_CNT_L1_2
//    // TDMA PACK Counter Group
//    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0
//    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1
//    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2
//    // Instruction Thread Counter Group
//    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0
//    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1
//    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2
//    ]; 

// other groups??

enum PerfCounterType { SFPU_COUNTER = 0 } uint8_t;

union PerfCounter {
    struct {
        uint64_t counter_value : 56;
        PerfCounterType counter_type;    
    } __attribute__((packed));
    uint64_t raw_data;

    PerfCounter(PerfCounterType counter_type, uint64_t counter_value): 
        counter_value(counter_value), counter_type(counter_type) {} 
};


// receives a perf counter group or multiple perf counter types?
// i.e. how do we elegantly handle all the option for counters we want?

// reg_select = RISCV_DEBUG_REG_PERF_CNT_FPU0
// counter_select = 1
void start_perf_counter(uint32_t reg_select, uint32_t counter_select) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(reg_select);
    p_reg[0] = 0;
    p_reg[1] = counter_select << PERF_CNT_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
    p_reg[2] = PERF_CNT_START_VALUE;
    p_reg[2] = 0;
};

// reg_select = RISCV_DEBUG_REG_PERF_CNT_FPU0
// read_reg = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU
void stop_perf_counter(uint32_t reg_select, uint32_t read_reg) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(reg_select);
    p_reg[2] = PERF_CNT_STOP_VALUE;
    p_reg[2] = 0;
    // read data
    p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(read_reg);
    PerfCounter counter(PerfCounterType::SFPU_COUNTER, p_reg[0]);
    timeStampedData<PerfCounterType::SFPU_COUNTER>(counter.raw_data);
};

}  // namespace kernel_profiler

#define StartPerfCounters() kernel_profiler::start_perf_counter(RISCV_DEBUG_REG_PERF_CNT_FPU0, 1);
#define StopPerfCounters() kernel_profiler::stop_perf_counter(RISCV_DEBUG_REG_PERF_CNT_FPU0, RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU);

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()

#endif
