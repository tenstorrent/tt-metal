// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4 };
enum PerfCounterType : uint16_t {
    UNDEF = 0,
    // FPU Group (3 counters)
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group
    MATH_SRC_DATA_READY,
    DATA_HAZARD_STALLS_MOVD2A,
    MATH_FIDELITY_STALL,
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_AVAILABLE,
    SRCA_WRITE_AVAILABLE,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    MATH_INSTRN_HF_1_CYCLE,
    MATH_INSTRN_HF_2_CYCLE,
    MATH_INSTRN_HF_4_CYCLE,
    // TDMA_PACK Group (3 counters)
    PACKER_DEST_READ_AVAILABLE,
    PACKER_BUSY,
    AVAILABLE_MATH,
    // INSTRN_THREAD Group
    CFG_INSTRN_AVAILABLE_0,
    CFG_INSTRN_AVAILABLE_1,
    CFG_INSTRN_AVAILABLE_2,
    SYNC_INSTRN_AVAILABLE_0,
    SYNC_INSTRN_AVAILABLE_1,
    SYNC_INSTRN_AVAILABLE_2,
    THCON_INSTRN_AVAILABLE_0,
    THCON_INSTRN_AVAILABLE_1,
    THCON_INSTRN_AVAILABLE_2,
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
    // L1 Bank 0 (mux=0, ports 0-7)
    L1_0_UNPACKER_0,
    L1_0_UNPACKER_1_ECC_PACK1,  // port 1: WH only
    L1_0_TDMA_BUNDLE_0_RISC,
    L1_0_TDMA_BUNDLE_1_TRISC,
    L1_0_NOC_RING0_OUTGOING_0,
    L1_0_NOC_RING0_OUTGOING_1,
    L1_0_NOC_RING0_INCOMING_0,
    L1_0_NOC_RING0_INCOMING_1,
    // L1 Bank 1 (mux=1, ports 8-15)
    L1_1_TDMA_PACKER_2,  // port 8: WH only
    L1_1_EXT_UNPACKER_1,
    L1_1_EXT_UNPACKER_2,
    L1_1_EXT_UNPACKER_3,
    L1_1_NOC_RING1_OUTGOING_0,
    L1_1_NOC_RING1_OUTGOING_1,
    L1_1_NOC_RING1_INCOMING_0,
    L1_1_NOC_RING1_INCOMING_1,
    // BH-specific L1 ports (port 1 and port 8 differ from WH)
    L1_0_UNIFIED_PACKER,
    L1_1_RISC_CORE,
    // L1 grant counters (reqif_ready)
    L1_0_UNPACKER_0_GRANT,
    L1_0_PORT1_GRANT,
    L1_0_TDMA_BUNDLE_0_GRANT,
    L1_0_TDMA_BUNDLE_1_GRANT,
    L1_0_NOC_RING0_OUTGOING_0_GRANT,
    L1_0_NOC_RING0_OUTGOING_1_GRANT,
    L1_0_NOC_RING0_INCOMING_0_GRANT,
    L1_0_NOC_RING0_INCOMING_1_GRANT,
    L1_1_PORT8_GRANT,
    L1_1_EXT_UNPACKER_1_GRANT,
    L1_1_EXT_UNPACKER_2_GRANT,
    L1_1_EXT_UNPACKER_3_GRANT,
    L1_1_NOC_RING1_OUTGOING_0_GRANT,
    L1_1_NOC_RING1_OUTGOING_1_GRANT,
    L1_1_NOC_RING1_INCOMING_0_GRANT,
    L1_1_NOC_RING1_INCOMING_1_GRANT,
    // === Grant-side counters (accessed via out_fmt bit 16 = 1) ===
    THREAD_INSTRUCTIONS_0,
    THREAD_INSTRUCTIONS_1,
    THREAD_INSTRUCTIONS_2,
    SRCB_WRITE_ACTUAL,
    SRCA_WRITE_NOT_BLOCKED_OVR,
    SRCA_WRITE_ACTUAL,
    SRCB_WRITE_NOT_BLOCKED_PORT,
    SRCA_WRITE_THREAD0,
    SRCB_WRITE_THREAD0,
    SRCA_WRITE_THREAD1,
    SRCB_WRITE_THREAD1,
    // TDMA_PACK additional req counters (WH only)
    PACKER_DEST_READ_1,
    PACKER_DEST_READ_2,
    PACKER_DEST_READ_3,
    PACKER_BUSY_0,
    PACKER_BUSY_1,
    PACKER_BUSY_2,
    DEST_READ_GRANTED_0,
    DEST_READ_GRANTED_1,
    DEST_READ_GRANTED_2,
    DEST_READ_GRANTED_3,
    MATH_NOT_STALLED_DEST_WR_PORT,
    // L1 Bank 4 (BH only, mux=4, misc ports 32-39)
    L1_4_MISC_PORT_0,
    L1_4_MISC_PORT_1,
    L1_4_MISC_PORT_2,
    L1_4_MISC_PORT_3,
    L1_4_MISC_PORT_4,
    L1_4_MISC_PORT_5,
    L1_4_MISC_PORT_6,
    L1_4_MISC_PORT_7,
    L1_4_MISC_PORT_0_GRANT,
    L1_4_MISC_PORT_1_GRANT,
    L1_4_MISC_PORT_2_GRANT,
    L1_4_MISC_PORT_3_GRANT,
    L1_4_MISC_PORT_4_GRANT,
    L1_4_MISC_PORT_5_GRANT,
    L1_4_MISC_PORT_6_GRANT,
    L1_4_MISC_PORT_7_GRANT,
    // L1 Bank 2 (BH only, mux=2, NOC Ring 2 ports 16-23)
    L1_2_NOC_RING2_PORT_0,
    L1_2_NOC_RING2_PORT_1,
    L1_2_NOC_RING2_PORT_2,
    L1_2_NOC_RING2_PORT_3,
    L1_2_NOC_RING2_PORT_4,
    L1_2_NOC_RING2_PORT_5,
    L1_2_NOC_RING2_PORT_6,
    L1_2_NOC_RING2_PORT_7,
    L1_2_NOC_RING2_PORT_0_GRANT,
    L1_2_NOC_RING2_PORT_1_GRANT,
    L1_2_NOC_RING2_PORT_2_GRANT,
    L1_2_NOC_RING2_PORT_3_GRANT,
    L1_2_NOC_RING2_PORT_4_GRANT,
    L1_2_NOC_RING2_PORT_5_GRANT,
    L1_2_NOC_RING2_PORT_6_GRANT,
    L1_2_NOC_RING2_PORT_7_GRANT,
    // L1 Bank 3 (BH only, mux=3, NOC Ring 3 ports 24-31)
    L1_3_NOC_RING3_PORT_0,
    L1_3_NOC_RING3_PORT_1,
    L1_3_NOC_RING3_PORT_2,
    L1_3_NOC_RING3_PORT_3,
    L1_3_NOC_RING3_PORT_4,
    L1_3_NOC_RING3_PORT_5,
    L1_3_NOC_RING3_PORT_6,
    L1_3_NOC_RING3_PORT_7,
    L1_3_NOC_RING3_PORT_0_GRANT,
    L1_3_NOC_RING3_PORT_1_GRANT,
    L1_3_NOC_RING3_PORT_2_GRANT,
    L1_3_NOC_RING3_PORT_3_GRANT,
    L1_3_NOC_RING3_PORT_4_GRANT,
    L1_3_NOC_RING3_PORT_5_GRANT,
    L1_3_NOC_RING3_PORT_6_GRANT,
    L1_3_NOC_RING3_PORT_7_GRANT,
    ANY_THREAD_STALL,
    // counter_type is a uint32_t:8 bitfield — keep all values below 256.
};
static_assert(ANY_THREAD_STALL <= 255, "PerfCounterType enum exceeds 8-bit counter_type field");

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

// Perf counter start/stop runs on TRISC1 (wraps the compute kernel).
// Counter readout and DRAM push runs on BRISC (has NOC access for DRAM writes).
#if defined(PROFILE_PERF_COUNTERS) && (COMPILE_FOR_TRISC == 1 || defined(COMPILE_FOR_BRISC))

#include "kernel_profiler.hpp"
#include "api/debug/assert.h"

namespace kernel_profiler {

// Architecture-specific counter arrays (fpu, unpack, pack, l1_0-l1_4, instrn)
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/hw/inc/internal/tt-1xx/blackhole/hw_counters.h"
#else
#include "tt_metal/hw/inc/internal/tt-1xx/wormhole/hw_counters.h"
#endif

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK (1 << 2)
#define PROFILE_PERF_COUNTERS_L1_0 (1 << 3)
#define PROFILE_PERF_COUNTERS_L1_1 (1 << 4)
#define PROFILE_PERF_COUNTERS_INSTRN (1 << 5)
#define PROFILE_PERF_COUNTERS_L1_2 (1 << 6)
#define PROFILE_PERF_COUNTERS_L1_3 (1 << 7)
#define PROFILE_PERF_COUNTERS_L1_4 (1 << 8)

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
        case PerfCounterGroup::L1_1:
        case PerfCounterGroup::L1_2:
        case PerfCounterGroup::L1_3:
        case PerfCounterGroup::L1_4: reg_addr = RISCV_DEBUG_REG_PERF_CNT_L1_0; break;
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
        case PerfCounterGroup::L1_1:
        case PerfCounterGroup::L1_2:
        case PerfCounterGroup::L1_3:
        case PerfCounterGroup::L1_4: reg_addr = RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1; break;
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
        case PerfCounterGroup::L1_2: flag = PROFILE_PERF_COUNTERS_L1_2; break;
        case PerfCounterGroup::L1_3: flag = PROFILE_PERF_COUNTERS_L1_3; break;
        case PerfCounterGroup::L1_4: flag = PROFILE_PERF_COUNTERS_L1_4; break;
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
        case PerfCounterGroup::L1_2: num_counters = NUM_L1_2_COUNTERS; break;
        case PerfCounterGroup::L1_3: num_counters = NUM_L1_3_COUNTERS; break;
        case PerfCounterGroup::L1_4: num_counters = NUM_L1_4_COUNTERS; break;
        default: {
            ASSERT(false);
            break;
        }
    }
    return num_counters;
}

FORCE_INLINE const std::pair<PerfCounterType, uint16_t>* get_counters_for_counter_group(
    PerfCounterGroup counter_group) {
    switch (counter_group) {
        case PerfCounterGroup::FPU: return fpu_counters.data();
        case PerfCounterGroup::UNPACK: return unpack_counters.data();
        case PerfCounterGroup::PACK: return pack_counters.data();
        case PerfCounterGroup::L1_0: return l1_0_counters.data();
        case PerfCounterGroup::L1_1: return l1_1_counters.data();
        case PerfCounterGroup::INSTRN: return instrn_counters.data();
        case PerfCounterGroup::L1_2: return l1_2_counters.data();
        case PerfCounterGroup::L1_3: return l1_3_counters.data();
        case PerfCounterGroup::L1_4: return l1_4_counters.data();
        default: {
            ASSERT(false);
            return fpu_counters.data();
        }
    }
}

void set_l1_mux_ctrl(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* mux_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
    uint32_t mux_val = *mux_reg;
    // L1_MUX_MASK defined in hw_counters.h (arch-specific width)
    uint32_t mux_sel = 0;
    if (counter_group == PerfCounterGroup::L1_1) {
        mux_sel = 1;
    } else if (counter_group == PerfCounterGroup::L1_2) {
        mux_sel = 2;
    } else if (counter_group == PerfCounterGroup::L1_3) {
        mux_sel = 3;
    } else if (counter_group == PerfCounterGroup::L1_4) {
        mux_sel = 4;
    }
    mux_val = (mux_val & ~L1_MUX_MASK) | (mux_sel << 4);
    *mux_reg = mux_val;
}

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1 ||
        counter_group == PerfCounterGroup::L1_2 || counter_group == PerfCounterGroup::L1_3 ||
        counter_group == PerfCounterGroup::L1_4) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[0] = 0xFFFFFFFF;
    cntl_reg[1] = PERF_CNT_CONTINUOUS_MODE;
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

__attribute__((noinline)) void stop_single_group(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_STOP_VALUE;
}

__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group) {
    if (counter_group == PerfCounterGroup::L1_0 || counter_group == PerfCounterGroup::L1_1 ||
        counter_group == PerfCounterGroup::L1_2 || counter_group == PerfCounterGroup::L1_3 ||
        counter_group == PerfCounterGroup::L1_4) {
        set_l1_mux_ctrl(counter_group);
    }
    volatile tt_reg_ptr uint32_t* cntl_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_cntl_register_for_counter_group(counter_group));
    volatile tt_reg_ptr uint32_t* read_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_read_register_for_counter_group(counter_group));
    const auto* counters = get_counters_for_counter_group(counter_group);
    const uint32_t counters_size = get_num_counters_for_counter_group(counter_group);
    for (unsigned int i = 0; i < counters_size; i++) {
        uint32_t counter_sel = counters[i].second;
        uint32_t expected_mode = counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
        cntl_reg[1] = expected_mode;
        // Wait for the mode register write (counter_sel mux change) to take
        // effect before reading the output registers. The original #33109
        // implementation used this readback poll + 50 NOPs. We briefly replaced
        // it with two dummy volatile reads as an implicit fence, which produced
        // identical counter values — but volatile reads have no formal RISC-V
        // spec guarantee for MMIO ordering. The explicit readback poll is
        // correct by construction: it completes only when the hardware confirms
        // the new mux select value, ensuring the output registers reflect the
        // selected counter bank. The 50 NOPs were dropped since the readback
        // alone is sufficient (the mux settles within the poll cycle).
        while (cntl_reg[1] != expected_mode);
        uint32_t ref_cnt_val = read_reg[0];
        uint32_t counter_val = read_reg[1];
        PerfCounter counter(counter_val, ref_cnt_val, counters[i].first);
        timeStampedData<PERF_COUNTER_PROFILER_ID>(counter.raw_data);
    }
}

// Counter groups and their corresponding enable bitmask bits.
constexpr std::pair<PerfCounterGroup, uint32_t> counter_group_flags[] = {
    {PerfCounterGroup::FPU, PROFILE_PERF_COUNTERS_FPU},
    {PerfCounterGroup::PACK, PROFILE_PERF_COUNTERS_PACK},
    {PerfCounterGroup::UNPACK, PROFILE_PERF_COUNTERS_UNPACK},
    {PerfCounterGroup::L1_0, PROFILE_PERF_COUNTERS_L1_0},
    {PerfCounterGroup::L1_1, PROFILE_PERF_COUNTERS_L1_1},
    {PerfCounterGroup::INSTRN, PROFILE_PERF_COUNTERS_INSTRN},
    {PerfCounterGroup::L1_2, PROFILE_PERF_COUNTERS_L1_2},
    {PerfCounterGroup::L1_3, PROFILE_PERF_COUNTERS_L1_3},
    {PerfCounterGroup::L1_4, PROFILE_PERF_COUNTERS_L1_4},
};
constexpr uint32_t NUM_COUNTER_GROUPS = sizeof(counter_group_flags) / sizeof(counter_group_flags[0]);

void start_perf_counter() {
    for (uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            start_single_group(counter_group_flags[i].first);
        }
    }
}

// stop_perf_counter: stops all enabled counter groups (freezes hardware counters).
// Called from TRISC1 at the end of the compute kernel scope.
// Does NOT read counter values — that happens on BRISC which has NOC access for DRAM push.
void stop_perf_counter() {
    for (uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            stop_single_group(counter_group_flags[i].first);
        }
    }
};

// Flush perf counter data from L1 to DRAM — body only (from CUSTOM_MARKERS..wIndex).
// Appends the body directly to HOST_BUFFER_END_INDEX without writing a header or
// sentinel — the BRISC-FW ZONE_END is still a placeholder at this point (destructor
// has not fired), so a header written now would produce an orphan ZONE_START.
// The host parses these markers as pre-sentinel TS_DATA and associates them with
// the run established by the sentinel that finish_profiler writes once the
// profileScopeGuaranteed destructor has populated the final marker timestamps.
__attribute__((noinline)) void perf_counter_flush() {
#if defined(COMPILE_FOR_BRISC)
    if (!profiler_control_buffer[DRAM_PROFILER_ADDRESS]) {
        return;
    }
    if (wIndex <= CUSTOM_MARKERS) {
        return;  // nothing to flush
    }

    uint32_t core_flat_id = profiler_control_buffer[FLAT_ID];
    uint32_t profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];

    uint32_t send_count = wIndex - CUSTOM_MARKERS;

    // Pad body to NOC alignment
    for (uint32_t i = 0; i < (send_count % NOC_ALIGNMENT_FACTOR); i++) {
        mark_padding();
        send_count += PROFILER_L1_MARKER_UINT32_SIZE;
    }

    uint32_t currEndIndex = profiler_control_buffer[HOST_BUFFER_END_INDEX] + send_count;
    if (currEndIndex > PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC) {
        wIndex = CUSTOM_MARKERS;
        return;  // DRAM full, drop remaining counters
    }

    uint32_t dram_offset = (core_flat_id % profiler_core_count_per_dram) * MaxProcessorsPerCoreType *
                               PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                           HOST_BUFFER_END_INDEX * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                           profiler_control_buffer[HOST_BUFFER_END_INDEX] * sizeof(uint32_t);

    const auto s = TensorAccessor(
        tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(),
        profiler_control_buffer[DRAM_PROFILER_ADDRESS],
        PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MaxProcessorsPerCoreType * profiler_core_count_per_dram);

    uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

    NocRegisterStateSave noc_state;
    profiler_noc_async_write_posted(
        reinterpret_cast<uint32_t>(&profiler_data_buffer[myRiscID].data[CUSTOM_MARKERS]),
        dram_bank_dst_noc_addr,
        send_count * sizeof(uint32_t));
    profiler_noc_async_flush_posted_write();

    profiler_control_buffer[HOST_BUFFER_END_INDEX] = currEndIndex;
    wIndex = CUSTOM_MARKERS;
#endif
}

// read_perf_counters: reads all enabled counter groups and writes markers to the profiler buffer.
// Called from BRISC after wait_ncrisc_trisc() — BRISC has NOC access so it can push the L1
// profiler buffer to DRAM (via perf_counter_flush) between groups when the buffer fills up.
// Flush BEFORE each group (starting from 2nd) to ensure the buffer has room.
void read_perf_counters() {
    bool first_group = true;
    for (uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            if (!first_group) {
                perf_counter_flush();
            }
            read_single_group(counter_group_flags[i].first);
            first_group = false;
        }
    }
};

// TRISC1: RAII wrapper that starts counters in constructor and stops in destructor.
// Counter readout happens later on BRISC via read_perf_counters().
struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

}  // namespace kernel_profiler

#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define ReadPerfCounters() kernel_profiler::read_perf_counters();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters()
#define RecordPerfCounters()

#endif
