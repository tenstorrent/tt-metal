// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4 };
enum PerfCounterType : uint16_t {
    UNDEF = 0,
    // FPU Group
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group
    MATH_SRC_DATA_READY,
    DATA_HAZARD_STALLS_MOVD2A,
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_AVAILABLE,
    SRCA_WRITE_AVAILABLE,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    // TDMA_PACK Group
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
    L1_0_UNPACKER_1_ECC,
    L1_1_TDMA_PACKER_0,
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
    // L1 Bank 4 (BH only, mux=4, ports 32-34): ext TDMA packer 4-5, tag-search/packer1 (35-39 unused)
    L1_4_TDMA_PACK_EXT_4,
    L1_4_TDMA_PACK_EXT_5,
    L1_4_TAG_SEARCH_PACKER1,
    // Reserved: L1 bank-4 ports 35-39 are tied to 0 in RTL (UNUSED_START_IDX=35), so NO counter
    // array on any arch references these enumerators. They are kept ONLY to hold the enum ordinals
    // stable — the profiler stores/decodes the compiled ordinal as the wire value, so deleting them
    // would shift every subsequent ordinal (INSTRN bank) and mis-decode captured data. Do not remove.
    L1_4_MISC_PORT_3,
    L1_4_MISC_PORT_4,
    L1_4_MISC_PORT_5,
    L1_4_MISC_PORT_6,
    L1_4_MISC_PORT_7,
    L1_4_TDMA_PACK_EXT_4_GRANT,
    L1_4_TDMA_PACK_EXT_5_GRANT,
    L1_4_TAG_SEARCH_PACKER1_GRANT,
    // Reserved (see above): grant twins of the tied-0 ports 35-39; kept for ordinal stability only.
    L1_4_MISC_PORT_3_GRANT,
    L1_4_MISC_PORT_4_GRANT,
    L1_4_MISC_PORT_5_GRANT,
    L1_4_MISC_PORT_6_GRANT,
    L1_4_MISC_PORT_7_GRANT,
    // L1 Bank 2 (BH only, mux=2, ports 16-23): ext unpacker 4-7 + NOC ring0 secondary channels
    L1_2_EXT_UNPACKER_4,
    L1_2_EXT_UNPACKER_5,
    L1_2_EXT_UNPACKER_6,
    L1_2_EXT_UNPACKER_7,
    L1_2_NOC_RING0_OUTGOING_2,
    L1_2_NOC_RING0_OUTGOING_3,
    L1_2_NOC_RING0_INCOMING_2,
    L1_2_NOC_RING0_INCOMING_3,
    L1_2_EXT_UNPACKER_4_GRANT,
    L1_2_EXT_UNPACKER_5_GRANT,
    L1_2_EXT_UNPACKER_6_GRANT,
    L1_2_EXT_UNPACKER_7_GRANT,
    L1_2_NOC_RING0_OUTGOING_2_GRANT,
    L1_2_NOC_RING0_OUTGOING_3_GRANT,
    L1_2_NOC_RING0_INCOMING_2_GRANT,
    L1_2_NOC_RING0_INCOMING_3_GRANT,
    // L1 Bank 3 (BH only, mux=3, ports 24-31): NOC ring1 secondary channels + ext TDMA packer 0-3
    L1_3_NOC_RING1_OUTGOING_2,
    L1_3_NOC_RING1_OUTGOING_3,
    L1_3_NOC_RING1_INCOMING_2,
    L1_3_NOC_RING1_INCOMING_3,
    L1_3_TDMA_PACK_EXT_0,
    L1_3_TDMA_PACK_EXT_1,
    L1_3_TDMA_PACK_EXT_2,
    L1_3_TDMA_PACK_EXT_3,
    L1_3_NOC_RING1_OUTGOING_2_GRANT,
    L1_3_NOC_RING1_OUTGOING_3_GRANT,
    L1_3_NOC_RING1_INCOMING_2_GRANT,
    L1_3_NOC_RING1_INCOMING_3_GRANT,
    L1_3_TDMA_PACK_EXT_0_GRANT,
    L1_3_TDMA_PACK_EXT_1_GRANT,
    L1_3_TDMA_PACK_EXT_2_GRANT,
    L1_3_TDMA_PACK_EXT_3_GRANT,
    ANY_THREAD_STALL,
    // counter_type is a uint32_t:8 bitfield — keep all values below 256.
};
static_assert(ANY_THREAD_STALL <= 255, "PerfCounterType enum exceeds 8-bit counter_type field");

union PerfCounter {
    struct {
        uint32_t counter_value;
        uint32_t ref_cnt;
        uint32_t counter_type : 8;
        uint32_t unused : 24;
    } __attribute__((packed));
    struct {
        uint64_t raw_data_1;
        uint64_t raw_data_2;
    } __attribute__((packed));

    PerfCounter() = delete;
    PerfCounter(uint32_t counter_value, uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(static_cast<uint32_t>(counter_type)) {}

    PerfCounter(uint64_t raw_data_1, uint64_t raw_data_2) : raw_data_1(raw_data_1), raw_data_2(raw_data_2) {}
};
static_assert(sizeof(PerfCounter) == sizeof(uint64_t) * 2, "PerfCounter must be 128-bit");

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

#define PERF_CNT_CONTINUOUS_MODE 0
#define PERF_CNT_BANK_SELECT_SHIFT 8
#define PERF_CNT_START_VALUE 1
#define PERF_CNT_STOP_VALUE 2

// Counter groups and their corresponding enable bitmask bits. Shared; used on both
// TRISC1 (start/stop loop) and BRISC (read loop).
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

// Lookup table indexed by PerfCounterGroup. Smaller than a 9-case switch.
// Keep ordered to match the enum (FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4).
constexpr uint32_t cntl_reg_for_group[9] = {
    RISCV_DEBUG_REG_PERF_CNT_FPU0,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,    // UNPACK
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_0
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_1
    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,  // INSTRN
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_2
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_3
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_4
};

FORCE_INLINE uint32_t get_cntl_register_for_counter_group(PerfCounterGroup counter_group) {
    return cntl_reg_for_group[static_cast<uint32_t>(counter_group)];
}

// Shared: sets the L1 mux select (bank 0..4) for the given group. 0 for non-L1 groups (unused).
constexpr uint32_t mux_sel_for_group[9] = {
    0,  // FPU (unused)
    0,  // PACK (unused)
    0,  // UNPACK (unused)
    0,  // L1_0 → bank 0
    1,  // L1_1 → bank 1
    0,  // INSTRN (unused)
    2,  // L1_2 → bank 2
    3,  // L1_3 → bank 3
    4,  // L1_4 → bank 4
};

FORCE_INLINE void set_l1_mux_ctrl(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr uint32_t* mux_reg =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
    uint32_t mux_sel = mux_sel_for_group[static_cast<uint32_t>(counter_group)];
    *mux_reg = (*mux_reg & ~L1_MUX_MASK) | (mux_sel << 4);
}

#if COMPILE_FOR_TRISC == 1
// --- TRISC1-only: start/stop counters around the compute kernel ------------

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group) {
    if (counter_group >= PerfCounterGroup::L1_0 && counter_group != PerfCounterGroup::INSTRN) {
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

void start_perf_counter() {
    for (uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            start_single_group(counter_group_flags[i].first);
        }
    }
}

void stop_perf_counter() {
    for (uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            stop_single_group(counter_group_flags[i].first);
        }
    }
}

struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

#endif  // COMPILE_FOR_TRISC == 1

#if defined(COMPILE_FOR_BRISC)
// --- BRISC-only: counter readout and DRAM push -----------------------------

// Lookup tables indexed by PerfCounterGroup (same ordering as cntl_reg_for_group).
constexpr uint32_t read_reg_for_group[9] = {
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK,    // UNPACK
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_0
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_1
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD,  // INSTRN
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_2
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_3
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_4
};

constexpr uint32_t num_counters_for_group[9] = {
    NUM_FPU_COUNTERS,     // FPU
    NUM_PACK_COUNTERS,    // PACK
    NUM_UNPACK_COUNTERS,  // UNPACK
    NUM_L1_0_COUNTERS,    // L1_0
    NUM_L1_1_COUNTERS,    // L1_1
    NUM_INSTRN_COUNTERS,  // INSTRN
    NUM_L1_2_COUNTERS,    // L1_2
    NUM_L1_3_COUNTERS,    // L1_3
    NUM_L1_4_COUNTERS,    // L1_4
};

constexpr const std::pair<PerfCounterType, uint16_t>* counters_for_group[9] = {
    fpu_counters.data(),     // FPU
    pack_counters.data(),    // PACK
    unpack_counters.data(),  // UNPACK
    l1_0_counters.data(),    // L1_0
    l1_1_counters.data(),    // L1_1
    instrn_counters.data(),  // INSTRN
    l1_2_counters.data(),    // L1_2
    l1_3_counters.data(),    // L1_3
    l1_4_counters.data(),    // L1_4
};

FORCE_INLINE uint32_t get_read_register_for_counter_group(PerfCounterGroup g) {
    return read_reg_for_group[static_cast<uint32_t>(g)];
}

FORCE_INLINE uint32_t get_num_counters_for_counter_group(PerfCounterGroup g) {
    return num_counters_for_group[static_cast<uint32_t>(g)];
}

FORCE_INLINE const std::pair<PerfCounterType, uint16_t>* get_counters_for_counter_group(PerfCounterGroup g) {
    return counters_for_group[static_cast<uint32_t>(g)];
}

__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group) {
    if (counter_group >= PerfCounterGroup::L1_0 && counter_group != PerfCounterGroup::INSTRN) {
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
        // Readback poll: MMIO fence for the mux select.
        while (cntl_reg[1] != expected_mode);
        uint32_t ref_cnt_val = read_reg[0];
        uint32_t counter_val = read_reg[1];
        PerfCounter counter(counter_val, ref_cnt_val, counters[i].first);
        kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>(
            kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * 2);
        kernel_profiler::timeStampedData<
            PERF_COUNTER_PROFILER_ID,
            kernel_profiler::DoingDispatch::DISPATCH,
            kernel_profiler::PacketTypes::TS_DATA_16B>(counter.raw_data_1, counter.raw_data_2);
    }
    // Toggle start bit to clear the counters for this group
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

void read_perf_counters() {
    if (kernel_profiler::get_profiler_zone_valid()) {
        return;
    }
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_FPU
    read_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_PACK
    read_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_UNPACK
    read_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_0
    read_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_1
    read_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_INSTRN
    read_single_group(PerfCounterGroup::INSTRN);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_2
    read_single_group(PerfCounterGroup::L1_2);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_3
    read_single_group(PerfCounterGroup::L1_3);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_4
    read_single_group(PerfCounterGroup::L1_4);
#endif
}

#endif  // COMPILE_FOR_BRISC

}  // namespace kernel_profiler

#if COMPILE_FOR_TRISC == 1
#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;
#else
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()
#endif

#if defined(COMPILE_FOR_BRISC)
#define ReadPerfCounters() kernel_profiler::read_perf_counters();
#else
#define ReadPerfCounters()
#endif

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters()
#define RecordPerfCounters()

#endif
