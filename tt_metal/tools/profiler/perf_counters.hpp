// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
constexpr std::uint16_t PERF_COUNTER_PROFILER_ID = 9090;

enum PerfCounterGroup : std::uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4, L1_5 };
enum PerfCounterType : std::uint16_t {
    UNDEF = 0,
    // FPU Group
    FPU_COUNTER,
    SFPU_COUNTER,
    MATH_COUNTER,
    // TDMA_UNPACK Group
    MATH_SRC_DATA_READY,
    MATH_NOT_D2S_STALLED,
    MATH_FIDELITY_STALL,
    MATH_INSTRN_STARTED,
    MATH_INSTRN_AVAILABLE,
    SRCB_WRITE_REQ,
    SRCA_WRITE_REQ,
    UNPACK0_BUSY_THREAD0,
    UNPACK1_BUSY_THREAD0,
    UNPACK0_BUSY_THREAD1,
    UNPACK1_BUSY_THREAD1,
    MATH_INSTRN_HF_1_CYCLE,
    MATH_INSTRN_HF_2_CYCLE,
    MATH_INSTRN_HF_4_CYCLE,
    // TDMA_PACK Group
    PACKER0_DEST_READ_REQ,
    PACKER_BUSY,
    MATH_NOT_SCOREBOARD_STALLED,
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
    MATH_INSTRN_AVAILABLE_0,
    MATH_INSTRN_AVAILABLE_1,
    MATH_INSTRN_AVAILABLE_2,
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
    WAITING_FOR_CFG_IDLE_0,
    WAITING_FOR_CFG_IDLE_1,
    WAITING_FOR_CFG_IDLE_2,
    WAITING_FOR_SFPU_IDLE_0,
    WAITING_FOR_SFPU_IDLE_1,
    WAITING_FOR_SFPU_IDLE_2,
    // L1 Bank 0 (mux=0, ports 0-7)
    L1_0_UNPACKER_0,
    L1_0_UNPACKER_1_ECC_PACK1,  // Wormhole port 1; Blackhole uses L1_0_UNPACKER_1_ECC
    L1_0_TDMA_BUNDLE_0_RISC,
    L1_0_TDMA_BUNDLE_1_TRISC,
    L1_0_NOC_RING0_OUTGOING_0,
    L1_0_NOC_RING0_OUTGOING_1,
    L1_0_NOC_RING0_INCOMING_0,
    L1_0_NOC_RING0_INCOMING_1,
    // L1 Bank 1 (mux=1, ports 8-15)
    L1_1_TDMA_PACKER_2,  // Wormhole port 8; Blackhole uses L1_1_PACKER_IF_0
    L1_1_EXT_UNPACKER_1,  // Wormhole ports 9-11; Blackhole uses L1_1_UNPACKER1_EXT_IF_1-3
    L1_1_EXT_UNPACKER_2,
    L1_1_EXT_UNPACKER_3,
    L1_1_NOC_RING1_OUTGOING_0,
    L1_1_NOC_RING1_OUTGOING_1,
    L1_1_NOC_RING1_INCOMING_0,
    L1_1_NOC_RING1_INCOMING_1,
    L1_0_UNIFIED_PACKER,  // retired, kept so the ordinals below do not shift
    L1_1_RISC_CORE,       // retired, kept so the ordinals below do not shift
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
    SRCB_WRITE_NOT_BLOCKED_OVR,
    SRCA_WRITE_NOT_BLOCKED_OVR,
    SRCA_WRITE_NOT_BLOCKED_PORT,
    SRCB_WRITE_NOT_BLOCKED_PORT,
    SRCA_WRITE_TID_EVEN,
    SRCB_WRITE_TID_EVEN,
    SRCA_WRITE_TID_ODD,
    SRCB_WRITE_TID_ODD,
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
    // L1 Bank 4 (BH only, mux=4, ports 32-39): extended packers 6-7, packer L1 interface 1 (port 34, shared with the
    // tag-search accelerator, debug L1 RAM and timestamp), unpacker 0's extended read interfaces 1-5 (ports 35-39).
    L1_4_EXT_PACKER_6,
    L1_4_EXT_PACKER_7,
    L1_4_PACKER_IF_1_TAG_SEARCH,
    L1_4_UNPACKER0_EXT_IF_1,
    L1_4_UNPACKER0_EXT_IF_2,
    L1_4_UNPACKER0_EXT_IF_3,
    L1_4_UNPACKER0_EXT_IF_4,
    L1_4_UNPACKER0_EXT_IF_5,
    L1_4_EXT_PACKER_6_GRANT,
    L1_4_EXT_PACKER_7_GRANT,
    L1_4_PACKER_IF_1_TAG_SEARCH_GRANT,
    L1_4_UNPACKER0_EXT_IF_1_GRANT,
    L1_4_UNPACKER0_EXT_IF_2_GRANT,
    L1_4_UNPACKER0_EXT_IF_3_GRANT,
    L1_4_UNPACKER0_EXT_IF_4_GRANT,
    L1_4_UNPACKER0_EXT_IF_5_GRANT,
    // L1 Bank 2 (BH only, mux=2, ports 16-23): unpacker 1's extended read interfaces 4-7 (also used by the
    // packer L1-to-L1 read) and NOC ring 0 ports 2-3.
    L1_2_UNPACKER1_EXT_IF_4,
    L1_2_UNPACKER1_EXT_IF_5,
    L1_2_UNPACKER1_EXT_IF_6,
    L1_2_UNPACKER1_EXT_IF_7,
    L1_2_NOC_RING0_OUTGOING_2,
    L1_2_NOC_RING0_OUTGOING_3,
    L1_2_NOC_RING0_INCOMING_2,
    L1_2_NOC_RING0_INCOMING_3,
    L1_2_UNPACKER1_EXT_IF_4_GRANT,
    L1_2_UNPACKER1_EXT_IF_5_GRANT,
    L1_2_UNPACKER1_EXT_IF_6_GRANT,
    L1_2_UNPACKER1_EXT_IF_7_GRANT,
    L1_2_NOC_RING0_OUTGOING_2_GRANT,
    L1_2_NOC_RING0_OUTGOING_3_GRANT,
    L1_2_NOC_RING0_INCOMING_2_GRANT,
    L1_2_NOC_RING0_INCOMING_3_GRANT,
    // L1 Bank 3 (BH only, mux=3, ports 24-31: NOC ring 1 ports 2-3 and extended packers 2-5)
    L1_3_NOC_RING1_OUTGOING_2,
    L1_3_NOC_RING1_OUTGOING_3,
    L1_3_NOC_RING1_INCOMING_2,
    L1_3_NOC_RING1_INCOMING_3,
    L1_3_EXT_PACKER_2,
    L1_3_EXT_PACKER_3,
    L1_3_EXT_PACKER_4,
    L1_3_EXT_PACKER_5,
    L1_3_NOC_RING1_OUTGOING_2_GRANT,
    L1_3_NOC_RING1_OUTGOING_3_GRANT,
    L1_3_NOC_RING1_INCOMING_2_GRANT,
    L1_3_NOC_RING1_INCOMING_3_GRANT,
    L1_3_EXT_PACKER_2_GRANT,
    L1_3_EXT_PACKER_3_GRANT,
    L1_3_EXT_PACKER_4_GRANT,
    L1_3_EXT_PACKER_5_GRANT,
    ANY_THREAD_STALL,
    // L1 Bank 5 (BH only, mux=5, ports 40-41): unpacker 0's extended read interfaces 6-7; slots 2-7 read 0.
    L1_5_UNPACKER0_EXT_IF_6,
    L1_5_UNPACKER0_EXT_IF_7,
    L1_5_UNPACKER0_EXT_IF_6_GRANT,
    L1_5_UNPACKER0_EXT_IF_7_GRANT,
    // Blackhole ports whose client differs from Wormhole (tapeout RTL): port 1 has no packer, port 8 is the
    // packer's L1 interface 0, ports 9-11 are unpacker 1's extended read interfaces (also the packer L1-to-L1 read).
    L1_0_UNPACKER_1_ECC,
    L1_1_PACKER_IF_0,
    L1_1_UNPACKER1_EXT_IF_1,
    L1_1_UNPACKER1_EXT_IF_2,
    L1_1_UNPACKER1_EXT_IF_3,
    L1_0_UNPACKER_1_ECC_GRANT,
    L1_1_PACKER_IF_0_GRANT,
    L1_1_UNPACKER1_EXT_IF_1_GRANT,
    L1_1_UNPACKER1_EXT_IF_2_GRANT,
    L1_1_UNPACKER1_EXT_IF_3_GRANT,
    // === Quasar (A0) ===
    // Thread 3 variants (Quasar NEOs run 4 threads)
    CFG_INSTRN_AVAILABLE_3,
    SYNC_INSTRN_AVAILABLE_3,
    THCON_INSTRN_AVAILABLE_3,
    MATH_INSTRN_AVAILABLE_3,
    UNPACK_INSTRN_AVAILABLE_3,
    PACK_INSTRN_AVAILABLE_3,
    THREAD_STALLS_3,
    THREAD_INSTRUCTIONS_3,
    // Quasar-only instruction classes. XSEARCH requests are tied to 0 in the RTL and its grants alias
    // THREAD_INSTRUCTIONS, so it is not in the table; the enumerators stay so the ordinals below do not shift.
    XSEARCH_INSTRN_AVAILABLE_0,
    XSEARCH_INSTRN_AVAILABLE_1,
    XSEARCH_INSTRN_AVAILABLE_2,
    XSEARCH_INSTRN_AVAILABLE_3,
    INSTISSUE_INSTRN_AVAILABLE_0,
    INSTISSUE_INSTRN_AVAILABLE_1,
    INSTISSUE_INSTRN_AVAILABLE_2,
    INSTISSUE_INSTRN_AVAILABLE_3,
    // Stall reasons, OR-reduced across the 4 threads (Quasar has no per-thread reason counters)
    TILE_COUNTER_STALL_PACK,
    TILE_COUNTER_STALL_UNPACK,
    SRCS_STALL_PACK,
    SRCS_STALL_SFPU,
    SRCS_STALL_UNPACK,
    DEST_STALL_PACK,
    DEST_STALL_SFPU,
    DEST_STALL_MATH,
    DEST_STALL_UNPACK,
    SFPU_DATA_HAZARD_STALL,
    FPU_DATA_HAZARD_STALL,
    SRCB_STALL_UNPACK,
    SRCA_STALL_UNPACK,
    DVALID_STALL_MATH,
    SRCA_STALL_MATH,
    // The l1_client CSR event counter; records carry QUASAR_L1_CLIENT_EVENT_BASE + (subport*8 + event).
    QUASAR_L1_CLIENT_EVENT,
    // Quasar runs 3 unpackers per thread.
    UNPACK2_BUSY_THREAD0,
    // Values stay below 256: the l1_client selections are encoded above them in counter_type.
};
static_assert(UNPACK2_BUSY_THREAD0 <= 255, "PerfCounterType must leave 256 and up for l1_client selections");

// Quasar's l1_client counter has 296 selections (subport*8 + event) behind one enum value. A record
// carries QUASAR_L1_CLIENT_EVENT_BASE + selection in counter_type; the host maps it back.
constexpr std::uint32_t QUASAR_L1_CLIENT_EVENT_BASE = 256;
constexpr std::uint32_t QUASAR_L1_CLIENT_NUM_SELECTIONS = 37 * 8;  // subports x events, host-visible copy

union PerfCounter {
    struct {
        std::uint32_t counter_value;
        std::uint32_t ref_cnt;
        std::uint32_t counter_type : 16;
        // NEO the record came from: Quasar's DM0 reads all four; always 0 on tt-1xx.
        std::uint32_t neo : 4;
        std::uint32_t unused : 12;
    } __attribute__((packed));
    struct {
        std::uint64_t raw_data_1;
        std::uint64_t raw_data_2;
    } __attribute__((packed));

    PerfCounter() = delete;
    PerfCounter(
        std::uint32_t counter_value, std::uint32_t ref_cnt, PerfCounterType counter_type, std::uint32_t neo = 0) :
        counter_value(counter_value),
        ref_cnt(ref_cnt),
        counter_type(static_cast<std::uint32_t>(counter_type)),
        neo(neo) {}
    // Raw counter_type, for the l1_client encoding above the enum.
    PerfCounter(std::uint32_t counter_value, std::uint32_t ref_cnt, std::uint32_t counter_type_raw, std::uint32_t neo) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(counter_type_raw), neo(neo) {}

    PerfCounter(std::uint64_t raw_data_1, std::uint64_t raw_data_2) : raw_data_1(raw_data_1), raw_data_2(raw_data_2) {}
};
static_assert(sizeof(PerfCounter) == sizeof(std::uint64_t) * 2, "PerfCounter must be 128-bit");

// The RISC that orchestrates the kernel owns the counters. tt-1xx: TRISC1 arms and stops, BRISC reads
// once the TRISCs are done. Quasar: DM0 arms all four NEOs before GO and stops and reads them after DONE.
#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_DM)
#define PERF_COUNTER_WRAP_RISC 1
#define PERF_COUNTER_READ_RISC 1
#endif
#else
#if COMPILE_FOR_TRISC == 1
#define PERF_COUNTER_WRAP_RISC 1
#endif
#if defined(COMPILE_FOR_BRISC)
#define PERF_COUNTER_READ_RISC 1
#endif
#endif

#if defined(PROFILE_PERF_COUNTERS) && (defined(PERF_COUNTER_WRAP_RISC) || defined(PERF_COUNTER_READ_RISC))

#include "kernel_profiler.hpp"
#include "api/debug/assert.h"

namespace kernel_profiler {

// Quasar's DM firmware has a 2 KB RW data region and its linker script folds .rodata into it; the
// constant counter tables go to the 12 KB text region instead. tt-1xx has room and is unchanged.
#if defined(ARCH_QUASAR)
#define PERF_COUNTER_TABLE __attribute__((section(".text.perf_counter_tables")))
#else
#define PERF_COUNTER_TABLE
#endif

// Architecture-specific counter arrays (fpu, unpack, pack, l1_0-l1_5, instrn)
#if defined(ARCH_QUASAR)
#include "tt_metal/hw/inc/internal/tt-2xx/quasar/hw_counters.h"
#elif defined(ARCH_BLACKHOLE)
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
#define PROFILE_PERF_COUNTERS_L1_5 (1 << 9)

#if defined(ARCH_QUASAR) &&                                                                                            \
    ((PROFILE_PERF_COUNTERS) & (PROFILE_PERF_COUNTERS_L1_0 | PROFILE_PERF_COUNTERS_L1_1 | PROFILE_PERF_COUNTERS_L1_2 | \
                                PROFILE_PERF_COUNTERS_L1_3 | PROFILE_PERF_COUNTERS_L1_4 | PROFILE_PERF_COUNTERS_L1_5))
#error "Quasar has no L1 perf counter groups; valid bits are FPU(1)|PACK(2)|UNPACK(4)|INSTRN(32) = 39"
#endif

#define PERF_CNT_CONTINUOUS_MODE 0
#define PERF_CNT_BANK_SELECT_SHIFT 8
#define PERF_CNT_START_VALUE 1
#define PERF_CNT_STOP_VALUE 2

// Counter groups and their corresponding enable bitmask bits. Shared; used on both
// TRISC1 (start/stop loop) and BRISC (read loop).
constexpr std::pair<PerfCounterGroup, std::uint32_t> counter_group_flags[] PERF_COUNTER_TABLE = {
    {PerfCounterGroup::FPU, PROFILE_PERF_COUNTERS_FPU},
    {PerfCounterGroup::PACK, PROFILE_PERF_COUNTERS_PACK},
    {PerfCounterGroup::UNPACK, PROFILE_PERF_COUNTERS_UNPACK},
#if !defined(ARCH_QUASAR)
    {PerfCounterGroup::L1_0, PROFILE_PERF_COUNTERS_L1_0},
    {PerfCounterGroup::L1_1, PROFILE_PERF_COUNTERS_L1_1},
#endif
    {PerfCounterGroup::INSTRN, PROFILE_PERF_COUNTERS_INSTRN},
#if !defined(ARCH_QUASAR)
    {PerfCounterGroup::L1_2, PROFILE_PERF_COUNTERS_L1_2},
    {PerfCounterGroup::L1_3, PROFILE_PERF_COUNTERS_L1_3},
    {PerfCounterGroup::L1_4, PROFILE_PERF_COUNTERS_L1_4},
    {PerfCounterGroup::L1_5, PROFILE_PERF_COUNTERS_L1_5},
#endif
};
constexpr std::uint32_t NUM_COUNTER_GROUPS = sizeof(counter_group_flags) / sizeof(counter_group_flags[0]);

// Indexed by PerfCounterGroup; keep the enum order.
#if defined(ARCH_QUASAR)
// L1 slots are 0: Quasar has no L1 counter unit and the L1 group bits are rejected at compile time.
constexpr std::uint32_t cntl_reg_for_group[10] PERF_COUNTER_TABLE = {
    RISCV_DEBUG_REG_PERF_CNT_FPU0,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,    // UNPACK
    0,                                        // L1_0
    0,                                        // L1_1
    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,  // INSTRN
    0,                                        // L1_2
    0,                                        // L1_3
    0,                                        // L1_4
    0,                                        // L1_5
};
#else
constexpr std::uint32_t cntl_reg_for_group[10] = {
    RISCV_DEBUG_REG_PERF_CNT_FPU0,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,    // UNPACK
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_0
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_1
    RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,  // INSTRN
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_2
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_3
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_4
    RISCV_DEBUG_REG_PERF_CNT_L1_0,            // L1_5
};
#endif

FORCE_INLINE std::uint32_t get_cntl_register_for_counter_group(PerfCounterGroup counter_group) {
    return cntl_reg_for_group[static_cast<std::uint32_t>(counter_group)];
}

// Quasar: NEO n's registers sit n strides above NEO0's. tt-1xx has one Tensix and no offset.
#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_NEOS = 4;
static_assert(
    NEO_REGS_1__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR -
            NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR ==
        QUASAR_NEO_REG_STRIDE,
    "l1_client CSRs must follow the per-NEO stride of the debug registers");
FORCE_INLINE std::uint32_t neo_reg_offset(std::uint32_t neo) { return neo * QUASAR_NEO_REG_STRIDE; }
#else
constexpr std::uint32_t NUM_NEOS = 1;
FORCE_INLINE std::uint32_t neo_reg_offset(std::uint32_t) { return 0; }
#endif

#if defined(ARCH_QUASAR)
// Quasar's profiler is L1-only (about 80 records per risc buffer), so DM0 files each NEO's readout into that
// NEO's own TRISC buffers (complete after wait_subordinates()); the record's neo field says which NEO.
constexpr std::uint32_t TRISCS_PER_NEO = 4;
constexpr std::uint32_t PERF_RECORD_WORDS = PROFILER_L1_MARKER_UINT32_SIZE * 3;  // marker + two 64-bit words

FORCE_INLINE std::uint32_t first_trisc_of(std::uint32_t neo) {
    return static_cast<std::uint32_t>(TensixProcessorTypes::E0_MATH0) + neo * TRISCS_PER_NEO;
}

FORCE_INLINE bool neo_enabled(std::uint32_t enables, std::uint32_t neo) {
    return (enables >> first_trisc_of(neo)) & 1u;
}

struct SpillCursor {
    std::uint32_t risc;
    std::uint32_t last_risc;
    std::uint32_t index;
};
static SpillCursor spill;

FORCE_INLINE std::uint32_t spill_end_index(std::uint32_t risc) {
    std::uint32_t index = profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + risc];
    // A TRISC that published nothing has no run sentinel for the host to anchor on; leave it alone.
    return index < CUSTOM_MARKERS ? PROFILER_L1_VECTOR_SIZE : index;
}

inline void spill_open(std::uint32_t neo) {
    spill.risc = first_trisc_of(neo);
    spill.last_risc = spill.risc + TRISCS_PER_NEO - 1;
    spill.index = spill_end_index(spill.risc);
}

inline void spill_close() { profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + spill.risc] = spill.index; }

inline void spill_record(const PerfCounter& counter) {
    while (spill.index + PERF_RECORD_WORDS > PROFILER_L1_VECTOR_SIZE) {
        if (spill.risc == spill.last_risc) {
            mark_dropped_timestamps(spill.risc);  // all four buffers full: let the host's dropped-markers warning fire
            return;
        }
        spill_close();
        spill.risc++;
        spill.index = spill_end_index(spill.risc);
    }
    volatile tt_l1_ptr std::uint32_t* data = profiler_data_buffer[spill.risc].data;
    std::uint64_t wall_clock = quasar_read_wall_clock_64();
    data[spill.index] =
        PROFILER_MARKER_VALID |
        ((get_const_id(PERF_COUNTER_PROFILER_ID, PacketTypes::TS_DATA_16B) & PROFILER_MARKER_TIMER_ID_MASK)
         << PROFILER_MARKER_TIMER_ID_SHIFT) |
        (static_cast<std::uint32_t>(wall_clock >> 32) & PROFILER_MARKER_TS_HIGH_MASK);
    data[spill.index + 1] = static_cast<std::uint32_t>(wall_clock);
    data[spill.index + 2] = counter.raw_data_1 >> 32;
    data[spill.index + 3] = static_cast<std::uint32_t>(counter.raw_data_1);
    data[spill.index + 4] = counter.raw_data_2 >> 32;
    data[spill.index + 5] = static_cast<std::uint32_t>(counter.raw_data_2);
    spill.index += PERF_RECORD_WORDS;
}
#endif  // ARCH_QUASAR

inline void emit_record(const PerfCounter& counter) {
#if defined(ARCH_QUASAR)
    spill_record(counter);
#else
    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>(
        kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * 2);
    kernel_profiler::timeStampedData<
        PERF_COUNTER_PROFILER_ID,
        kernel_profiler::DoingDispatch::DISPATCH,
        kernel_profiler::PacketTypes::TS_DATA_16B>(counter.raw_data_1, counter.raw_data_2);
#endif
}

#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
constexpr std::uint32_t QUASAR_L1_CLIENT_SEL = PROFILE_PERF_COUNTERS_L1_SEL;
static_assert(
    QUASAR_L1_CLIENT_SEL < QUASAR_L1_CLIENT_NUM_SUBPORTS * QUASAR_L1_CLIENT_NUM_EVENTS,
    "PROFILE_PERF_COUNTERS_L1_SEL must be subport*8 + event, below 296");
static_assert(
    QUASAR_L1_CLIENT_NUM_SUBPORTS * QUASAR_L1_CLIENT_NUM_EVENTS == QUASAR_L1_CLIENT_NUM_SELECTIONS,
    "host copy of the l1_client selection count is stale");
static_assert(
    QUASAR_L1_CLIENT_EVENT_BASE + QUASAR_L1_CLIENT_NUM_SUBPORTS * QUASAR_L1_CLIENT_NUM_EVENTS < (1u << 16),
    "l1_client encoding must fit the 16-bit counter_type");
// The CSR has no reference counter; the wall-clock span from arming to freezing is the denominator.
static std::uint32_t l1_client_start_cycles;
static std::uint32_t l1_client_elapsed_cycles;

inline void start_l1_client_event_counter(std::uint32_t neo) {
    constexpr std::uint32_t subport = QUASAR_L1_CLIENT_SEL / QUASAR_L1_CLIENT_NUM_EVENTS;
    constexpr std::uint32_t event = QUASAR_L1_CLIENT_SEL % QUASAR_L1_CLIENT_NUM_EVENTS;
    volatile tt_reg_ptr std::uint32_t* ctrl = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CTRL + neo_reg_offset(neo));
    volatile tt_reg_ptr std::uint32_t* cnt = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CNT + neo_reg_offset(neo));
    *ctrl = (subport << QUASAR_L1_CLIENT_PERF_SUBPORT_SHIFT) | (event << QUASAR_L1_CLIENT_PERF_EVENT_SHIFT) |
            QUASAR_L1_CLIENT_PERF_CTRL_ENABLE;
    // Clear-on-read: drop anything accumulated before this window.
    (void)*cnt;
}

inline void stop_l1_client_event_counter(std::uint32_t neo) {
    volatile tt_reg_ptr std::uint32_t* ctrl = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CTRL + neo_reg_offset(neo));
    *ctrl = 0;
}

inline void read_l1_client_event_counter(std::uint32_t neo) {
    volatile tt_reg_ptr std::uint32_t* cnt = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CNT + neo_reg_offset(neo));
    PerfCounter counter(*cnt, l1_client_elapsed_cycles, QUASAR_L1_CLIENT_EVENT_BASE + QUASAR_L1_CLIENT_SEL, neo);
    emit_record(counter);
}
#endif  // ARCH_QUASAR && PROFILE_PERF_COUNTERS_L1_SEL

#if !defined(ARCH_QUASAR)
// Shared: sets the L1 mux select (bank 0..5) for the given group. 0 for non-L1 groups (unused).
constexpr std::uint32_t mux_sel_for_group[10] = {
    0,  // FPU (unused)
    0,  // PACK (unused)
    0,  // UNPACK (unused)
    0,  // L1_0 → bank 0
    1,  // L1_1 → bank 1
    0,  // INSTRN (unused)
    2,  // L1_2 → bank 2
    3,  // L1_3 → bank 3
    4,  // L1_4 → bank 4
    5,  // L1_5 → bank 5
};

FORCE_INLINE void set_l1_mux_ctrl(PerfCounterGroup counter_group) {
    volatile tt_reg_ptr std::uint32_t* mux_reg =
        reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
    std::uint32_t mux_sel = mux_sel_for_group[static_cast<std::uint32_t>(counter_group)];
    *mux_reg = (*mux_reg & ~L1_MUX_MASK) | (mux_sel << 4);
}
#endif  // !ARCH_QUASAR

#if defined(PERF_COUNTER_WRAP_RISC)
// --- Wrap-thread only: start/stop counters around the compute kernel -------

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group, std::uint32_t neo) {
#if !defined(ARCH_QUASAR)
    if (counter_group >= PerfCounterGroup::L1_0 && counter_group != PerfCounterGroup::INSTRN) {
        set_l1_mux_ctrl(counter_group);
    }
#endif
    volatile tt_reg_ptr std::uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        get_cntl_register_for_counter_group(counter_group) + neo_reg_offset(neo));
    cntl_reg[0] = 0xFFFFFFFF;
    cntl_reg[1] = PERF_CNT_CONTINUOUS_MODE;
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

__attribute__((noinline)) void stop_single_group(PerfCounterGroup counter_group, std::uint32_t neo) {
    volatile tt_reg_ptr std::uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        get_cntl_register_for_counter_group(counter_group) + neo_reg_offset(neo));
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_STOP_VALUE;
}

void start_perf_counter() {
    for (std::uint32_t neo = 0; neo < NUM_NEOS; neo++) {
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        start_l1_client_event_counter(neo);
#endif
        for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
            if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
                start_single_group(counter_group_flags[i].first, neo);
            }
        }
    }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
    l1_client_start_cycles = static_cast<std::uint32_t>(quasar_read_wall_clock_64());
#endif
}

void stop_perf_counter() {
    for (std::uint32_t neo = 0; neo < NUM_NEOS; neo++) {
        for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
            if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
                stop_single_group(counter_group_flags[i].first, neo);
            }
        }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        stop_l1_client_event_counter(neo);
#endif
    }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
    l1_client_elapsed_cycles = static_cast<std::uint32_t>(quasar_read_wall_clock_64()) - l1_client_start_cycles;
#endif
}

#endif  // PERF_COUNTER_WRAP_RISC

#if defined(PERF_COUNTER_READ_RISC)
// --- Readout thread only: counter readout (BRISC on tt-1xx, DM0 on Quasar) ---

// Lookup tables indexed by PerfCounterGroup (same ordering as cntl_reg_for_group).
#if defined(ARCH_QUASAR)
constexpr std::uint32_t read_reg_for_group[10] PERF_COUNTER_TABLE = {
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK,    // UNPACK
    0,                                             // L1_0
    0,                                             // L1_1
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD,  // INSTRN
    0,                                             // L1_2
    0,                                             // L1_3
    0,                                             // L1_4
    0,                                             // L1_5
};

constexpr std::uint32_t num_counters_for_group[10] PERF_COUNTER_TABLE = {
    NUM_FPU_COUNTERS,     // FPU
    NUM_PACK_COUNTERS,    // PACK
    NUM_UNPACK_COUNTERS,  // UNPACK
    0,                    // L1_0
    0,                    // L1_1
    NUM_INSTRN_COUNTERS,  // INSTRN
    0,                    // L1_2
    0,                    // L1_3
    0,                    // L1_4
    0,                    // L1_5
};

constexpr const std::pair<PerfCounterType, std::uint16_t>* counters_for_group[10] PERF_COUNTER_TABLE = {
    fpu_counters.data(),     // FPU
    pack_counters.data(),    // PACK
    unpack_counters.data(),  // UNPACK
    nullptr,                 // L1_0
    nullptr,                 // L1_1
    instrn_counters.data(),  // INSTRN
    nullptr,                 // L1_2
    nullptr,                 // L1_3
    nullptr,                 // L1_4
    nullptr,                 // L1_5
};
#else
constexpr std::uint32_t read_reg_for_group[10] = {
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU,            // FPU
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK,      // PACK
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK,    // UNPACK
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_0
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_1
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD,  // INSTRN
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_2
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_3
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_4
    RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1,         // L1_5
};

constexpr std::uint32_t num_counters_for_group[10] = {
    NUM_FPU_COUNTERS,     // FPU
    NUM_PACK_COUNTERS,    // PACK
    NUM_UNPACK_COUNTERS,  // UNPACK
    NUM_L1_0_COUNTERS,    // L1_0
    NUM_L1_1_COUNTERS,    // L1_1
    NUM_INSTRN_COUNTERS,  // INSTRN
    NUM_L1_2_COUNTERS,    // L1_2
    NUM_L1_3_COUNTERS,    // L1_3
    NUM_L1_4_COUNTERS,    // L1_4
    NUM_L1_5_COUNTERS,    // L1_5
};

constexpr const std::pair<PerfCounterType, std::uint16_t>* counters_for_group[10] = {
    fpu_counters.data(),     // FPU
    pack_counters.data(),    // PACK
    unpack_counters.data(),  // UNPACK
    l1_0_counters.data(),    // L1_0
    l1_1_counters.data(),    // L1_1
    instrn_counters.data(),  // INSTRN
    l1_2_counters.data(),    // L1_2
    l1_3_counters.data(),    // L1_3
    l1_4_counters.data(),    // L1_4
    l1_5_counters.data(),    // L1_5
};
#endif  // ARCH_QUASAR

FORCE_INLINE std::uint32_t get_read_register_for_counter_group(PerfCounterGroup g) {
    return read_reg_for_group[static_cast<std::uint32_t>(g)];
}

FORCE_INLINE std::uint32_t get_num_counters_for_counter_group(PerfCounterGroup g) {
    return num_counters_for_group[static_cast<std::uint32_t>(g)];
}

FORCE_INLINE const std::pair<PerfCounterType, std::uint16_t>* get_counters_for_counter_group(PerfCounterGroup g) {
    return counters_for_group[static_cast<std::uint32_t>(g)];
}

__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group, std::uint32_t neo) {
#if !defined(ARCH_QUASAR)
    if (counter_group >= PerfCounterGroup::L1_0 && counter_group != PerfCounterGroup::INSTRN) {
        set_l1_mux_ctrl(counter_group);
    }
#endif
    volatile tt_reg_ptr std::uint32_t* cntl_reg = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        get_cntl_register_for_counter_group(counter_group) + neo_reg_offset(neo));
    volatile tt_reg_ptr std::uint32_t* read_reg = reinterpret_cast<volatile tt_reg_ptr std::uint32_t*>(
        get_read_register_for_counter_group(counter_group) + neo_reg_offset(neo));
    const auto* counters = get_counters_for_counter_group(counter_group);
    const std::uint32_t counters_size = get_num_counters_for_counter_group(counter_group);
    for (unsigned int i = 0; i < counters_size; i++) {
        std::uint32_t counter_sel = counters[i].second;
        std::uint32_t expected_mode = counter_sel << PERF_CNT_BANK_SELECT_SHIFT | PERF_CNT_CONTINUOUS_MODE;
        cntl_reg[1] = expected_mode;
        // Readback poll: MMIO fence for the mux select.
        while (cntl_reg[1] != expected_mode);
        std::uint32_t ref_cnt_val = read_reg[0];
        std::uint32_t counter_val = read_reg[1];
        PerfCounter counter(counter_val, ref_cnt_val, counters[i].first, neo);
        emit_record(counter);
    }
    // Toggle start bit to clear the counters for this group
    cntl_reg[2] = 0;
    cntl_reg[2] = PERF_CNT_START_VALUE;
}

// trisc_enables is the launch message's processor enable mask; Quasar skips NEOs that ran nothing.
void read_perf_counters([[maybe_unused]] std::uint32_t trisc_enables) {
    if (kernel_profiler::get_profiler_zone_invalid()) {
        return;
    }
    for (std::uint32_t neo = 0; neo < NUM_NEOS; neo++) {
#if defined(ARCH_QUASAR)
        if (!neo_enabled(trisc_enables, neo)) {
            continue;
        }
        spill_open(neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_FPU
        read_single_group(PerfCounterGroup::FPU, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_PACK
        read_single_group(PerfCounterGroup::PACK, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_UNPACK
        read_single_group(PerfCounterGroup::UNPACK, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_0
        read_single_group(PerfCounterGroup::L1_0, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_1
        read_single_group(PerfCounterGroup::L1_1, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_INSTRN
        read_single_group(PerfCounterGroup::INSTRN, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_2
        read_single_group(PerfCounterGroup::L1_2, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_3
        read_single_group(PerfCounterGroup::L1_3, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_4
        read_single_group(PerfCounterGroup::L1_4, neo);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_5
        read_single_group(PerfCounterGroup::L1_5, neo);
#endif
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        read_l1_client_event_counter(neo);
#endif
#if defined(ARCH_QUASAR)
        spill_close();
#endif
    }
}

#endif  // PERF_COUNTER_READ_RISC

// WRAP_RISC arms and stops the counters around the kernel; READ_RISC collects them. Both are
// chosen per architecture at the top of this file.
#if defined(PERF_COUNTER_WRAP_RISC)

struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};
#endif  // PERF_COUNTER_WRAP_RISC

}  // namespace kernel_profiler

#if defined(PERF_COUNTER_WRAP_RISC)
#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;
#else
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()
#endif

#if defined(PERF_COUNTER_READ_RISC)
#define ReadPerfCounters(trisc_enables) kernel_profiler::read_perf_counters(trisc_enables);
#else
#define ReadPerfCounters(trisc_enables)
#endif

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters(trisc_enables)
#define RecordPerfCounters()

#endif
