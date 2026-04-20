// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <utility>

// Blackhole-specific perf counter arrays.
// Included by perf_counters.hpp after PerfCounterType enum is defined.

// FPU bank (2 banks, same on WH and BH):
//   sel 0 req   = th_fpu_op_valid       (FPU_COUNTER)
//   sel 1 req   = th_sfpu_op_valid_s1   (SFPU_COUNTER)
//   sel 257 grant = th_sfpu_op_valid_s1 | th_fpu_op_valid  (MATH_COUNTER)
// Sel 256 grant is fpu_req_ready — a driven signal but not useful as a utilization metric;
// we prefer sel 257 grant which counts "any FPU/SFPU op issued".
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 3> fpu_counters = {
    {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257}}};
constexpr size_t NUM_FPU_COUNTERS = 3;

// BH TDMA_UNPACK: 8 req + 8 grant counters read (live in RTL).
// Grant banks 4-6 (sels 260-262) have identical wiring on WH and BH (verified in RTL):
//   grant[4] (sel 260) = srcB not blocked by write port   (dma_srcb_wr_port_avail)
//   grant[5] (sel 261) = srcA not blocked by overwrite    (srca_write_ready)
//   grant[6] (sel 262) = srcA not blocked by write port   (dma_srca_wr_port_avail)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 18> unpack_counters = {
    {{PerfCounterType::MATH_SRC_DATA_READY, 0},
     {PerfCounterType::DATA_HAZARD_STALLS_MOVD2A, 1},
     {PerfCounterType::MATH_INSTRN_STARTED, 3},
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4},
     {PerfCounterType::SRCB_WRITE_AVAILABLE, 5},
     {PerfCounterType::SRCA_WRITE_AVAILABLE, 6},
     {PerfCounterType::UNPACK0_BUSY_THREAD0, 7},
     {PerfCounterType::UNPACK1_BUSY_THREAD0, 8},
     {PerfCounterType::UNPACK0_BUSY_THREAD1, 9},
     {PerfCounterType::UNPACK1_BUSY_THREAD1, 10},
     {PerfCounterType::SRCB_WRITE_ACTUAL, 259},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 260},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 261},
     {PerfCounterType::SRCA_WRITE_ACTUAL, 262},
     {PerfCounterType::SRCA_WRITE_THREAD0, 263},
     {PerfCounterType::SRCB_WRITE_THREAD0, 264},
     {PerfCounterType::SRCA_WRITE_THREAD1, 265},
     {PerfCounterType::SRCB_WRITE_THREAD1, 266}}};
constexpr size_t NUM_UNPACK_COUNTERS = 18;

// BH TDMA_PACK: PACK_COUNT=1, 2 req + 3 grant live.
// Empirically-dead counters (RTL-live, filtered in Python):
//   MATH_INSTRN_STARTED (o_math_instrnbuf_rden never fires on BH)
//   WAITING_FOR_SFPU_IDLE_0/1/2 (empirically 0 across all workloads)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 5> pack_counters = {
    {{PerfCounterType::PACKER_DEST_READ_AVAILABLE, 11},
     {PerfCounterType::PACKER_BUSY, 18},
     {PerfCounterType::DEST_READ_GRANTED_0, 267},
     {PerfCounterType::MATH_NOT_STALLED_DEST_WR_PORT, 271},
     {PerfCounterType::AVAILABLE_MATH, 272}}};
constexpr size_t NUM_PACK_COUNTERS = 5;

// Tensix L1 bank 0 counters
// Tensix L1 bank 0 (MUX_CTRL[6:4] = 0): unpacker, TDMA bundles, ring0 NOC
// Port 1: BH has unified packer
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_0_counters = {
    {{PerfCounterType::L1_0_UNPACKER_0, 0},
     {PerfCounterType::L1_0_UNIFIED_PACKER, 1},
     {PerfCounterType::L1_0_TDMA_BUNDLE_0_RISC, 2},
     {PerfCounterType::L1_0_TDMA_BUNDLE_1_TRISC, 3},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_0, 4},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_1, 5},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_0, 6},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_1, 7},
     // Grant counters (counter_sel + 256 = out_fmt bit 16 set)
     {PerfCounterType::L1_0_UNPACKER_0_GRANT, 256},
     {PerfCounterType::L1_0_PORT1_GRANT, 257},
     {PerfCounterType::L1_0_TDMA_BUNDLE_0_GRANT, 258},
     {PerfCounterType::L1_0_TDMA_BUNDLE_1_GRANT, 259},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_0_GRANT, 260},
     {PerfCounterType::L1_0_NOC_RING0_OUTGOING_1_GRANT, 261},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_0_GRANT, 262},
     {PerfCounterType::L1_0_NOC_RING0_INCOMING_1_GRANT, 263}}};
constexpr size_t NUM_L1_0_COUNTERS = 16;

// Tensix L1 bank 1 (MUX_CTRL[6:4] = 1): packer/risc, ext unpacker, ring1 NOC
// Port 8: BH has RISC core
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_1_counters = {
    {{PerfCounterType::L1_1_RISC_CORE, 0},
     {PerfCounterType::L1_1_EXT_UNPACKER_1, 1},
     {PerfCounterType::L1_1_EXT_UNPACKER_2, 2},
     {PerfCounterType::L1_1_EXT_UNPACKER_3, 3},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_0, 4},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_1, 5},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_0, 6},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_1, 7},
     // Grant counters
     {PerfCounterType::L1_1_PORT8_GRANT, 256},
     {PerfCounterType::L1_1_EXT_UNPACKER_1_GRANT, 257},
     {PerfCounterType::L1_1_EXT_UNPACKER_2_GRANT, 258},
     {PerfCounterType::L1_1_EXT_UNPACKER_3_GRANT, 259},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_0_GRANT, 260},
     {PerfCounterType::L1_1_NOC_RING1_OUTGOING_1_GRANT, 261},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_0_GRANT, 262},
     {PerfCounterType::L1_1_NOC_RING1_INCOMING_1_GRANT, 263}}};
constexpr size_t NUM_L1_1_COUNTERS = 16;

// Tensix L1 bank 2 (BH only, MUX_CTRL[6:4] = 2): NOC Ring 2 ports 16-23
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_2_counters = {
    {{PerfCounterType::L1_2_NOC_RING2_PORT_0, 0},
     {PerfCounterType::L1_2_NOC_RING2_PORT_1, 1},
     {PerfCounterType::L1_2_NOC_RING2_PORT_2, 2},
     {PerfCounterType::L1_2_NOC_RING2_PORT_3, 3},
     {PerfCounterType::L1_2_NOC_RING2_PORT_4, 4},
     {PerfCounterType::L1_2_NOC_RING2_PORT_5, 5},
     {PerfCounterType::L1_2_NOC_RING2_PORT_6, 6},
     {PerfCounterType::L1_2_NOC_RING2_PORT_7, 7},
     {PerfCounterType::L1_2_NOC_RING2_PORT_0_GRANT, 256},
     {PerfCounterType::L1_2_NOC_RING2_PORT_1_GRANT, 257},
     {PerfCounterType::L1_2_NOC_RING2_PORT_2_GRANT, 258},
     {PerfCounterType::L1_2_NOC_RING2_PORT_3_GRANT, 259},
     {PerfCounterType::L1_2_NOC_RING2_PORT_4_GRANT, 260},
     {PerfCounterType::L1_2_NOC_RING2_PORT_5_GRANT, 261},
     {PerfCounterType::L1_2_NOC_RING2_PORT_6_GRANT, 262},
     {PerfCounterType::L1_2_NOC_RING2_PORT_7_GRANT, 263}}};
constexpr size_t NUM_L1_2_COUNTERS = 16;

// Tensix L1 bank 3 (BH only, MUX_CTRL[6:4] = 3): NOC Ring 3 ports 24-31
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_3_counters = {
    {{PerfCounterType::L1_3_NOC_RING3_PORT_0, 0},
     {PerfCounterType::L1_3_NOC_RING3_PORT_1, 1},
     {PerfCounterType::L1_3_NOC_RING3_PORT_2, 2},
     {PerfCounterType::L1_3_NOC_RING3_PORT_3, 3},
     {PerfCounterType::L1_3_NOC_RING3_PORT_4, 4},
     {PerfCounterType::L1_3_NOC_RING3_PORT_5, 5},
     {PerfCounterType::L1_3_NOC_RING3_PORT_6, 6},
     {PerfCounterType::L1_3_NOC_RING3_PORT_7, 7},
     {PerfCounterType::L1_3_NOC_RING3_PORT_0_GRANT, 256},
     {PerfCounterType::L1_3_NOC_RING3_PORT_1_GRANT, 257},
     {PerfCounterType::L1_3_NOC_RING3_PORT_2_GRANT, 258},
     {PerfCounterType::L1_3_NOC_RING3_PORT_3_GRANT, 259},
     {PerfCounterType::L1_3_NOC_RING3_PORT_4_GRANT, 260},
     {PerfCounterType::L1_3_NOC_RING3_PORT_5_GRANT, 261},
     {PerfCounterType::L1_3_NOC_RING3_PORT_6_GRANT, 262},
     {PerfCounterType::L1_3_NOC_RING3_PORT_7_GRANT, 263}}};
constexpr size_t NUM_L1_3_COUNTERS = 16;

// L1 bank 4 counters (BH only, MUX_CTRL[6:4] = 4): misc ports 32-39
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_4_counters = {
    {{PerfCounterType::L1_4_MISC_PORT_0, 0},
     {PerfCounterType::L1_4_MISC_PORT_1, 1},
     {PerfCounterType::L1_4_MISC_PORT_2, 2},
     {PerfCounterType::L1_4_MISC_PORT_3, 3},
     {PerfCounterType::L1_4_MISC_PORT_4, 4},
     {PerfCounterType::L1_4_MISC_PORT_5, 5},
     {PerfCounterType::L1_4_MISC_PORT_6, 6},
     {PerfCounterType::L1_4_MISC_PORT_7, 7},
     // Grant counters
     {PerfCounterType::L1_4_MISC_PORT_0_GRANT, 256},
     {PerfCounterType::L1_4_MISC_PORT_1_GRANT, 257},
     {PerfCounterType::L1_4_MISC_PORT_2_GRANT, 258},
     {PerfCounterType::L1_4_MISC_PORT_3_GRANT, 259},
     {PerfCounterType::L1_4_MISC_PORT_4_GRANT, 260},
     {PerfCounterType::L1_4_MISC_PORT_5_GRANT, 261},
     {PerfCounterType::L1_4_MISC_PORT_6_GRANT, 262},
     {PerfCounterType::L1_4_MISC_PORT_7_GRANT, 263}}};
constexpr size_t NUM_L1_4_COUNTERS = 16;

// BH: 3-bit L1 mux at MUX_CTRL[6:4], values 0-4
constexpr uint32_t L1_MUX_MASK = 0x7 << 4;

// BH INSTRN_THREAD: 82 counters
// Sel 27-30: shared stall conditions (1 slot each on BH)
// Sel 31-57: per-thread stall reasons (9 types x 3 threads)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 113> instrn_counters = {
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
     // Sel 24-26: total stall cycles per thread
     {PerfCounterType::THREAD_STALLS_0, 24},
     {PerfCounterType::THREAD_STALLS_1, 25},
     {PerfCounterType::THREAD_STALLS_2, 26},
     // Sel 27-30: shared stall conditions (1 slot each on BH)
     {PerfCounterType::WAITING_FOR_SRCA_CLEAR, 27},
     {PerfCounterType::WAITING_FOR_SRCB_CLEAR, 28},
     {PerfCounterType::WAITING_FOR_SRCA_VALID, 29},
     {PerfCounterType::WAITING_FOR_SRCB_VALID, 30},
     // Sel 31-57: per-thread stall reasons — THREAD-MAJOR layout
     // Sels 31-39 = 9 stall types for thread 0
     // Sels 40-48 = 9 stall types for thread 1
     // Sels 49-57 = 9 stall types for thread 2
     // Order within each thread: thcon, unpack, pack, math, sem_zero, sem_max, move, mmio, sfpu
     {PerfCounterType::WAITING_FOR_THCON_IDLE_0, 31},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_0, 32},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_0, 33},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_0, 34},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_0, 35},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_0, 36},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_0, 37},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_0, 38},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_0, 39},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_1, 40},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_1, 41},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_1, 42},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_1, 43},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_1, 44},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_1, 45},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_1, 46},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_1, 47},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_1, 48},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_2, 49},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_2, 50},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_2, 51},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_2, 52},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_2, 53},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_2, 54},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_2, 55},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_2, 56},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_2, 57},
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
     {PerfCounterType::PACK_INSTRN_ISSUED_2, 279},
     // Grant counters for shared stall conditions (grant = |inst_stall_thread)
     // BH: sels 27-30 → grant sels 283-286
     {PerfCounterType::STALL_GRANT_SRCA_CLEAR, 283},
     {PerfCounterType::STALL_GRANT_SRCB_CLEAR, 284},
     {PerfCounterType::STALL_GRANT_SRCA_VALID, 285},
     {PerfCounterType::STALL_GRANT_SRCB_VALID, 286},
     // Grant counters for per-thread stall reasons — THREAD-MAJOR
     // Thread 0 grants (sels 287-295)
     {PerfCounterType::STALL_GRANT_THCON_0, 287},
     {PerfCounterType::STALL_GRANT_UNPACK_0, 288},
     {PerfCounterType::STALL_GRANT_PACK_0, 289},
     {PerfCounterType::STALL_GRANT_MATH_0, 290},
     {PerfCounterType::STALL_GRANT_SEM_ZERO_0, 291},
     {PerfCounterType::STALL_GRANT_SEM_MAX_0, 292},
     {PerfCounterType::STALL_GRANT_MOVE_0, 293},
     {PerfCounterType::STALL_GRANT_MMIO_0, 294},
     {PerfCounterType::STALL_GRANT_SFPU_0, 295},
     // Thread 1 grants (sels 296-304)
     {PerfCounterType::STALL_GRANT_THCON_1, 296},
     {PerfCounterType::STALL_GRANT_UNPACK_1, 297},
     {PerfCounterType::STALL_GRANT_PACK_1, 298},
     {PerfCounterType::STALL_GRANT_MATH_1, 299},
     {PerfCounterType::STALL_GRANT_SEM_ZERO_1, 300},
     {PerfCounterType::STALL_GRANT_SEM_MAX_1, 301},
     {PerfCounterType::STALL_GRANT_MOVE_1, 302},
     {PerfCounterType::STALL_GRANT_MMIO_1, 303},
     {PerfCounterType::STALL_GRANT_SFPU_1, 304},
     // Thread 2 grants (sels 305-313)
     {PerfCounterType::STALL_GRANT_THCON_2, 305},
     {PerfCounterType::STALL_GRANT_UNPACK_2, 306},
     {PerfCounterType::STALL_GRANT_PACK_2, 307},
     {PerfCounterType::STALL_GRANT_MATH_2, 308},
     {PerfCounterType::STALL_GRANT_SEM_ZERO_2, 309},
     {PerfCounterType::STALL_GRANT_SEM_MAX_2, 310},
     {PerfCounterType::STALL_GRANT_MOVE_2, 311},
     {PerfCounterType::STALL_GRANT_MMIO_2, 312},
     {PerfCounterType::STALL_GRANT_SFPU_2, 313}}};
constexpr size_t NUM_INSTRN_COUNTERS = 113;
