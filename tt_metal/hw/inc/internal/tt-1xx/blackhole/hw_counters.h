// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Blackhole-specific perf counter arrays.
// Included by perf_counters.hpp after PerfCounterType enum is defined.

// BH TDMA_UNPACK: 11 banks, 22 counter_sels (11 req + 11 grant).
// BH grant banks 4-6 (sels 260-262) have DIFFERENT wiring than WH despite identical RTL source.
// Empirically verified on BH silicon — the original BH mapping is correct:
//   grant[4] (sel 260) = srcA not blocked by overwrite
//   grant[5] (sel 261) = srcA write actual (not blocked by port)
//   grant[6] (sel 262) = srcB not blocked by port
// On WH, these are swapped: 260=srcB port, 261=srcA overwrite, 262=srcA port.
// fidelity_phases_ongoing = 1'b0 on BH, so FIDELITY_PHASE_STALLS (sel 2) is always 0.
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 22> unpack_counters = {
    {{PerfCounterType::MATH_SRC_DATA_READY, 0},
     {PerfCounterType::DATA_HAZARD_STALLS_MOVD2A, 1},
     {PerfCounterType::FIDELITY_PHASE_STALLS, 2},
     {PerfCounterType::MATH_INSTRN_STARTED, 3},
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4},
     {PerfCounterType::SRCB_WRITE_AVAILABLE, 5},
     {PerfCounterType::SRCA_WRITE_AVAILABLE, 6},
     {PerfCounterType::UNPACK0_BUSY_THREAD0, 7},
     {PerfCounterType::UNPACK1_BUSY_THREAD0, 8},
     {PerfCounterType::UNPACK0_BUSY_THREAD1, 9},
     {PerfCounterType::UNPACK1_BUSY_THREAD1, 10},
     {PerfCounterType::MATH_INSTRN_NOT_BLOCKED_SRC, 256},  // dead: hf_cycles never 2'b11
     {PerfCounterType::INSTRN_2_HF_CYCLES, 257},           // dead: hf_cycles never 2'b01
     {PerfCounterType::INSTRN_1_HF_CYCLE, 258},
     {PerfCounterType::SRCB_WRITE_ACTUAL, 259},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 260},   // BH grant wiring differs from WH
     {PerfCounterType::SRCA_WRITE_ACTUAL, 261},             // empirically verified on BH silicon
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 262},
     {PerfCounterType::SRCA_WRITE_THREAD0, 263},
     {PerfCounterType::SRCB_WRITE_THREAD0, 264},
     {PerfCounterType::SRCA_WRITE_THREAD1, 265},
     {PerfCounterType::SRCB_WRITE_THREAD1, 266}}};
constexpr size_t NUM_UNPACK_COUNTERS = 22;

// BH TDMA_PACK: 8 banks, 16 counter_sels (8 req + 8 grant).
// Empirically verified across 8 diverse workloads on BH silicon:
//   LIVE:  11(DEST_READ_AVAIL), 12(DEST_READ_1*), 267(GRANTED_0), 268(GRANTED_1*),
//          272(AVAILABLE_MATH), 273(BANK6_GRANT*)   (*RTL shows tied-to-0 but silicon disagrees)
//   DEAD:  13-14(DEST_READ_2/3), 15-17(BUSY_0/1/2), 18(BUSY), 269-270(GRANTED_2/3), 274(BANK7)
//   RTL-predicted-live but empirically dead: 18(PACKER_BUSY), 271(MATH_NOT_STALLED_DEST_WR_PORT)
// 15 RTL-confirmed dead counters are filtered out in perf_counter_analysis.py.
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> pack_counters = {
    {{PerfCounterType::PACKER_DEST_READ_AVAILABLE, 11},
     {PerfCounterType::PACKER_DEST_READ_1, 12},
     {PerfCounterType::PACKER_DEST_READ_2, 13},
     {PerfCounterType::PACKER_DEST_READ_3, 14},
     {PerfCounterType::PACKER_BUSY_0, 15},
     {PerfCounterType::PACKER_BUSY_1, 16},
     {PerfCounterType::PACKER_BUSY_2, 17},
     {PerfCounterType::PACKER_BUSY, 18},
     {PerfCounterType::DEST_READ_GRANTED_0, 267},
     {PerfCounterType::DEST_READ_GRANTED_1, 268},
     {PerfCounterType::DEST_READ_GRANTED_2, 269},
     {PerfCounterType::DEST_READ_GRANTED_3, 270},
     {PerfCounterType::MATH_NOT_STALLED_DEST_WR_PORT, 271},
     {PerfCounterType::AVAILABLE_MATH, 272},
     {PerfCounterType::PACK_BANK6_GRANT, 273},
     {PerfCounterType::PACK_BANK7_GRANT, 274}}};
constexpr size_t NUM_PACK_COUNTERS = 16;

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

// BH INSTRN_THREAD: 82 counters
// Sel 27-30: shared stall conditions (1 slot each on BH)
// Sel 31-57: per-thread stall reasons (9 types x 3 threads)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 82> instrn_counters = {
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
     // Sel 31-57: per-thread stall reasons (9 types x 3 threads)
     {PerfCounterType::WAITING_FOR_THCON_IDLE_0, 31},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_1, 32},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_2, 33},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_0, 34},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_1, 35},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_2, 36},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_0, 37},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_1, 38},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_2, 39},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_0, 40},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_1, 41},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_2, 42},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_0, 43},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_1, 44},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_2, 45},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_0, 46},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_1, 47},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_2, 48},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_0, 49},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_1, 50},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_2, 51},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_0, 52},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_1, 53},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_2, 54},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_0, 55},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_1, 56},
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
     {PerfCounterType::PACK_INSTRN_ISSUED_2, 279}}};
constexpr size_t NUM_INSTRN_COUNTERS = 82;
