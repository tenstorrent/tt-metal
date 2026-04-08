// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Wormhole-specific perf counter arrays.
// Included by perf_counters.hpp after PerfCounterType enum is defined.

// WH: all 22 counters active.
// Grant bank mapping differs from BH at banks 0 and 4-6:
//   WH grant[0] (sel 256) = 4 HF cycles  (BH: math not blocked by src — dead)
//   WH grant[4] (sel 260) = srcB not blocked by port
//   WH grant[5] (sel 261) = srcA not blocked by overwrite
//   WH grant[6] (sel 262) = srcA not blocked by port
// Also: FIDELITY_PHASE_STALLS is always 0 (fidelity_phases_ongoing = 1'b0 on WH).
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
     {PerfCounterType::MATH_INSTRN_NOT_BLOCKED_SRC, 256},  // WH: actually 4-HF-cycle counter (hf_cycles==2'b11)
     {PerfCounterType::INSTRN_2_HF_CYCLES, 257},
     {PerfCounterType::INSTRN_1_HF_CYCLE, 258},
     {PerfCounterType::SRCB_WRITE_ACTUAL, 259},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 260},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 261},
     {PerfCounterType::SRCA_WRITE_ACTUAL, 262},
     {PerfCounterType::SRCA_WRITE_THREAD0, 263},
     {PerfCounterType::SRCB_WRITE_THREAD0, 264},
     {PerfCounterType::SRCA_WRITE_THREAD1, 265},
     {PerfCounterType::SRCB_WRITE_THREAD1, 266}}};
constexpr size_t NUM_UNPACK_COUNTERS = 22;

// WH: PACK_COUNT=4, all 16 counter_sels (8 req + 8 grant). Banks 6-7 grants
// are tied to 2'b00 in RTL, so counter_sels 273-274 always read 0.
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

// L1 bank 0 counters (Tensix only — ERISC disabled: MMIO readback hardwired to 0 in RTL)
// Tensix L1 bank 0 (MUX_CTRL[4] = 0): unpacker, TDMA bundles, ring0 NOC
// Port 1: WH has unpacker#1/ECC/pack1
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_0_counters = {
    {{PerfCounterType::L1_0_UNPACKER_0, 0},
     {PerfCounterType::L1_0_UNPACKER_1_ECC_PACK1, 1},
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

// Tensix L1 bank 1 (MUX_CTRL[4] = 1): packer/risc, ext unpacker, ring1 NOC
// Port 8: WH has TDMA packer 2
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 16> l1_1_counters = {
    {{PerfCounterType::L1_1_TDMA_PACKER_2, 0},
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

// WH INSTRN_THREAD: 90 counters
// Sel 27-38: shared stall conditions BROADCAST from thread 0 to all 3 slots (read slot 0 only)
// Sel 39-65: per-thread stall reasons (9 types x 3 threads)
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 90> instrn_counters = {
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
     // Sel 27-38: shared stall conditions (broadcast from thread 0; read slot 0 only)
     {PerfCounterType::WAITING_FOR_SRCA_CLEAR, 27},   // srca_cleared (broadcast)
     {PerfCounterType::WAITING_FOR_SRCB_CLEAR, 30},   // srcb_cleared (broadcast)
     {PerfCounterType::WAITING_FOR_SRCA_VALID, 33},   // srca_valid (broadcast)
     {PerfCounterType::WAITING_FOR_SRCB_VALID, 36},   // srcb_valid (broadcast)
     // Sel 39-65: per-thread stall reasons (9 types x 3 threads)
     {PerfCounterType::WAITING_FOR_THCON_IDLE_0, 39},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_1, 40},
     {PerfCounterType::WAITING_FOR_THCON_IDLE_2, 41},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_0, 42},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_1, 43},
     {PerfCounterType::WAITING_FOR_UNPACK_IDLE_2, 44},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_0, 45},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_1, 46},
     {PerfCounterType::WAITING_FOR_PACK_IDLE_2, 47},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_0, 48},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_1, 49},
     {PerfCounterType::WAITING_FOR_MATH_IDLE_2, 50},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_0, 51},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_1, 52},
     {PerfCounterType::WAITING_FOR_NONZERO_SEM_2, 53},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_0, 54},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_1, 55},
     {PerfCounterType::WAITING_FOR_NONFULL_SEM_2, 56},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_0, 57},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_1, 58},
     {PerfCounterType::WAITING_FOR_MOVE_IDLE_2, 59},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_0, 60},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_1, 61},
     {PerfCounterType::WAITING_FOR_MMIO_IDLE_2, 62},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_0, 63},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_1, 64},
     {PerfCounterType::WAITING_FOR_SFPU_IDLE_2, 65},
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
constexpr size_t NUM_INSTRN_COUNTERS = 90;
