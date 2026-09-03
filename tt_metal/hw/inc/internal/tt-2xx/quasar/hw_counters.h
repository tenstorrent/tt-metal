// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <utility>

// Quasar counter tables, verified against the A0 tapeout RTL. Included by perf_counters.hpp after
// the PerfCounterType enum; each NEO's math TRISC reaches its own units through the local-regs window.

// The tt-1xx register names in quasar/tensix.h carry Blackhole offsets that are wrong here; alias
// from the generated tensix_neo_reg.h offsets instead.
#define RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0 \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0 \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK0_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_FPU0 (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU0_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0 \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK0_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_INSTRN_THREAD_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_UNPACK_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_PACK_REG_OFFSET)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU \
    (LOCAL_REGS_BASE + NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_FPU_REG_OFFSET)

// l1_client: one clear-on-read CSR behind a subport*8+event mux (subports 0-3 TRISC, 4 THCON,
// 5-24 unpack, 25-36 pack; events 0 unused, 1 SBank pop, 2-6 stall/work/pending carries, 7 order FIFO active).
#define RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CTRL                                  \
    (LOCAL_REGS_BASE + (NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR - \
                        NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR))
#define RISCV_DEBUG_REG_QUASAR_L1_CLIENT_PERF_CNT                                  \
    (LOCAL_REGS_BASE + (NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR - \
                        NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR))
#define QUASAR_L1_CLIENT_PERF_CTRL_ENABLE 0x1u
#define QUASAR_L1_CLIENT_PERF_SUBPORT_SHIFT 4
#define QUASAR_L1_CLIENT_PERF_EVENT_SHIFT 12
#define QUASAR_L1_CLIENT_NUM_SUBPORTS 37
#define QUASAR_L1_CLIENT_NUM_EVENTS 8

// FPU unit: bank 0 grant is tied, bank 1 grant is fpu-or-sfpu valid; same wiring as Blackhole.
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 3> fpu_counters = {
    {{PerfCounterType::FPU_COUNTER, 0}, {PerfCounterType::SFPU_COUNTER, 1}, {PerfCounterType::MATH_COUNTER, 257}}};
constexpr std::size_t NUM_FPU_COUNTERS = 3;

// TDMA_UNPACK unit: 11 banks. Sels 2/256/257 are dead on A0 (fidelity logic hardwired off) and
// sel 258 duplicates sel 3, so they are not captured. Quasar runs 3 unpackers per thread: sel 9 is
// unpacker2/thread0 and sel 10 is unpacker0/thread1 (the Blackhole names do not carry over).
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 18> unpack_counters = {
    {{PerfCounterType::MATH_SRC_DATA_READY, 0},
     {PerfCounterType::DATA_HAZARD_STALLS_MOVD2A, 1},
     {PerfCounterType::MATH_INSTRN_STARTED, 3},
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4},
     {PerfCounterType::SRCB_WRITE_AVAILABLE, 5},
     {PerfCounterType::SRCA_WRITE_AVAILABLE, 6},
     {PerfCounterType::UNPACK0_BUSY_THREAD0, 7},
     {PerfCounterType::UNPACK1_BUSY_THREAD0, 8},
     {PerfCounterType::UNPACK2_BUSY_THREAD0, 9},
     {PerfCounterType::UNPACK0_BUSY_THREAD1, 10},
     {PerfCounterType::SRCB_WRITE_ACTUAL, 259},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 260},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 261},
     {PerfCounterType::SRCA_WRITE_ACTUAL, 262},
     {PerfCounterType::SRCA_WRITE_THREAD0, 263},
     {PerfCounterType::SRCB_WRITE_THREAD0, 264},
     {PerfCounterType::SRCA_WRITE_THREAD1, 265},
     {PerfCounterType::SRCB_WRITE_THREAD1, 266}}};
constexpr std::size_t NUM_UNPACK_COUNTERS = 18;

// TDMA_PACK shares the 21-slice readout with unpack: pack is slices 11-18, 12-14 and 17 tied on A0.
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 5> pack_counters = {
    {{PerfCounterType::PACKER_DEST_READ_AVAILABLE, 11},
     {PerfCounterType::PACKER_BUSY, 18},
     {PerfCounterType::DEST_READ_GRANTED_0, 267},
     {PerfCounterType::MATH_NOT_STALLED_DEST_WR_PORT, 271},
     {PerfCounterType::AVAILABLE_MATH, 272}}};
constexpr std::size_t NUM_PACK_COUNTERS = 5;

// INSTRN readout: sel = class*4+thread (cfg,sync,thcon,xsearch,instissue,math,unpack,pack), 32-35
// per-thread any-stall, 36-50 thread-ORed backend-stage stall conditions (sampled downstream of the
// ibuffer, so they can exceed the any-stall counts); grants (sel >= 256) = the thread's ibuffer dequeues.
constexpr std::array<std::pair<PerfCounterType, uint16_t>, 55> instrn_counters = {
    {{PerfCounterType::CFG_INSTRN_AVAILABLE_0, 0},
     {PerfCounterType::CFG_INSTRN_AVAILABLE_1, 1},
     {PerfCounterType::CFG_INSTRN_AVAILABLE_2, 2},
     {PerfCounterType::CFG_INSTRN_AVAILABLE_3, 3},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_0, 4},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_1, 5},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_2, 6},
     {PerfCounterType::SYNC_INSTRN_AVAILABLE_3, 7},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_0, 8},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_1, 9},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_2, 10},
     {PerfCounterType::THCON_INSTRN_AVAILABLE_3, 11},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_0, 12},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_1, 13},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_2, 14},
     {PerfCounterType::XSEARCH_INSTRN_AVAILABLE_3, 15},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_0, 16},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_1, 17},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_2, 18},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_3, 19},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_0, 20},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_1, 21},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_2, 22},
     {PerfCounterType::FPU_INSTRN_AVAILABLE_3, 23},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_0, 24},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_1, 25},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_2, 26},
     {PerfCounterType::UNPACK_INSTRN_AVAILABLE_3, 27},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_0, 28},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_1, 29},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_2, 30},
     {PerfCounterType::PACK_INSTRN_AVAILABLE_3, 31},
     {PerfCounterType::THREAD_STALLS_0, 32},
     {PerfCounterType::THREAD_STALLS_1, 33},
     {PerfCounterType::THREAD_STALLS_2, 34},
     {PerfCounterType::THREAD_STALLS_3, 35},
     {PerfCounterType::TILE_COUNTER_STALL_PACK, 36},
     {PerfCounterType::TILE_COUNTER_STALL_UNPACK, 37},
     {PerfCounterType::SRCS_STALL_PACK, 38},
     {PerfCounterType::SRCS_STALL_SFPU, 39},
     {PerfCounterType::SRCS_STALL_UNPACK, 40},
     {PerfCounterType::DEST_STALL_PACK, 41},
     {PerfCounterType::DEST_STALL_SFPU, 42},
     {PerfCounterType::DEST_STALL_MATH, 43},
     {PerfCounterType::DEST_STALL_UNPACK, 44},
     {PerfCounterType::SFPU_DATA_HAZARD_STALL, 45},
     {PerfCounterType::FPU_DATA_HAZARD_STALL, 46},
     {PerfCounterType::SRCB_STALL_UNPACK, 47},
     {PerfCounterType::SRCA_STALL_UNPACK, 48},
     {PerfCounterType::DVALID_STALL_MATH, 49},
     {PerfCounterType::SRCA_STALL_MATH, 50},
     {PerfCounterType::THREAD_INSTRUCTIONS_0, 256},
     {PerfCounterType::THREAD_INSTRUCTIONS_1, 257},
     {PerfCounterType::THREAD_INSTRUCTIONS_2, 258},
     {PerfCounterType::THREAD_INSTRUCTIONS_3, 259}}};
constexpr std::size_t NUM_INSTRN_COUNTERS = 55;
