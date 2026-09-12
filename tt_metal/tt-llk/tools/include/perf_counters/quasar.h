// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/types.h"

// Quasar (A0) select tables, verified against the tapeout RTL. Quasar has no L1 counter bank; the L1 slot
// is taken by the l1_client CSR (see l1_client_selection_is_valid). Include through perf_counters/inventory.h.

// Metal's Quasar DM firmware folds .rodata into a 2 KB data region, so an includer that defines
// LLK_PERF_TABLES_IN_TEXT gets the tables in the text region instead.
// The tables are not `inline`: LTO drops the section attribute of COMDAT variables, and metal's DM firmware
// needs them in .text (see LLK_PERF_TABLES_IN_TEXT below).
#ifndef LLK_PERF_TABLE_SECTION
#if defined(LLK_PERF_TABLES_IN_TEXT)
#define LLK_PERF_TABLE_SECTION __attribute__((section(".text.perf_counter_tables")))
#else
#define LLK_PERF_TABLE_SECTION
#endif
#endif

namespace llk::perf
{

// clang-format off
// FPU unit: bank 0 grant is tied, bank 1 grant is fpu-or-sfpu valid; same wiring as Blackhole.
constexpr std::array<Entry, 3> fpu_counters LLK_PERF_TABLE_SECTION = {
    {{PerfCounterType::FPU_COUNTER, 0},
     {PerfCounterType::SFPU_COUNTER, 1},
     {PerfCounterType::MATH_COUNTER, 257}}};
inline constexpr std::size_t NUM_FPU_COUNTERS = 3;

// TDMA_UNPACK: sels 2/256/257 are dead on A0 (fidelity off) and 258 duplicates 3. Three unpackers per thread:
// sel 9 is unpacker2/thread0 and sel 10 unpacker0/thread1; both and the odd-TID writes read 0 on every op swept so far.
constexpr std::array<Entry, 18> unpack_counters LLK_PERF_TABLE_SECTION = {
    {{PerfCounterType::MATH_SRC_DATA_READY, 0},
     {PerfCounterType::MATH_NOT_D2S_STALLED, 1},
     {PerfCounterType::MATH_INSTRN_STARTED, 3},
     {PerfCounterType::MATH_INSTRN_AVAILABLE, 4},
     {PerfCounterType::SRCB_WRITE_REQ, 5},
     {PerfCounterType::SRCA_WRITE_REQ, 6},
     {PerfCounterType::UNPACK0_BUSY_THREAD0, 7},
     {PerfCounterType::UNPACK1_BUSY_THREAD0, 8},
     {PerfCounterType::UNPACK2_BUSY_THREAD0, 9},
     {PerfCounterType::UNPACK0_BUSY_THREAD1, 10},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_OVR, 259},
     {PerfCounterType::SRCB_WRITE_NOT_BLOCKED_PORT, 260},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_OVR, 261},
     {PerfCounterType::SRCA_WRITE_NOT_BLOCKED_PORT, 262},
     {PerfCounterType::SRCA_WRITE_TID_EVEN, 263},
     {PerfCounterType::SRCB_WRITE_TID_EVEN, 264},
     {PerfCounterType::SRCA_WRITE_TID_ODD, 265},
     {PerfCounterType::SRCB_WRITE_TID_ODD, 266}}};
inline constexpr std::size_t NUM_UNPACK_COUNTERS = 18;

// TDMA_PACK shares the 21-slice readout with unpack: pack is slices 11-18; request slices 12-17 are tied to 0 on A0
// and the live pack-side grants are slices 11, 15 and 16.
constexpr std::array<Entry, 5> pack_counters LLK_PERF_TABLE_SECTION = {
    {{PerfCounterType::PACKER0_DEST_READ_REQ, 11},
     {PerfCounterType::PACKER_BUSY, 18},
     {PerfCounterType::DEST_READ_GRANTED_0, 267},
     {PerfCounterType::MATH_NOT_STALLED_DEST_WR_PORT, 271},
     {PerfCounterType::MATH_NOT_SCOREBOARD_STALLED, 272}}};
inline constexpr std::size_t NUM_PACK_COUNTERS = 5;

// INSTRN: sel = class*4+thread (cfg,sync,thcon,xsearch,instissue,math,unpack,pack), 32-35 any-stall per thread, 36-50
// thread-ORed backend stalls; grants (>= 256) are ibuffer dequeues. Xsearch is tied to 0 (its grants alias
// THREAD_INSTRUCTIONS); thread 3, THCON, CFG_1, UNPACK_1/2, PACK_0/1 and SRCS_STALL_* read 0 on every op swept so far.
constexpr std::array<Entry, 51> instrn_counters LLK_PERF_TABLE_SECTION = {
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
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_0, 16},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_1, 17},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_2, 18},
     {PerfCounterType::INSTISSUE_INSTRN_AVAILABLE_3, 19},
     {PerfCounterType::MATH_INSTRN_AVAILABLE_0, 20},
     {PerfCounterType::MATH_INSTRN_AVAILABLE_1, 21},
     {PerfCounterType::MATH_INSTRN_AVAILABLE_2, 22},
     {PerfCounterType::MATH_INSTRN_AVAILABLE_3, 23},
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
inline constexpr std::size_t NUM_INSTRN_COUNTERS = 51;
// clang-format on

// No L1 counter bank and no L1 mux on Quasar.
inline constexpr std::uint32_t L1_MUX_MASK   = 0;
inline constexpr std::uint8_t L1_MUX_POSITIONS = 0;

// l1_client CSR selection = subport*8 + event (subports 0-3 TRISC, 4 THCON, 5-24 unpack, 25-36 pack; events 0
// unused, 1 SBank pop, 2-6 stall/work/pending carries, 7 order FIFO active). Event 0 is tied to 0 in the RTL and
// THCON events 1-3 alias the TRISC port's SBank 0 counters (selections 1-3), so both are rejected.
constexpr bool l1_client_selection_is_valid(std::uint32_t sel)
{
    const std::uint32_t subport = sel / QUASAR_L1_CLIENT_NUM_EVENTS;
    const std::uint32_t event   = sel % QUASAR_L1_CLIENT_NUM_EVENTS;
    return sel < QUASAR_L1_CLIENT_NUM_SELECTIONS && event != 0 && !(subport == 4 && event <= 3);
}

} // namespace llk::perf
