// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 XSL expert FFN in-program probe (data-movement RISCs; GLM53_PF_MOE_XFFN_PROBE=1 compiles it in as
// XFFN_PROBE): per (core, RISC) 16 uint32 counters -- wall-clock cycles spent in each data-movement activity (a NoC
// transfer counts from its first issue to its barrier), in each CB wait, the span, and the start / end clock -- written
// at the end of the kernel as one 64-byte row of a DRAM uint32 tensor [2 P, 16] (reader of core i: row i, writer: row
// P + i). The host decodes them (EpMoe.xffn_probe_read): per phase DRAM-read-active / write-active / wait time vs span.
// Without XFFN_PROBE every macro vanishes.
#pragma once

#include <cstdint>

#if defined(XFFN_PROBE)
#include "api/dataflow/dataflow_api.h"

namespace xffn_probe {
inline uint32_t g_c[16];
inline uint64_t g_t0 = 0;
inline uint64_t g_start = 0;
FORCE_INLINE void begin() {
    for (uint32_t i = 0; i < 16; ++i) {
        g_c[i] = 0;
    }
    g_start = get_timestamp();
}
FORCE_INLINE void tic() { g_t0 = get_timestamp(); }
FORCE_INLINE void toc(uint32_t slot) { g_c[slot] += static_cast<uint32_t>(get_timestamp() - g_t0); }
FORCE_INLINE void add(uint32_t slot, uint32_t v) { g_c[slot] += v; }
// counters 13 / 14 / 15 = span, start (low 32 bits), end (low 32 bits); row `row` of the probe tensor at `addr`
FORCE_INLINE void flush(uint32_t scratch_l1, uint32_t addr, uint32_t row) {
    const uint64_t end = get_timestamp();
    g_c[13] = static_cast<uint32_t>(end - g_start);
    g_c[14] = static_cast<uint32_t>(g_start);
    g_c[15] = static_cast<uint32_t>(end);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = g_c[i];
    }
    const InterleavedAddrGen<true> probe{.bank_base_address = addr, .page_size = 64};
    noc_async_write(scratch_l1, probe.get_noc_addr(row), 64);
    noc_async_write_barrier();
}
}  // namespace xffn_probe
#define XP_BEGIN() xffn_probe::begin()
#define XP_TIC() xffn_probe::tic()
#define XP_TOC(slot) xffn_probe::toc(slot)
#define XP_ADD(slot, v) xffn_probe::add(slot, v)
#define XP_FLUSH(scratch, addr, row) xffn_probe::flush(scratch, addr, row)
#else
#define XP_BEGIN() \
    do {           \
    } while (0)
#define XP_TIC() \
    do {         \
    } while (0)
#define XP_TOC(slot) \
    do {             \
    } while (0)
#define XP_ADD(slot, v) \
    do {                \
    } while (0)
#define XP_FLUSH(scratch, addr, row) \
    do {                             \
    } while (0)
#endif

// counter slots. reader (BRISC, NoC0): weights
namespace xffn_probe_slot {
constexpr uint32_t R_GU_READ = 0;  // gate|up weight chunk: first read issued .. read barrier
constexpr uint32_t R_DN_READ = 1;  // down weight chunk: first read .. barrier
constexpr uint32_t R_GU_WAIT = 2;  // cb_reserve_back(CB_WGU): the compute has not freed a gate|up slot yet
constexpr uint32_t R_DN_WAIT = 3;  // cb_reserve_back(CB_WD)
constexpr uint32_t R_UNITS = 4;    // work units run
constexpr uint32_t R_KB = 5;       // KB of weights read
constexpr uint32_t R_TABLES = 6;   // the control tables' read at the start
constexpr uint32_t R_RELAY = 7;    // PMK_YW_BRISC (pmk): the row-bfp8 re-lay rows done inside the reader's waits
// writer (NCRISC, NoC1): the x gather + the output rows
constexpr uint32_t W_X_READ = 0;     // x rows of one K chunk: first read .. barrier
constexpr uint32_t W_X_WAIT = 1;     // cb_reserve_back(CB_XRM): the compute has not consumed the previous block
constexpr uint32_t W_Y_WAIT = 2;     // cb_wait_front(CB_YRM): waiting for the compute's output rows
constexpr uint32_t W_Y_WRITE = 3;    // output rows: first write .. write barrier
constexpr uint32_t W_CTL_WAIT = 4;   // cb_wait_front(CB_WCTL): waiting for the reader's next unit
constexpr uint32_t W_DISP_READ = 5;  // the group's disp rows
constexpr uint32_t W_X_KB = 6;       // KB of x rows read
constexpr uint32_t W_Y_KB = 7;       // KB of output rows written
}  // namespace xffn_probe_slot
