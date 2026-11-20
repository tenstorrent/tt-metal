// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared layout of the Hy4 decode MoE program (fused/moe/program.py).
//
// The "meta page" is one CB page at the same L1 address on every worker core of the chip. The hub core fills it and
// multicasts it before it releases the cores (hand-over word SEM_GO):
//   0     uint16 U, uint16 ids[U]          the union of the rows' picked experts (local expert ids of this chip)
//   252   uint16 n_plain                   the last n_plain slots are plain SwiGLU (no clamp): the shared expert, the
//                                          dense FFN's slices
//   256   bf16 w[ROWS][MAX_SLOTS]          row r's routing weight of union slot i (0 when the row did not pick it)
//   ..    uint32 ctr_fin[MAX_SLOTS]        per finisher: activation tiles it has delivered (multicast increments)
//   ..    uint32 ctr_part[MAX_UNITS]       per unit (slot * P + pair): gate|up partial sums delivered to its finisher
// The counters travel as zeros in the same multicast, so they are zero on every core before any increment can land.

#pragma once

#include <stdint.h>

#ifndef ROWS
#define ROWS 8
#endif
// Spins of a bounded wait (on another core, or on the NOC). The time of a spin is not measured yet (a load and a
// compare: tens of ns is the estimate, so 10M spins is a few hundred ms against a program of about a ms and an
// operation timeout of 120 s). A core that times out reports and carries on with what it has, so the program
// always ends.
#ifndef HY4_SPIN_LIMIT
#define HY4_SPIN_LIMIT 10000000u
#endif

namespace hy4moe {

// Hand-over words: 16 uint32 per core in a host-zeroed L1 buffer at the same address on every core (HY4_SYNC, a
// define). They are not program semaphores: those are rewritten on every core at every launch
// (tt_metal/impl/program/dispatch.cpp), which costs launch time. The core that waits on a word clears it once it
// has seen it, so every launch starts from zero; after a run that reported a timeout the host clears the buffer
// (program.py reset_sync).
constexpr uint32_t SEM_GO = 0;      // hub -> every core: x and the meta page are in place
constexpr uint32_t SEM_LOCAL = 1;   // BRISC -> NCRISC of the same core: the control words are valid
constexpr uint32_t SEM_X = 2;       // hub -> every core: x is in place (the router cores start on it)
constexpr uint32_t SEM_SCORES = 3;  // router cores -> hub: score tiles delivered
constexpr uint32_t SEM_OUT = 4;     // down cores -> the core that holds the output buffer: output tiles written
// Words 8.. of the same buffer are the core's constants, written by the host (program.py): they are not runtime
// arguments, because arguments that differ from core to core are dispatched core by core at every launch.
constexpr uint32_t CFG_BANK = 8;   // the core's DRAM bank
constexpr uint32_t CFG_INDEX = 9;  // its index among its role's cores of that bank (gate|up, down), or among its role
constexpr uint32_t CFG_IDLE =
    0xFFFF;                          // CFG_INDEX of a core with no work: it only makes its role's core set a rectangle
constexpr uint32_t CFG_VC = 10;      // the NOC0 virtual channel of its DRAM reads
constexpr uint32_t CFG_W_ADDR = 11;  // the address of its weight stream in the bank, >> 5
#ifdef HY4_SYNC
constexpr uint32_t sem(uint32_t i) { return HY4_SYNC + 4 * i; }
inline uint32_t cfg(uint32_t i) { return *reinterpret_cast<volatile uint32_t*>(HY4_SYNC + 4 * i); }
#endif

constexpr uint32_t MAX_SLOTS = 80;   // 64 routed (8 rows x 8 picks) and up to 16 plain slots
constexpr uint32_t MAX_UNITS = 640;  // slots x P
constexpr uint32_t META_IDS = 2;
constexpr uint32_t META_N_PLAIN = 252;
constexpr uint32_t META_W = 256;
constexpr uint32_t META_CTR_ACT = META_W + ROWS * MAX_SLOTS * 2;
constexpr uint32_t META_CTR_PART = META_CTR_ACT + MAX_SLOTS * 4;
constexpr uint32_t META_IN_BYTES = META_CTR_ACT;                // [U, ids | w]: what the routing provides
constexpr uint32_t META_BYTES = META_CTR_PART + MAX_UNITS * 4;  // multicast by the hub
constexpr uint32_t META_SCRATCH = META_BYTES;                   // 64 bytes of core-local scratch behind it
constexpr uint32_t META_PAGE_BYTES = META_BYTES + 64;

constexpr uint32_t ROW_TILE_BYTES = 64 * ROWS;  // one [ROWS, 32] BF16 tile
constexpr uint32_t ROW_TILE_BYTES_F32 = 128 * ROWS;

// status words (uint32 each) in the hub's status tensor: [0] = runs, [role] = last timeout (noc x << 24 | y << 16 |
// code)
constexpr uint32_t ST_RUNS = 0;
constexpr uint32_t ST_GATE_UP = 1;
constexpr uint32_t ST_FINISH = 2;
constexpr uint32_t ST_DOWN = 3;
constexpr uint32_t ST_ROUTER = 4;
constexpr uint32_t ST_HUB = 5;         // the hub could not send (1: x, 2: meta page, 3: release, 4: acknowledgements)
constexpr uint32_t ST_HUB_TIMING = 8;  // HY4_TIMING: hub cycles to x sent, scores in, release sent
constexpr uint32_t ST_TIMING = 16;     // HY4_TIMING: 2 words per core (moe_dm.hpp report_timing)
constexpr uint32_t ST_WORDS = 16 + 2 * 160;

}  // namespace hy4moe
