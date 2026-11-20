// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared layout of the Hy4 decode projection programs (fused/projections/program.py).
//
// The "meta page" is one CB page at the same L1 address on every worker core of the chip. The hub fills it and
// multicasts it before it releases the cores. The release is one multicast increment of the "go" word, a word of a
// persistent L1 tensor at the same address on every core (not a program semaphore: those are rewritten on every core
// at each launch); a unit consumes it by writing it back to zero.
//   0     uint16 [128]            unused (word 0 = 1)
//   256   fp32 scale[ROWS]        the per-row multiplier of a unit's finishing (the deferred input norm's 1 / RMS)
//   288   fp32 lanes[ROWS][8]     lane sums of the squares of q_a (summed over the chips): the query units take
//                                 rsqrt(sum / n + eps) of them
//   1536  uint32 ctr[16]          hand-over counters between stages: counter i = first-stage units of group i that
//                                 have delivered their tiles to the activation buffer (multicast increments). They
//                                 travel as zeros in the hub's multicast, so they are zero on every core before any
//                                 increment can land.
//   1600  64 bytes of core-local scratch

#pragma once

#include <stdint.h>

#ifndef ROWS
#define ROWS 8
#endif
// Spins of a bounded wait (on another core, or on the NOC). A core that times out reports and carries on with what
// it has, so the program always ends.
#ifndef HY4_SPIN_LIMIT
#define HY4_SPIN_LIMIT 10000000u
#endif

namespace hy4proj {

constexpr uint32_t PARAM_SCALE = 256;
#if ROWS == 8
constexpr uint32_t PARAM_LANES = 288;
#else
// (16 token rows: one pass of the weights for two row tiles, proj_compute_hifi.cpp with hifi_mm16) the lane sums
// follow the 16 scales
constexpr uint32_t PARAM_LANES = PARAM_SCALE + 4 * ROWS;
static_assert(ROWS == 16, "a unit's row tile is [8, 32] or [16, 32]");
#endif
constexpr uint32_t LANES = 8;
constexpr uint32_t META_IN_BYTES = 1536;  // what the hub's parameter tensor provides
constexpr uint32_t META_CTR = META_IN_BYTES;
constexpr uint32_t NUM_CTR = 16;
constexpr uint32_t META_BYTES = META_CTR + 4 * NUM_CTR;  // multicast by the hub
constexpr uint32_t META_SCRATCH = META_BYTES;
constexpr uint32_t META_PAGE_BYTES = META_BYTES + 64;

constexpr uint32_t ROW_TILE_BYTES = 64 * ROWS;  // one [ROWS, 32] BF16 tile
constexpr uint32_t ROW_TILE_BYTES_F32 = 128 * ROWS;

// status words (uint32) in the hub's status tensor
constexpr uint32_t ST_RUNS = 0;
constexpr uint32_t ST_UNIT = 1;  // last unit that timed out or could not send: noc x << 24 | y << 16 | code
                                 // (1: no release, 2: its group's tiles did not arrive, 3: could not send)
constexpr uint32_t ST_HUB = 5;   // the hub could not send (1: input, 2: meta page, 3: release, 4: acknowledgements)
constexpr uint32_t ST_HUB_TIMING = 8;  // HY4_TIMING: hub cycles until the release was sent
constexpr uint32_t ST_TIMING = 16;     // HY4_TIMING: 2 words per core (proj_dm.hpp report_timing)
constexpr uint32_t ST_WORDS = 16 + 2 * 160;

// finishing modes of a unit (layout.py FIN_*)
constexpr uint32_t FIN_NONE = 0;        // the matmul's FP32 sums, packed as they are
constexpr uint32_t FIN_SCALE = 1;       // * scale[row]
constexpr uint32_t FIN_SIGMOID = 2;     // sigmoid(* scale[row])
constexpr uint32_t FIN_STAT = 3;        // * scale[row]; one more FP32 tile: the lane sums of the squares
constexpr uint32_t FIN_NORM = 4;        // * scale[row] * rsqrt(sum(lanes[row]) / n + eps)
constexpr uint32_t FIN_NORM_ROPE = 5;   // RoPE on interleaved pairs (2 tiles = 64 dims), then as FIN_NORM
constexpr uint32_t FIN_GATE = 6;        // * the gate tiles (elementwise)
constexpr uint32_t FIN_ROUND = 7;       // rounding only
constexpr uint32_t FIN_SCALE_ROPE = 8;  // RoPE on interleaved pairs (2 tiles), then * scale[row]
constexpr uint32_t FIN_CONST = 9;       // * one constant (the unit's `param`, FP32 bits)

// whether a mode reads the constant page the unit's data-movement kernel builds
constexpr bool fin_has_const(uint32_t fin) {
    return (fin >= FIN_SCALE && fin <= FIN_NORM_ROPE) || fin == FIN_SCALE_ROPE;
}

// rounding of a finished value before the packer cuts it to BF16
constexpr uint32_t RND_NONE = 0;  // the packer truncates
constexpr uint32_t RND_BF16 = 1;  // to nearest BF16
constexpr uint32_t RND_7 = 2;     // to 7 significant bits: what the next LoFi matmul keeps of its left operand

constexpr uint32_t NO_WAIT = 0xFFFF;

}  // namespace hy4proj
