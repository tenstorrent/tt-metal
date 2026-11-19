// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14 pull-form go / no-go microbenchmark (w8/progb_2ms_design.md step 4; coordinator 2026-10-25 22:26:06Z):
// can compute cores PULL weight chunks from bank-adjacent stagers' L1 rings at ~470-490 GB/s aggregate (vs 7a's 493
// GB/s for 24 bank-adjacent direct readers and ~304 GB/s for all-core rotating reads)? One interleaved DRAM tensor of
// PB-byte pages (page p in bank p mod 8). Probe row per core (uint32 x 8): [start lo, end lo, KB moved, wait cycles /
// 16, polls, 0, 0, magic].
#pragma once

#include <cstdint>

namespace pullb {
constexpr uint32_t MAGIC = 0x9017b001;
// stager L1 words (a scratch CB at the same address on every core of the program): PRODUCED at +0 (written locally,
// polled remotely by consumers), CONSUMED[slot] at +16 * (1 + slot) (atomically incremented by the consumers, waited
// locally by the stager: cumulative per slot -- chunk n goes to slot n mod S, so slot k needs n / S consumes before
// chunk n reuses it)
constexpr uint32_t W_PRODUCED = 0;
FORCE_INLINE uint32_t w_consumed(uint32_t slot) { return 16 * (1 + slot); }
constexpr uint32_t SYNC_B = 16 * 17;  // up to 16 slots
// the CB set, allocated on EVERY core of the program so the L1 addresses match across cores (a consumer addresses a
// stager's ring / sync words with its own CB bases)
constexpr uint32_t CB_RING = 0, CB_SYNC = 1, CB_LAND = 2;
}  // namespace pullb
