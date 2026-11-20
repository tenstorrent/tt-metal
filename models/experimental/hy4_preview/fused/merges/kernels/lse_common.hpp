// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the kernels of the cross-chip log-sum-exp merge (fused/merges/lse_merge.py).
//
// A partial of one user and one head group (8 heads) travels and lands as 17 "tiny" BF16 tiles of [8 heads, 32]
// (512 bytes each): first the max / sum tile (max of the unscaled logits in column 0, sum in column 1, as the
// flash-attention LLKs pack it), then the 16 tiles of O (unnormalised, latent space). With the max / sum tile in
// front, one circular buffer laid over a landing serves sdpa_tail both as its max / sum input and as its O input.

#include <cstdint>

namespace hy4_lse {

constexpr uint32_t TINY_BYTES = 512;                      // one [8, 32] BF16 tile
constexpr uint32_t O_TILES = 16;                          // 512 latent columns
constexpr uint32_t PART_TILES = 1 + O_TILES;              // max / sum, then O
constexpr uint32_t O_BYTES = O_TILES * TINY_BYTES;        // 8192
constexpr uint32_t PART_BYTES = PART_TILES * TINY_BYTES;  // 8704
constexpr uint32_t FACE_BYTES = TINY_BYTES / 2;           // an [8, 16] face
constexpr uint32_t ROW_BYTES = 32;                        // one row of a face

// State tile of a merge core (uint32 words, persistent L1; read by the host).
constexpr uint32_t ST_STATUS = 0;  // bit 0: the column landing timed out, bit 1: the row landing, bits 2, 3: the
                                   // merge kernel's column / final result did not arrive, bit 8: the kernel ended
constexpr uint32_t ST_STAMPS = 4;  // start, local loaded, column landed, column merged, row handed, row landed,
                                   // final ready, written

}  // namespace hy4_lse
