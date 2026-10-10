// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared constants for the qwen_q_prep kernels.
//
// Two core groups run one program per device:
//   N cores (norm):  q bf8 -> bf16 (typecast) -> RMSNorm (q_norm)      : HiFi2, fp32 DEST
//   R cores (rope):  RoPE (rotary_embedding_llama) -> bf8 (typecast)    : HiFi4, 16-bit DEST
// Each N core hands its normalised bf16 tile rows to one paired R core through CB_NORMED
// (an L1-to-L1 write plus two semaphores), because one compute kernel cannot run the two
// stock compute configurations at once.
//
// A tile row is the 4 tiles (head_dim 128) of 32 consecutive positions of one head.

#pragma once

#include <stdint.h>

namespace qprep {

constexpr uint32_t WT = 4;  // head_dim / 32

// N cores
constexpr uint32_t CB_QIN = 0;     // bf8  q tiles (reader -> compute)
constexpr uint32_t CB_X = 1;       // bf16 typecast output = stock RMSNorm input/xmm buffer
constexpr uint32_t CB_SCALER = 2;  // bf16 reduce scaler (1.0 in row 0 of each face)
constexpr uint32_t CB_EPS = 3;     // bf16 eps in column 0
constexpr uint32_t CB_GAMMA = 4;   // bf16 gamma, row 0 of faces 0 and 1
constexpr uint32_t CB_XMM2 = 5;    // fp32 x^2
constexpr uint32_t CB_EX2 = 6;     // fp32 mean(x^2)
constexpr uint32_t CB_EX2PE = 7;   // fp32 rsqrt(mean(x^2) + eps)
constexpr uint32_t CB_FUSION = 8;  // fp32 x * rstd
constexpr uint32_t CB_NOUT = 16;   // bf16 normed tiles (compute -> handoff writer)

// Shared (declared on every core so it has one L1 address on all of them)
constexpr uint32_t CB_NORMED = 9;  // bf16 normed tiles, filled remotely by the paired N core

// R cores
constexpr uint32_t CB_TRANS = 10;  // bf16 RoPE transformation matrix (1 tile)
constexpr uint32_t CB_COS = 11;    // bf16
constexpr uint32_t CB_SIN = 12;    // bf16
constexpr uint32_t CB_ROT = 13;    // bf16 x @ trans_mat
constexpr uint32_t CB_SIN_I = 14;  // bf16 rotated * sin
constexpr uint32_t CB_COS_I = 15;  // bf16 x * cos
constexpr uint32_t CB_QOUT = 17;   // bf8  final q tiles (compute -> writer)

// Semaphore ids
constexpr uint32_t SEM_CREDIT = 0;  // on N cores: number of CB_NORMED slots the R core has granted
constexpr uint32_t SEM_READY = 1;   // on R cores: number of normed rows the N core has delivered

// Walks this core's tile rows in the global order (local slice j, tile row in slice, head): the head
// index runs fastest, so consecutive rows share one position block and one cos/sin row.
// Local slice 0 is ring slice s0, local slice 1 is ring slice s1 (= 2 * cols - 1 - s0).
struct RowWalker {
    uint32_t h;
    uint32_t j;
    uint32_t tl;
    uint32_t sl_t;     // tile rows per ring slice
    uint32_t s0_t;     // first tile row of slice s0
    uint32_t s1_t;     // first tile row of slice s1
    uint32_t ht;       // tile rows per head (S / 32)
    uint32_t n_heads;  // H

    // Tile row index within the head (also the cos/sin tile row).
    uint32_t t() const { return (j == 0 ? s0_t : s1_t) + tl; }
    // First q page (tile) of the current row in a [1, H, S, 128] TILE tensor.
    uint32_t q_page() const { return (h * ht + t()) * WT; }
    // First page of the current row in the [1, H, sl, 128] TILE tensor of local slice j (split output).
    uint32_t slice_page() const { return (h * sl_t + tl) * WT; }
    // First cos/sin page of the current row in a [1, 1, S_rot, 128] TILE tensor.
    uint32_t rot_page() const { return t() * WT; }

    void next() {
        if (++h == n_heads) {
            h = 0;
            if (++tl == sl_t) {
                tl = 0;
                ++j;
            }
        }
    }
};

}  // namespace qprep
