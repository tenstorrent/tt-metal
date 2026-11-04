// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7b DSA index core (5d m4_phase_spec I1 / I2, replicated per chip): from the I1 reader phase's fp32 projection
// y = hn [wk | kpool_compress_gate | weights_proj] (columns 0..127 k, 128..255 g, 256..287 w; rows = tokens):
//   kn = LayerNorm_128(k) (weight, bias, eps 1e-6)                 -> ring row p % 4 of the token's slot  [kn | g] bf16
//   pooled = sum_j softmax_j(g_j + ape_j) k_j over ring rows 0..3   -> pooled-key row p / 4 of the slot   bf16
//            (fp32; rows past p % 4 hold the previous pool's tokens: the row is final at p % 4 == 3 and masked by
//            n_closed until then, 5d's contract)
//   wsc = w * 32^-0.5 * 128^-0.5 (both indexer scales folded, relu(c x) = c relu(x))  -> row t of a [32, 32] fp32
//            tile (columns = index heads): the in0 of the owners' head-weighted score matmul
// The index core's BRISC then coordinates the owners' top-512 radix select (kernels/mla/dsa_common.hpp I5).
// CB indices (the index core's own):
#pragma once

#include <cstdint>

namespace mlaix {
constexpr uint32_t CB_Y = 0;  // 9 fp32 tiles: k 0..3, g 4..7, w 8
constexpr uint32_t CB_C = 1;  // 13 fp32 tiles: kn_w 0..3 (row 0), kn_b 4..7 (row 0), ape 8..11 (rows 0..3), rowmask 12
constexpr uint32_t CB_ROW = 2;    // 8 bf16 tiles [kn | g], rows = tokens
constexpr uint32_t CB_W = 3;      // 1 fp32 tile wsc (row t = token t)
constexpr uint32_t CB_RING = 4;   // 8 bf16 tiles: the token's slot ring after the row update
constexpr uint32_t CB_POOL = 5;   // 4 bf16 tiles, row 0 = the pooled key
constexpr uint32_t CB_INFO = 6;   // 1 page (128 B): cur_pos[0..T), T <= 32
constexpr uint32_t CB_SCR = 7;    // NCRISC scratch: 4 pool tiles + cur_pos + 64
constexpr uint32_t CB_T0 = 8;     // fp32 temporaries 8..13
constexpr uint32_t CB_CSCR = 14;  // the coordinator's scratch (512 B): cur_pos [0, 128), message [128, 192), quotas
                                  // [192, 192 + 4 C), the multicast count word at 496 (rev 50)
constexpr uint32_t KT = 4, GT = 4, RING_TILES = 8, POOL_TILES = 4;
}  // namespace mlaix
