// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 indexer-score + running top-512 phase (tt/ix_phase.py, M-B2 section 2 S2 / S3; host CB model
// tests/s16/ix_phase_model.py). Per chip: R query rows (32 indexer heads x 128 each, query-major), P pooled keys
// (32 pools x 128 = one key tile KT = 4 bf16 tiles), 64 member cores with NQ resident queries each, one streamer core.
//   streamer BRISC   keys from DRAM in KC-key-tile blocks -> multicast into every member's CB_K ring slot b % DEPTH;
//                    flow control on two monotonic counters: READY (on the streamer: +1 per member and block whose slot
//                    is free) and VALID (on the members: set to b + 1 after block b landed)
//   member BRISC     Q (4 NQ tiles) and the per-query head-weight tiles WS (row r of tile r = query r's 32 weights)
//   into
//                    L1 once; then per block: reserve the CB_K slot, +1 READY, wait VALID >= b + 1, push
//   member TRISC     per block and query r: S_r = relu(Q_r @ K^T) (KC tiles, 4 K-steps each) -> CB_SR; then
//                    WS_r @ S_r packed onto the block's KC score tiles with the packer's L1 accumulation (query 0
//                    overwrites) -> CB_SC: tile j row r = query r's scores of the 32 pools of key tile j
//   member NCRISC /  the running top-512 (ixp_filter.hpp): NCRISC consumes CB_SC (rows [0, NQA)), BRISC CB_SC2 (rows
//   BRISC            [NQA, NQ), between its CB_K receives); per query a threshold + a candidate buffer
//   (order-preserving
//                    uint32 key, pool id), an in-place quickselect on overflow; at the end the top 512 pool ids per
//                    query -> the selection rows [R, 512] uint32 (0xFFFFFFFF padding below 512 visible)
// CB_K slot layout (v3): DIM-major, tile (d, j) at d KC + j, so a K-step's KC tiles are contiguous (matmul_block reads
// in1 columns at consecutive tile indices, transposed or not)
// One producer / one consumer per CB. Visibility: query q sees pools p < (CS + q + 1) / 4 (complete pools whose last
// token is visible), CS = the chunk's absolute start (runtime arg).
#pragma once

#include <cstdint>

namespace ixp {
constexpr uint32_t CB_Q = 0, CB_WS = 1, CB_K = 2, CB_SR = 3, CB_SC = 4, CB_CAND = 5, CB_SC2 = 6,
                   CB_CAND2 = 7;  // members
constexpr uint32_t CB_SM = 8,
                   CB_ONE = 9;  // v8: the compute's own copy of the scores (row-max input), the reduce scaler
// v8: a score CB block = KC score tiles then KC row-max tiles (row r's max over the key tile's 32 pools at column 0;
// FPU reduce on tf32-truncated operands -> the filter skips a row-tile only if okey(max) + MAX_MARGIN <= thr)
constexpr uint32_t MAX_MARGIN = 1u << 15;  // ~2^-8 relative: 4x the tf32 truncation of a positive max
constexpr uint32_t CB_ST = 0;              // streamer staging
constexpr uint32_t SEM_READY = 0, SEM_ADDR = 1, SEM_VALID = 2;
constexpr uint32_t TOPK = 512;
constexpr uint32_t T16 = 2048, T32 = 4096;

// element (r, c) of a 32 x 32 tile in face layout (16 x 16 faces, row-major inside a face)
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
// order-preserving uint32 of an fp32 bit pattern (larger float -> larger key)
inline uint32_t okey(uint32_t u) {  // branchless: negative -> ~u, else u | sign
    return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}
}  // namespace ixp
