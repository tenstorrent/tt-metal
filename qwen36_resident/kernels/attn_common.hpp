// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident Qwen3.6 full-attention (gated attention) decode: shared schedule of the attention cores.
// The streamers (GDN-layer streamer kernels without conv columns) send this chip's projection row
// [q heads | gates | k | v] to the leader and to the tail core, and receive the gated attention
// output o back. Per layer:
//   leader: q = rmsnorm_rows(q) * (1 + w_q) / sqrt(head_dim), rope on the first rot_tiles tiles of every
//           head, multicast to the workers; gs = sigmoid(gate);
//   tail core (a worker): k = rmsnorm_rows(k) * (1 + w_k), rope; k and v go into row `r` of the tail
//           tile (the KV tile holding this position, positions tp * 32 .. tp * 32 + 31), read from the
//           set's DRAM cache and written back as bf8 (bf8 shares exponents within a row, so the other
//           rows round-trip exactly); its chunk is the tail tile (causal mask on the columns past r);
//   workers: each owns a chunk of the complete KV tiles in DRAM, derived from the position, streamed in
//           blocks of chunk_max tiles (the first block prefetched before q arrives); a worker without one
//           sends a zero partial and a -inf row max;
// The KV cache of a layer copy is DRAM height-sharded over the banks: position tile t (Dt bf8 tiles,
// one kv_row of bytes) lives in bank t % banks at row t / banks of that bank's shard, copies at a fixed
// per-bank stride. Chunk worker w (of workers - 1) takes bank w % L (L = banks holding complete
// tiles) and, as worker j of the J on that bank, rows j, j + J, ..., so each bank sees one stream.
// The position, and the rope tables of it, come from the token state (see streamer_common.hpp).
//   every worker: online softmax over its blocks (s = q k^T, running row max m, p = exp(s - m), o and
//           l = rowsum(p) rescaled by exp(m_old - m_new) as m grows), sends m to the leader, receives the
//           global row max M and scales o and l by exp(m - M) into its partial (the tail core has one
//           tile: p = exp(s - M) directly); the partials are summed up a tree (a
//           group head adds its children's partials to its own and sends the sum on; the group heads'
//           sums go to the leader), sending only the top two faces of each tile (head rows < 16);
//   leader: o = sum of the partials / row sum * gs, written into every streamer.
// All attention tiles are 32x32 with head h in row h (heads <= 32 rows).
//
// Compile-time args (all attention-core kernels): 0 attention layers, 1 copies, 2 workers, 3 num_streamers,
// 4 kv_block (KV tiles per streamed block of a worker's chunk, CB sizing), 5 heads, 6 Dt (head_dim tiles),
// 7 rot_tiles (rotated head_dim tiles), 8 sem_rows, 9 sem_q, 10 sem_m, 11 sem_M, 12 sem_part, 13 sem_heads
// (the streamers' head-output semaphore), 14 eps bits, 15 sem_addr (a worker learns the leader's partial
// buffer address through it), 16 fanin (partial slots per node of the reduction tree), 17 banks,
// 18 sem_tail (tail core, local: tail write-backs done; a copy's tail is re-read within a step only after
// its previous write-back), 19 batch, 20 verify (0 / 1)
//
// Verify: the two rows are one user at positions p and p + 1 and share its KV cache (user 0's). Row 1's
// tail tile holds row 0's new row: on the same tile the tail core reads it after row 0's write-back; when
// p is the last row of its tile, that tile is complete for row 1 and the chunk worker reading it waits for
// the write-back (the tail writer bumps the worker's sem_tail, unused on chunk workers otherwise).
//
// Batch: the users run one after another through every attention layer (iteration i = layer * batch +
// user; the semaphores count iterations). User u has its own position (token state word 1 + u), rope
// tables and KV cache (per bank: users, then copies, then rows); the streamers' rows and the leader's
// output are batch x 32 tiles, row u for user u.
#pragma once

#include <stdint.h>

namespace resident_attn {

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t copies = get_compile_time_arg_val(1);
constexpr uint32_t workers = get_compile_time_arg_val(2);
constexpr uint32_t num_streamers = get_compile_time_arg_val(3);
constexpr uint32_t chunk_max = get_compile_time_arg_val(4);  // KV tiles per streamed block
constexpr uint32_t heads = get_compile_time_arg_val(5);
constexpr uint32_t Dt = get_compile_time_arg_val(6);
constexpr uint32_t rot_tiles = get_compile_time_arg_val(7);
constexpr uint32_t sem_rows = get_compile_time_arg_val(8);
constexpr uint32_t sem_q = get_compile_time_arg_val(9);
constexpr uint32_t sem_m = get_compile_time_arg_val(10);
constexpr uint32_t sem_M = get_compile_time_arg_val(11);
constexpr uint32_t sem_part = get_compile_time_arg_val(12);
constexpr uint32_t sem_heads = get_compile_time_arg_val(13);
constexpr uint32_t eps_bits = get_compile_time_arg_val(14);
constexpr uint32_t sem_addr = get_compile_time_arg_val(15);
constexpr uint32_t fanin = get_compile_time_arg_val(16);
constexpr uint32_t banks = get_compile_time_arg_val(17);
constexpr uint32_t sem_tail = get_compile_time_arg_val(18);
constexpr uint32_t batch = get_compile_time_arg_val(19);
constexpr bool verify = get_compile_time_arg_val(20) != 0;
static_assert(!verify || batch == 2, "verify runs two rows of one user");
constexpr uint32_t iters = layers * batch;
static_assert(heads <= 16, "head rows live in the top faces");
static_assert(rot_tiles == 2, "rope pairs dims (i, i + 32) of the first 64 head dims");

constexpr uint32_t kPart = Dt + 1;         // a partial: Dt output tiles, then the row-sum tile
constexpr uint32_t kMSlot = 256;           // bytes per worker row-max slot (rows 0..heads-1 of column 0)
constexpr uint32_t kTile = 2048;           // bf16 32x32 tile
constexpr uint32_t kKvTile = 1088;         // bf8 32x32 tile
constexpr uint32_t kFaceRow = 32;          // bytes of one 16-element face row (bf16)
constexpr uint32_t kFace = 512;            // bytes of one bf16 face
constexpr uint32_t kRowTile = 64 * batch;  // a batch x 32 bf16 tile (streamer rows, leader output)
constexpr uint32_t kKvRow = Dt * kKvTile;  // one position tile of k (or v) in the cache

// shared CBs
constexpr uint32_t cb_q = 0;  // worker: q (Dt tiles, multicast target); leader: its own copy of q
constexpr uint32_t cb_k = 1;  // worker: KV blocks of its chunk, 2 x chunk_max x Dt bf8 tiles
constexpr uint32_t cb_v = 2;
constexpr uint32_t cb_M = 3;         // global row max (1 tile, multicast target)
constexpr uint32_t cb_one = 4;       // reduce scaler 1.0
constexpr uint32_t cb_s = 5;         // scores (fp32) of one block, chunk_max tiles
constexpr uint32_t cb_p = 6;         // exp(s - M) of one block, chunk_max tiles
constexpr uint32_t cb_m = 7;         // this core's row max
constexpr uint32_t cb_o = 8;         // this core's partial (kPart tiles)
constexpr uint32_t cb_ntiles = 10;   // worker: its chunk's KV tiles per user (uint32s, for compute and writer)
constexpr uint32_t kNegInfOff = 64;  // in the cb_ntiles page: the writer's -inf row-max rows (heads x 32 B)
static_assert(batch * 4 <= kNegInfOff, "chunk counts overlap the -inf rows");
// chunk worker: online-softmax state over the chunk's blocks (fp32): output, row sum, row max, rescale
constexpr uint32_t cb_orun = 9;    // 2 x Dt
constexpr uint32_t cb_lrun = 15;   // 2
constexpr uint32_t cb_mrun = 24;   // 2
constexpr uint32_t cb_alpha = 26;  // 1
// leader / tail-core CBs
constexpr uint32_t cb_q_mc = 9;   // q for the multicast
constexpr uint32_t cb_qraw = 10;  // q rows as placed from the streamers' row
constexpr uint32_t cb_kraw = 11;
constexpr uint32_t cb_gate = 24;    // leader
constexpr uint32_t cb_sq = 13;      // scratch (Dt)
constexpr uint32_t cb_rs = 14;      // 1 / rms per row
constexpr uint32_t cb_qw = 15;      // this layer's (1 + w_q) / sqrt(head_dim), Dt tiles, rows replicated
constexpr uint32_t cb_kw = 16;      // this layer's (1 + w_k)
constexpr uint32_t cb_rope = 17;    // per user: cos, sin, -sin of its position (rows replicated)
constexpr uint32_t cb_knew = 18;    // normalized + rotated k row (row 0)
constexpr uint32_t cb_tail8 = 31;   // tail core: tail k | v tiles as read from DRAM (2 Dt bf8)
constexpr uint32_t cb_rowsel = 12;  // tail core: per user R (ones in row r), 1 - R
constexpr uint32_t cb_tailk = 19;   // tail tiles with the new row (bf16, Dt)
constexpr uint32_t cb_tailv = 20;
constexpr uint32_t cb_vraw = 21;   // v row (row 0)
constexpr uint32_t cb_mask = 22;   // per user 0 / -inf over the tail columns
constexpr uint32_t cb_parts = 23;  // fanin x kPart partials of this node's children (written by them)
constexpr uint32_t cb_gs = 26;     // leader: sigmoid(gate)
constexpr uint32_t cb_osum = 25;   // own + children's partials (kPart)
constexpr uint32_t cb_rl = 30;     // leader: 1 / row sum
constexpr uint32_t cb_out = 27;    // gated output (Dt)
constexpr uint32_t cb_flush = 28;  // updated tail k | v as bf8 (2 Dt), written back to DRAM
constexpr uint32_t cb_mean = 29;   // reduce scaler 1 / (32 Dt)
constexpr uint32_t cb_stage = 2;   // leader: gated output rows as batch x 32 tiles (heads x Dt)

// byte offset of row r inside a 32x32 bf16 tile, face 0 (columns 0..15); face 1 is + kFace
FORCE_INLINE uint32_t row_offset(uint32_t r) { return (r < 16 ? 0 : 2 * kFace) + (r & 15) * kFaceRow; }

// byte offset of user u's face 0 row inside a batch x 32 tile; its face 1 row is + batch * kFaceRow
FORCE_INLINE uint32_t user_offset(uint32_t u) { return u * kFaceRow; }

// chunk of chunk worker w at KV tile rows tp (complete position tiles): bank, first row j, row step J,
// number of rows (0: no chunk)
struct Chunk {
    uint32_t bank, j, J, n;
};
FORCE_INLINE Chunk chunk_of(uint32_t w, uint32_t tp) {
    const uint32_t live = tp < banks ? tp : banks;
    if (live == 0) {
        return {0, 0, 1, 0};
    }
    const uint32_t b = w % live;
    const uint32_t j = w / live;
    const uint32_t J = (workers - 1 - b + live - 1) / live;
    const uint32_t rows = (tp - b + banks - 1) / banks;  // tiles b, b + banks, ... below tp
    return {b, j, J, rows > j ? (rows - j + J - 1) / J : 0};
}

}  // namespace resident_attn
