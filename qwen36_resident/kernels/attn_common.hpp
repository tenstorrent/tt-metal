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
// its previous write-back), 19 batch, 20 verify (0 / 1), 21 round_robin (0 / 1: chunk worker w takes the
// complete tiles w, w + workers - 1, ... from every bank instead of its own bank's rows, which balances the
// chunks when workers - 1 is not a multiple of the banks; the partials are summed in another order),
// 22 two_nocs (0 / 1: a chunk worker's reader alternates its KV reads between both NOCs; its writer issues no
// reads, so the reader owns the read counters of the writer's NOC too), 23 group_rows (verify: rows per pass, 1
// = one row per pass, see below)
//
// Verify: the R = batch rows are one user at positions p .. p + R - 1 and share its KV cache (user 0's). Row r's
// tail tile holds the new rows of the rows before it: on the same tile the tail core reads it after row r - 1's
// write-back; once a row's position starts the next tile, row 0's tile is complete for it and the rows after it,
// and the chunk worker reading it waits for the write-back of its last row (the tail writer bumps the worker's
// sem_tail, unused on chunk workers otherwise).
//
// Packed verify (group_rows P > 1): the R rows run as R / P passes ("users") of P rows each. A pass's q tile holds
// row (r, h) = r * heads + h for row r of the group and head h (P * heads <= 32 rows, all four faces when more than
// 16), with per-row rope. The chunk workers take the tiles below row 0's tile for every pass, so the KV cache is
// read once per pass instead of once per row. The tail core takes row 0's tile and the next one if a row reaches it
// (n_tail tiles): once per layer it inserts every row's new k / v (a 0/1 placement matmul per tile, then the
// tiles stay in L1 for the passes), and per pass it scores the tail tiles under per-row causal masks; it writes
// the tail tiles back once per layer.
//
// Batch: the users run one after another through every attention layer (iteration i = layer * batch +
// user; the semaphores count iterations). User u has its own position (token state word 1 + u), rope
// tables and KV cache (per bank: users, then copies, then rows); the streamers' rows and the leader's
// output are batch x 32 tiles, row u for user u.
#pragma once

#include <stdint.h>

#include "active_rows.hpp"  // groups (row groups), before the constants that depend on it

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
constexpr bool round_robin = get_compile_time_arg_val(21) != 0;
constexpr bool two_nocs = get_compile_time_arg_val(22) != 0;
static_assert(!verify || batch >= 2, "verify runs two or more rows of one user");
constexpr uint32_t group_rows = get_compile_time_arg_val(23);
constexpr bool packed = group_rows > 1;
static_assert(!packed || (verify && batch % group_rows == 0), "packed passes take groups of the verify rows");
constexpr uint32_t users = batch / group_rows;  // passes per layer: users, or groups of verify rows
// row groups (active_rows.hpp, rows mode): every group's users run through the layer (global row g * batch + u; the
// streamers' rows / the leader's output are one block of row_tiles / o_tiles batch x 32 tiles per group)
// verify rows (DB1, verify with row groups): group g is user g's verify of batch rows, run as its `users` packed passes
// (pass ug = g * users + p: rows p * group_rows .. of user g); per pass its user's KV, tail tiles and masks
constexpr bool kVRows = verify && groups > 1;
static_assert(
    groups == 1 || (!verify && group_rows == 1) || (kVRows && packed),
    "row groups are rows mode: one row per user (verify rows: packed passes)");
constexpr uint32_t kUsers = users * groups;
// per-row arrays / lists (verify: the batch rows of one user; verify rows: every row of every group)
constexpr uint32_t kRows = kVRows ? batch * groups : (groups == 1 ? batch : kUsers);
// verify rows: rows per user kVR (= batch at 4 x 8, 4 at 8 x 4: two users per group), packed passes per user kPPU, the
// users; pass ug belongs to user ug / kPPU, whose rows are k kVR .. (a single-user verify: kVR = batch, one user)
constexpr uint32_t kVR = kVRows ? vrows_r : batch;
static_assert(!kVRows || (kVR % group_rows == 0 && batch % kVR == 0), "verify rows: whole passes per user");
constexpr uint32_t kPPU = kVR / group_rows;
constexpr uint32_t kVUsers = kVRows ? kRows / kVR : 1;
// the active mask the attention kernels iterate over: rows mode's bit per row, or (verify rows) bit ug per pass, set
// when its group's row 0 is active (a user's rows run together)
// verify rows: a user's tail takes 1 or 2 tiles, users of one launch differ; the tail CBs (tail8, tailk / tailv, flush)
// then move a fixed 2 tiles' pages per user so a reservation never wraps around a CB's end mid-way (its pages are
// addressed as one contiguous block); a single user keeps one size per launch
FORCE_INLINE uint32_t tail_pages(uint32_t n_tail) { return kVRows ? 2 : n_tail; }
FORCE_INLINE uint32_t pass_mask(uint32_t row_mask) {
    if constexpr (!kVRows) {
        return row_mask;
    }
    uint32_t m = 0;
    for (uint32_t ug = 0; ug < kUsers; ug++) {
        m |= ((row_mask >> ((ug / kPPU) * kVR)) & 1u) << ug;
    }
    return m;
}
constexpr uint32_t qrows = heads * group_rows;  // q / output rows of a pass
static_assert(qrows <= 32, "a pass's rows fit one tile");
constexpr bool top_faces = qrows <= 16;  // a pass's rows live in the top faces only
constexpr uint32_t iters = layers * users;
static_assert(rot_tiles == 2, "rope pairs dims (i, i + 32) of the first 64 head dims");

constexpr uint32_t kPart = Dt + 1;         // a partial: Dt output tiles, then the row-sum tile
// bytes per worker row-max slot: the m tile's face 0 rows (0 .. 15), then (more than 16 rows) its face 2 rows at 512
constexpr uint32_t kMSlot = top_faces ? 256 : 1024;
constexpr uint32_t kMRows0 = (qrows < 16 ? qrows : 16) * 32, kMRows2 = top_faces ? 0 : (qrows - 16) * 32;
constexpr uint32_t kTile = 2048;           // bf16 32x32 tile
// the K cache's tile: bf8 (1088 B), or bfp4 (576 B) with S0-8's RESIDENT_KV_K_BFP4 (named arg kv_k_tile)
constexpr uint32_t kKvTile = get_named_compile_time_arg_val("kv_k_tile");
// S0-8 (RESIDENT_KV_K_BFP4): q and k rotated by HAD after RoPE (norm_rope), HAD's 8 x 8 tiles resident on the leader
// and the tail core(s)
constexpr bool kKRot = get_named_compile_time_arg_val("kv_k_rot") != 0;
constexpr bool kKRtn = get_named_compile_time_arg_val("kv_k_rtn") != 0;  // (the decode-written K rows rounded)
// the V cache's tile: bf8 (1088 B), or bfp4 (576 B) with S0's RESIDENT_KV_V_BFP4 (named arg kv_v_tile)
constexpr uint32_t kVTile = get_named_compile_time_arg_val("kv_v_tile");
constexpr uint32_t kFaceRow = 32;          // bytes of one 16-element face row (bf16)
constexpr uint32_t kFace = 512;            // bytes of one bf16 face
constexpr uint32_t kRowTile = 64 * batch;  // a batch x 32 bf16 tile (streamer rows, leader output)
constexpr uint32_t kKvRow = Dt * kKvTile;  // one position tile of k in the cache
constexpr uint32_t kVRow = Dt * kVTile;    // one position tile of v in the cache

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
// in the cb_ntiles page: the chunk KV tiles per user (words 0 .. kUsers - 1), the active mask (word kUsers), and at
// kNegInfOff the writer's -inf row-max rows (a row-max slot); row groups need room for up to 32 users' words
constexpr uint32_t kNegInfOff = groups == 1 ? 64 : 256;
constexpr uint32_t kMaskWord = groups == 1 ? batch : kUsers;
static_assert(batch * 4 <= kNegInfOff && (kMaskWord + 1) * 4 <= kNegInfOff, "chunk counts overlap the -inf rows");
// packed: the bytes of a partial's tiles a worker sends (the top two faces, or all four)
constexpr uint32_t kPartBytes = top_faces ? 2 * 512 : 4 * 512;
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
// cb_rope pages: per token-state row cos, sin, -sin (rows replicated); packed: then per pass the q rope (row (r, h)
// from the group's row r) and the tail's k rope (row r from row r)
// (verify rows: after every row's pages, per pass its q rope, then per user its tail's k rope)
constexpr uint32_t kRopeQ = 3 * batch * groups, kRopeK = kRopeQ + 3 * kUsers;
constexpr uint32_t kRopePages = packed ? kRopeK + 3 * kVUsers : 3 * batch * groups;
// row groups (rows mode): the tail core normalizes and ropes every active user's new k row of a layer at once (user u
// in row u of one tile set, per-row rope tiles at kRopeRows of cb_rope, built by the tail reader), then per user a 0/1
// placement matmul puts its row into the tail tile (cb_rowsel per user: P_u = 1 at (r_u, u), then keep = rows != r_u)
constexpr bool kTailBatched = rows_mode && groups > 1 && !verify;
static_assert(!kTailBatched || kUsers <= 32, "a user per tile row");
// row groups: two tail cores take the active users alternately (named arg attn_tails): role 2 (the last worker) the
// even active indices, role 3 (the worker before it) the odd ones; a tail core runs the other tail's users as a chunk
// worker without a chunk. The chunk workers are the other workers - kTails.
constexpr uint32_t kTails = get_named_compile_time_arg_val("attn_tails");
static_assert(kTails == 1 || (kTails == 2 && kTailBatched), "two tails: the batched tail (row groups)");
constexpr uint32_t kChunkWorkers = workers - kTails;
// whether tail core `role` (2 / 3) takes the user at active index ka
FORCE_INLINE bool tail_takes(uint32_t role, uint32_t ka) { return kTails == 1 || (ka & 1) == role - 2; }
// rows mode: a group head adds only the children with KV tiles for the user (the others' partials are exact zeros; at
// short contexts most chunk workers have none, and the head's sum gates its next row max). The reader leaves a mask per
// user (bit c: child c has tiles) at byte kChildMaskOff of the cb_ntiles page.
constexpr bool kSkipZeroParts = rows_mode && !verify && !packed;
constexpr uint32_t kChildMaskOff = groups == 1 ? 64 + 512 : 640;  // (past the -inf rows / the reader's positions)
constexpr uint32_t kRopeRows = 3 * batch * groups;
constexpr uint32_t kRopeTail = kRopePages + (kTailBatched ? 3 : 0);  // the tail core's cb_rope pages
// row groups (rows mode): the leader normalizes and ropes q for a pass of kQPerPass active users at once (user j of
// the pass in rows j * heads + h, per-row rope tiles at kRopeQPass + 3 p of cb_rope, built by the leader reader); the
// writer extracts each user's heads rows into cb_qstage (rows 0 .. heads - 1) and multicasts that
constexpr bool kLeadBatched = rows_mode && groups > 1 && !verify;
constexpr uint32_t kQPerPass = 32 / heads;
// O-2b: a pass's q prep (norm, rope, S0-8 rotation, the gates' sigmoid) all at once cost the per-user loop a ~30 us
// stall every kQPerPass users; the leader instead runs pass p + 1's prep as four stages (norm + rope, the HAD tile
// matmuls, the butterflies, the sigmoid) before pass p's first four users' outputs, and the reader places pass p + 2's
// rows a user per iteration of pass p (a pass's ~480 row reads at once held up that iteration's partials). (Passes of
// fewer than four users keep the one-shot prep.)
constexpr bool kLeadStaged = kLeadBatched && kQPerPass >= 4 && get_named_compile_time_arg_val("lead_staged") != 0;
constexpr uint32_t kQPasses = (kUsers + kQPerPass - 1) / kQPerPass;
constexpr uint32_t kRopeQPass = 3 * batch * groups;
constexpr uint32_t kRopeLead = kRopePages + (kLeadBatched ? 3 * kQPasses : 0);  // the leader's cb_rope pages
constexpr uint32_t cb_qstage = 41;  // leader writer: one user's q rows for the multicast (kLeadBatched)
// row groups: sigmoid(gate) per pass too — the reader places a pass's gate rows like its q rows, compute takes the
// sigmoid once per pass into cb_gs_pass, the reader extracts user j's rows into rows 0 .. heads - 1 of cb_gs
constexpr bool kGatePass = kLeadBatched;
constexpr uint32_t cb_gs_pass = 42;
// row groups: the leader writer sends user k's M (and q of k + 1) before it stages k - 1's output; the leader's partial
// slots (cb_parts) hold two iterations, the group heads write iteration i's at half i & 1. M(k + 1) goes out only
// after out(k - 1) is staged, i.e. after the leader compute consumed partials(k - 1) — the half k + 1 reuses.
constexpr bool kLeadPipelined = kLeadBatched;
// O-2 (named arg attn_pipe; row groups' batched single tail): a layer's users overlap. Between sending m(i) and
// finishing user i with M(i), the chunk workers compute user i + 1 (scores, online softmax, PV) and the tail core its
// placement, scores and row max, so m(i + 1) reaches the leader while M(i) is on its way. On the workers q and M take
// two slots (user i in slot i & 1; the leader sends q two users ahead), the leader's m slots and every parent's partial
// slots two sets by parity, each with its own semaphore (sem_m / sem_m1, sem_part / sem_part1, counting the
// iterations of their parity). The chunk workers overlap only a user of one KV block (its state then queues behind
// the previous user's in the same CBs; a longer user starts after the previous one finished). Never across layers: a
// layer's first user waits for its rows, which need the previous layer's output.
constexpr bool kAttnPipe = get_named_compile_time_arg_val("attn_pipe") != 0;
static_assert(!kAttnPipe || (kTailBatched && kLeadPipelined && kTails == 1), "O-2: the batched single tail");
constexpr uint32_t sem_m1 = get_named_compile_time_arg_val("sem_m1");
constexpr uint32_t sem_part1 = get_named_compile_time_arg_val("sem_part1");
// (O-2: iterations of i's parity up to and including i — the per-parity semaphores' target)
FORCE_INLINE uint32_t parity_count(uint32_t i) { return i / 2 + 1; }
constexpr uint32_t cb_knew = 18;    // normalized + rotated k row (row 0)
constexpr uint32_t cb_tail8 = 31;   // tail core: tail k tiles as read from DRAM (Dt bf8 per tail tile)
constexpr uint32_t cb_tail8v = 32;  // tail core: tail v tiles as read from DRAM (Dt per tail tile, V's format)
constexpr uint32_t cb_rowsel = 12;  // tail core: per user R (ones in row r), 1 - R (packed: per tail tile the
                                    // placement matrix of the new rows and the rows it keeps)
constexpr uint32_t cb_tailk = 19;   // tail tiles with the new row (bf16, Dt)
constexpr uint32_t cb_tailv = 20;
constexpr uint32_t cb_vraw = 21;   // v row (row 0)
constexpr uint32_t cb_mask = 22;   // per user 0 / -inf over the tail columns (packed: per pass and tail tile)
constexpr uint32_t cb_parts = 23;  // fanin x kPart partials of this node's children (written by them)
constexpr uint32_t cb_gs = 26;     // leader: sigmoid(gate)
constexpr uint32_t cb_osum = 25;   // own + children's partials (kPart)
constexpr uint32_t cb_rl = 30;     // leader: 1 / row sum
constexpr uint32_t cb_out = 27;    // gated output (Dt)
constexpr uint32_t cb_flush = 28;  // updated tail k as bf8 (Dt per tail tile), written back to DRAM
constexpr uint32_t cb_flushv = 33;  // updated tail v in V's format (Dt per tail tile), written back to DRAM
constexpr uint32_t cb_mean = 29;   // reduce scaler 1 / (32 Dt)
constexpr uint32_t cb_had = 43;    // S0-8: H32 / 16 (one bf16 tile; resident) — leader and tail cores
constexpr uint32_t cb_rot2 = 46;   // S0-8: the rotation's butterfly ping-pong (Dt, bf16)
constexpr uint32_t cb_rotx = 44;   // S0-8: norm_rope's output before the rotation (Dt, bf16)
constexpr uint32_t cb_knr = 45;    // S0-8: the tail's rotated k rows before the writer rounds them into cb_knew
// S0-9 stage 1 (named arg v_rot): V is stored Hadamard-rotated (the V projection rows folded with HAD), so the leader
// rotates each user's normalized output back (HAD is its own inverse) before the gate — in its own scratch (cb_vr1 /
// cb_vr2, Dt each), apart from the staged q prep's cb_rotx / cb_rot2
constexpr bool kVRot = get_named_compile_time_arg_val("v_rot") != 0;
constexpr uint32_t cb_vr1 = 48, cb_vr2 = 49;
// S0-9 2d (kVRotPass: the batched leader's passes, kGatePass): the rotation back once per q pass instead of per user —
// compute normalizes user j of a pass into cb_vr1, the reader places its rows at rows j * heads .. of the pass's tile
// set (cb_vp: cb_gs's space, the per-user gate rows are not extracted then), and at the pass's end compute rotates the
// set (cb_vp -> cb_vr2, the butterflies through cb_rot2 into cb_rotx: the q prep's scratch, empty at a pass's end) and
// gates it with the pass's cb_gs_pass into cb_out; the writer stages the pass's users from it
constexpr bool kVRotPass = kVRot && kGatePass;
constexpr uint32_t cb_vp = cb_gs;
// (kVRotPass: a token a user, leader compute -> writer, pushed once its partials are summed. The writer's wait for user
// i - 1's output before M(i + 1) is what keeps the group heads from writing partial(i + 1) into the parity half compute
// still reads partial(i - 1) from; with the output once a pass, the writer waits for this token instead.)
constexpr uint32_t cb_vdone = 50;

// S0-8: the device packer truncates toward zero when it packs bfp4, so a decode-written K row is rounded to nearest on
// the bfp4 grid first (by the tail writer, as prefill/kernels/v_rtn.cpp rounds the prefill's): one face row (16 bf16
// sharing an exponent in the packed tile), 3 magnitude bits, ties away from zero, m capped at 7
FORCE_INLINE uint32_t rtn_bfp4_one(uint32_t b, uint32_t emax) {
    const uint32_t e = (b >> 7) & 0xFF, sign = b & 0x8000;
    const uint32_t sh = emax - e + 5;
    if (e == 0 || sh >= 16) {
        return sign;  // (a signed zero: below the row's smallest step)
    }
    uint32_t m = (128 + (b & 0x7F) + (1u << (sh - 1))) >> sh;
    if (m == 0) {
        return sign;
    }
    m = m > 7 ? 7 : m;
    const uint32_t p = m >= 4 ? 2 : (m >= 2 ? 1 : 0);  // m's leading bit
    return sign | ((emax - 2 + p) << 7) | ((m << (7 - p)) & 0x7F);
}

// one face row as 8 words (two bf16 each)
FORCE_INLINE void rtn_bfp4_face_row(volatile tt_l1_ptr uint32_t* w) {
    uint32_t emax = 0;
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t x = w[i], e0 = (x >> 7) & 0xFF, e1 = (x >> 23) & 0xFF;
        emax = e0 > emax ? e0 : emax;
        emax = e1 > emax ? e1 : emax;
    }
    if (emax < 3) {
        for (uint32_t i = 0; i < 8; i++) {
            w[i] = w[i] & 0x80008000u;
        }
        return;
    }
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t x = w[i];
        w[i] = rtn_bfp4_one(x & 0xFFFF, emax) | (rtn_bfp4_one(x >> 16, emax) << 16);
    }
}

// S0-8 (off the critical path): a decode-written K row re-encoded in the packed bfp4 tile before the write-back
#include "bfp4_rtn.hpp"
// (A/B: kv_k_patch 0 skips the re-encoding — the stored row stays truncated)
constexpr uint32_t kKPatch = get_named_compile_time_arg_val("kv_k_patch");  // (2: re-encode truncating, a debug A/B)

// row `row` of a bf16 32 x 32 tile at `tile` (L1): its two face rows (faces 0 / 1 for rows < 16, 2 / 3 above)
FORCE_INLINE void rtn_bfp4_tile_row(uint32_t tile, uint32_t row) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    const uint32_t base = (row < 16 ? 0 : 256) + (row & 15) * 8;  // (words: a face is 128)
    rtn_bfp4_face_row(t + base);
    rtn_bfp4_face_row(t + base + 128);
}

// S0-9 stage 2 (named arg kv_v_bfp2): V is stored in Bfp2_b — per 16-element face row one shared exponent (the row's
// max), a value is 0 or +-2^e (common.py bfp2_round: +-2^e where |x| >= 2^(e - 1), else 0 — bf16 exponent field >=
// emax - 1). The device packer truncates (values below 2^e become 0), so a decode-written v row is rounded in software:
// plain rows (row groups or one user a pass) as S0-8's K — the step's attention takes the unrounded row; the tail
// reader encodes the user's new row to nearest (bfp2_encode_row: per face row its exponent byte and magnitude / sign
// word, from the row alone) while it waits for q, into cb_venc, and the tail writer places that into the packed tail
// tile at the write-back (bfp2_place_row: 16 bytes and words, the tail writer being the 32-row step's per-user
// bound); verify rows — whose tail tiles are kept in V's format within the step — are
// rounded in bf16 by the tail reader before the tail update (rtn_bfp2_tile_row), which the packer then carries exactly.
// Both on two bf16 a word at once: E = the two exponent fields (bits 0..7, 16..23); a value stays iff E >= emax - 1
// (emax >= 2; a row of emax <= 1 — magnitudes below 2^-125 — becomes 0), the lane test borrow-free with guard bits 8 /
// 24
constexpr bool kVBfp2 = get_named_compile_time_arg_val("kv_v_bfp2") != 0;
constexpr uint32_t cb_venc = 47;     // S0-9 plain rows: a user's encoded new v row (bfp2_encode_row's page)
constexpr uint32_t kVEncPage = 128;  // (Dt x 2 face rows: their magnitude words, then their exponent bytes at 64)
FORCE_INLINE uint32_t bfp2_emax(const uint32_t* w) {
    uint32_t m = 0;  // (per lane: the max over the row's 8 words, lanes kept apart)
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t e = (w[i] >> 7) & 0x00FF00FFu;
        const uint32_t lo = e & 0xFF, hi = e >> 16;
        const uint32_t x = lo > hi ? lo : hi;
        m = x > m ? x : m;
    }
    return m;
}
FORCE_INLINE uint32_t bfp2_keep(uint32_t x, uint32_t t2) {
    // bits 8 / 24 set where the lane's exponent >= emax - 1 (t2 = (emax - 1) in both lanes)
    return (((x >> 7) & 0x00FF00FFu) | 0x01000100u) - t2;
}
FORCE_INLINE void rtn_bfp2_face_row(volatile tt_l1_ptr uint32_t* row) {
    uint32_t w[8];
    for (uint32_t i = 0; i < 8; i++) {
        w[i] = row[i];
    }
    const uint32_t emax = bfp2_emax(w);
    if (emax <= 1) {
        for (uint32_t i = 0; i < 8; i++) {
            row[i] = 0;
        }
        return;
    }
    const uint32_t t2 = (emax - 1) * 0x00010001u, val = (emax << 7) * 0x00010001u;
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t g = (bfp2_keep(w[i], t2) >> 8) & 0x00010001u;
        row[i] = ((w[i] & 0x80008000u) | val) & (g * 0xFFFFu);
    }
}
FORCE_INLINE void rtn_bfp2_tile_row(uint32_t tile, uint32_t row) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    const uint32_t base = (row < 16 ? 0 : 256) + (row & 15) * 8;
    rtn_bfp2_face_row(t + base);
    rtn_bfp2_face_row(t + base + 128);
}
// Bfp2_b packing (64 exponent bytes, face f row r at 16 f + r; then a word a face row, element c at bits 2c:
// magnitude, 2c + 1: sign): row src_row of the Dt bf16 tiles at s encoded to nearest into a kVEncPage page at p (word
// 2 d + h: tile d's face-row half h; exponent byte 64 + 2 d + h), then placed as row `row` of the packed tiles at t
// (tile stride kVTile)
FORCE_INLINE void bfp2_encode_row(uint32_t s, uint32_t src_row, uint32_t p) {
    volatile tt_l1_ptr uint32_t* code = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
    volatile tt_l1_ptr uint8_t* code_ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(p + 64);
    for (uint32_t d = 0; d < Dt; d++) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + d * kTile);
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t sf = (src_row < 16 ? 0 : 2) + h, sr = src_row & 15;
            uint32_t w[8];
            for (uint32_t i = 0; i < 8; i++) {
                w[i] = src[sf * 128 + sr * 8 + i];
            }
            const uint32_t emax = bfp2_emax(w);
            uint32_t o = 0;
            if (emax > 1) {
                const uint32_t t2 = (emax - 1) * 0x00010001u;
                for (uint32_t i = 0; i < 8; i++) {
                    const uint32_t g = bfp2_keep(w[i], t2);
                    const uint32_t m0 = (g >> 8) & 1, m1 = (g >> 24) & 1;
                    const uint32_t s0 = (w[i] >> 15) & m0, s1 = (w[i] >> 31) & m1;
                    o |= (m0 | (s0 << 1) | (m1 << 2) | (s1 << 3)) << (4 * i);
                }
            }
            code[2 * d + h] = o;
            code_ex[2 * d + h] = emax;
        }
    }
}
FORCE_INLINE void bfp2_place_row(uint32_t p, uint32_t t, uint32_t row) {
    volatile tt_l1_ptr uint32_t* code = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
    volatile tt_l1_ptr uint8_t* code_ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(p + 64);
    for (uint32_t d = 0; d < Dt; d++) {
        volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(t + d * kVTile);
        volatile tt_l1_ptr uint32_t* man = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + d * kVTile + 64);
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t f = (row < 16 ? 0 : 2) + h, r = row & 15;
            ex[f * 16 + r] = code_ex[2 * d + h];
            man[f * 16 + r] = code[2 * d + h];
        }
    }
}
constexpr uint32_t cb_stage = 2;   // leader: gated output rows as batch x 32 tiles (heads x Dt)

// byte offset of row r inside a 32x32 bf16 tile, face 0 (columns 0..15); face 1 is + kFace
FORCE_INLINE uint32_t row_offset(uint32_t r) { return (r < 16 ? 0 : 2 * kFace) + (r & 15) * kFaceRow; }

// byte offset of user u's face 0 row inside a batch x 32 tile; its face 1 row is + batch * kFaceRow
FORCE_INLINE uint32_t user_offset(uint32_t u) { return u * kFaceRow; }

// chunk of chunk worker w at tp complete position tiles: its tiles t0, t0 + dt, ... (n of them, 0: no chunk);
// tile t lives in bank t % banks, row t / banks
struct Chunk {
    uint32_t t0, dt, n;
};
FORCE_INLINE Chunk chunk_of(uint32_t w, uint32_t tp) {
    if constexpr (round_robin) {
        constexpr uint32_t W = kChunkWorkers;
        return {w, W, tp > w ? (tp - w + W - 1) / W : 0};
    }
    const uint32_t live = tp < banks ? tp : banks;
    if (live == 0) {
        return {0, 1, 0};
    }
    const uint32_t b = w % live;
    const uint32_t j = w / live;
    const uint32_t J = (kChunkWorkers - b + live - 1) / live;
    const uint32_t rows = (tp - b + banks - 1) / banks;  // tiles b, b + banks, ... below tp
    return {j * banks + b, J * banks, rows > j ? (rows - j + J - 1) / J : 0};
}

// whether chunk ch holds tile t
FORCE_INLINE bool holds(const Chunk& ch, uint32_t t) {
    return t >= ch.t0 && (t - ch.t0) % ch.dt == 0 && (t - ch.t0) / ch.dt < ch.n;
}

// the chunk worker whose chunk at tp complete tiles holds tile t < tp
FORCE_INLINE uint32_t owner_of(uint32_t t, uint32_t tp) {
    if constexpr (round_robin) {
        return t % kChunkWorkers;
    }
    const uint32_t live = tp < banks ? tp : banks;
    const uint32_t b = t % banks;
    const uint32_t J = (kChunkWorkers - b + live - 1) / live;
    return b + live * ((t / banks) % J);
}

}  // namespace resident_attn
