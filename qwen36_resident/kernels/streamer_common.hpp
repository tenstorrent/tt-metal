// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident Qwen3.6 decode: shared compile-time schedule of the streamer cores. One launch runs one
// decode step through `layers` decoder layers (layer l is a full-attention layer when
// attn_interval > 0 and (l + 1) % attn_interval == 0, else a linear-attention (Gated DeltaNet) layer)
// and, optionally, the lm_head. Every layer is two residual halves:
//   mixer: h = rmsnorm(x) (the norm weight is folded into the weights);  y = h @ W_proj (this core's
//          columns; GDN: qkvzab, conv1d(4 taps) + silu on the q|k|v columns, the conv history kept in
//          DRAM; attention: q | gate | k | v);  y goes to the layer's mixer cores (GDN head cores or
//          the attention leader + tail core), which return o;  partial = o @ W_out
//   mlp:   h = rmsnorm(x);  partial = down(silu(h @ G) * (h @ U))
// After each half the hub all-reduces the chips' partials (chip 0 folds x in), sums them and multicasts
// the new x to every streamer. The lm_head computes
// this core's logit columns of rmsnorm(x) @ W_head into an L1 buffer read by the host.
//
// Weights: one DRAM tensor per entry holding `distinct` layers of that kind at a fixed per-bank stride;
// the k-th layer of a kind uses copy k % distinct (benchmarks run many layers over few copies).
//
// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 attn_interval, 2 num_chips, 3 num_streamers, 4 Ht, 5 It, 6 chip, 7 ring_bytes,
//   8 rms eps bits, 9 1/sqrt(hidden) bits, 10 sem_slots, 11 sem_act, 12 sem_gather, 13 sem_heads,
//   14 sem_rows, 15 ng_max, 16 nd_max, 17 nq_max (projection tiles per core, <= kBlk), 18 nv_max
//   (lm_head tiles per core), 19 conv_tiles (q|k|v tiles per chip), 20 Ot (out-proj K tiles),
//   21 gdn_heads (GDN head cores), 22 lm_head (0 / 1), then per entry e (see kEntries): 23 + 5e Kt, sb, pages,
//   page_size, block_bytes, then 53 sem_local (conv history write-backs done, for the reader), 54 batch
//   (users per step, 1 / 2 / 4 / 8), 55 sem_addr (CB addresses exchanged with the hub), 56 o_heads (GDN
//   head cores that write o, one per value head), 57 column block tiles (kBlk), 58 sem_ring (local: the
//   ring step + 1, from the reader to the writer), 59 verify (0 / 1), 60 fc (0 / 1), 61 sample (0 / 1: the
//   lm_head writes this core's top sampling::kTopK logits instead of its largest, see sampling.hpp)
//
// fc (the MTP draft layer, no GDN layers): x of layer 0 is fc([norm_e(e) | norm_h(h)]) for the token state's
// two vectors per row (embedding e, then the main model's last x as h). Chip c's K slice of the fc is slice
// c % (chips / 2) of rmsnorm(e) (first half of the chips) or rmsnorm(h) (second half), the norm weights folded
// into the fc rows; its weights stream through entry kQkvz and the partials make an extra hub round 0, so the
// layers' rounds start at 1.
//
// Verify: the batch is two rows of one user at positions p and p + 1 (a token and its draft), so row 1's
// conv window holds row 0's input of the same step. The history of a GDN layer is then a ring of 4 pair
// slots: T(j) holds (v_j, v_j+1) as rows (0, 1) of every 32x32 view, for the layer's inputs v. At ring
// step k (token state word 0) the window is T(k), T(k + 1) and T(k + 2) with row 1 of T(k + 2) masked off
// and replaced by row 0 of this step's input y (a 0/1 matmul moves view row R - 1 to odd row R); the
// step writes T(k + 2) = (v_k+2, y0), T(k + 3) = y and T(k + 4) = (y1, -). The next step starts at ring
// step k + 1 (row 0 kept) or k + 2 (both rows kept); only row 0 of the third slot is ever read.
//
// Batch: every activation tile is a batch x 32 tile, row u for user u (face 0 holds columns 0..15 of
// all rows, face 1 columns 16..31). The weights stream once per step for all users.
#pragma once

#include <stdint.h>

namespace resident {

constexpr uint32_t kEntries = 6;
constexpr uint32_t kQkvz = 0, kQkvg = 1, kOut = 2, kGateUp = 3, kDown = 4, kHead = 5;

// streamer CBs
constexpr uint32_t cb_w0 = 1;  // weight ring, entry e -> cb_w0 + e (1..6)
constexpr uint32_t cb_consumed = 7;
constexpr uint32_t cb_act = 8;    // mlp intermediate activation (It)
constexpr uint32_t cb_dout = 9;   // out-proj / down columns before the chip-0 residual fold
constexpr uint32_t cb_pout = 10;  // partial sent to the hub
constexpr uint32_t cb_x = 11;
constexpr uint32_t cb_x_full = 12;
constexpr uint32_t cb_h = 13;
constexpr uint32_t cb_h_full = 14;
constexpr uint32_t cb_g = 15;  // gate columns of this core (one full-tile block of 1x32 tiles)
constexpr uint32_t cb_aslice = 16;
constexpr uint32_t cb_qkvz = 17;       // projection columns of this core
constexpr uint32_t cb_qkvz_out = 18;   // after conv1d + silu (q|k|v columns) / passthrough
constexpr uint32_t cb_conv_w = 19;     // this GDN layer's 4 taps x kBlk (streamed)
constexpr uint32_t cb_conv_hist = 20;  // this GDN layer's 3 previous inputs x kBlk, oldest first (streamed)
constexpr uint32_t cb_o_in = 21;       // mixer output (Ot), written by the mixer cores
constexpr uint32_t cb_u = 26;          // up columns of this core
constexpr uint32_t cb_hist_out = 30;   // this step's input to the conv (the newest history entry)
constexpr uint32_t cb_logits = 31;     // lm_head columns of this core (read by the host)
// The per-core column blocks of the elementwise phases (conv1d + silu, silu(g) * u) hold kBlk batch x 32
// tiles (CT 57: a whole number of 32x32 views, 32 / batch tiles each), so the SFPU math runs on the 32x32
// views below instead of once per small tile (an SFPU op covers a full 32x32 tile).
constexpr uint32_t kBlk = get_compile_time_arg_val(57);
constexpr uint32_t kBlkViews = kBlk * get_compile_time_arg_val(54) / 32;
static_assert(kBlkViews * 32 == kBlk * get_compile_time_arg_val(54), "a column block is whole 32x32 views");
constexpr uint32_t cb_qkvz_full = 22;
constexpr uint32_t cb_qkvz_out_full = 23;
constexpr uint32_t cb_conv_w_full = 24;     // [tap] full tiles
constexpr uint32_t cb_conv_hist_full = 25;  // [slot] full tiles, oldest first
constexpr uint32_t cb_g_full = 27;
constexpr uint32_t cb_u_full = 28;
constexpr uint32_t cb_aslice_full = 29;
constexpr uint32_t kUnit = 64;
constexpr uint32_t batch = get_compile_time_arg_val(54);
static_assert(batch == 1 || batch == 2 || batch == 4 || batch == 8, "custom_mm takes 1, 2, 4 or 8 rows");
constexpr uint32_t kTileBytes = 64 * batch;        // a batch x 32 bf16 tile
constexpr uint32_t kBlkBytes = kBlk * kTileBytes;  // kBlkViews 32x32 bf16 tiles
// 32x32 views of a column block that hold its first n tiles
constexpr uint32_t views_of(uint32_t n) { return (n * batch + 31) / 32; }
// rmsnorm of a batch > 1: ones and the row-fold matrix (see rmsnorm_rows)
constexpr uint32_t cb_ones_full = 32;
constexpr uint32_t cb_fold = 33;
constexpr uint32_t cb_sumsq = 34;  // fp32 scratch, 2 tiles
constexpr uint32_t cb_rinv = 35;   // fp32, per-row 1 / rms on the view rows
constexpr uint32_t cb_eye = 38;    // rmsnorm of a batch > 1: the identity (bf16 32x32)
constexpr uint32_t cb_shift = 36;  // verify: even-row identity, row R - 1 -> odd R, row R + 1 -> even R

// token state (DRAM, uint32 words): [0] conv ring step (advances once per step), [1 + u] position of
// user u; x0 (hidden bf16 as batch x 32 tiles) at kTokX0; per user the rope cos, sin, -sin tiles of its
// position (32x32 bf16, rows replicated) at kTokRope
constexpr uint32_t kTokX0 = 64;

// optional timeline: per layer kTsWords words, reader [0, 8), writer [8, 16), compute [16, 26)
constexpr uint32_t kTsWords = 32;

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t attn_interval = get_compile_time_arg_val(1);
constexpr uint32_t num_chips = get_compile_time_arg_val(2);
constexpr uint32_t num_streamers = get_compile_time_arg_val(3);
constexpr uint32_t Ht = get_compile_time_arg_val(4);
constexpr uint32_t It = get_compile_time_arg_val(5);
constexpr uint32_t chip = get_compile_time_arg_val(6);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(7);
constexpr uint32_t eps_bits = get_compile_time_arg_val(8);
constexpr uint32_t inv_sqrt_hidden_bits = get_compile_time_arg_val(9);
constexpr uint32_t sem_slots = get_compile_time_arg_val(10);
constexpr uint32_t sem_act = get_compile_time_arg_val(11);
constexpr uint32_t sem_gather = get_compile_time_arg_val(12);
constexpr uint32_t sem_heads = get_compile_time_arg_val(13);
constexpr uint32_t sem_rows = get_compile_time_arg_val(14);
constexpr uint32_t ng_max = get_compile_time_arg_val(15);
constexpr uint32_t nd_max = get_compile_time_arg_val(16);
constexpr uint32_t nq_max = get_compile_time_arg_val(17);
constexpr uint32_t nv_max = get_compile_time_arg_val(18);
constexpr uint32_t conv_tiles = get_compile_time_arg_val(19);
constexpr uint32_t Ot = get_compile_time_arg_val(20);
constexpr uint32_t gdn_heads = get_compile_time_arg_val(21);
constexpr bool lm_head = get_compile_time_arg_val(22) != 0;
constexpr uint32_t Hf = Ht / 32;
constexpr uint32_t kTokRope = kTokX0 + (get_compile_time_arg_val(60) ? 2 : 1) * Ht * kTileBytes;
constexpr uint32_t kTokBytes = kTokRope + 3 * 2048 * batch;
static_assert(
    get_compile_time_arg_val(15) <= 32 && get_compile_time_arg_val(17) <= 32, "column blocks exceed one full tile");

constexpr uint32_t sem_local = get_compile_time_arg_val(53);
constexpr uint32_t sem_addr = get_compile_time_arg_val(55);
constexpr uint32_t o_heads = get_compile_time_arg_val(56);
constexpr uint32_t sem_ring = get_compile_time_arg_val(58);
constexpr bool verify = get_compile_time_arg_val(59) != 0;
static_assert(!verify || batch == 2, "verify runs two rows of one user");
constexpr uint32_t kHistSlots = verify ? 4 : 3;  // conv history slots per (streamer, GDN copy) row
constexpr uint32_t gdn_layers = layers - (attn_interval > 0 ? layers / attn_interval : 0);
constexpr bool fc_in = get_compile_time_arg_val(60) != 0;
constexpr bool sample = get_compile_time_arg_val(61) != 0;
static_assert(!sample || batch == 1, "on-device sampling serves batch 1");
static_assert(!fc_in || gdn_layers == 0, "fc streams through the GDN projection entry");
constexpr uint32_t kRound0 = fc_in ? 1 : 0;  // hub round of layer 0's mixer half

FORCE_INLINE bool is_attn(uint32_t l) { return attn_interval > 0 && (l + 1) % attn_interval == 0; }

// The conv history of a GDN copy is a ring of 3 slots advanced once per use: GDN layer g (use g / copies
// of copy g % copies within the step) at ring step s (token state word 0) is the ring's step
// s * uses + g / copies; slot (step + j) % 3 holds the j-th oldest input and the step's input replaces slot
// step % 3. All users share the ring order, so it follows the step count, not a position.
FORCE_INLINE uint32_t conv_step(uint32_t s, uint32_t g, uint32_t copies) {
    const uint32_t c = g % copies;
    const uint32_t uses = (gdn_layers - c + copies - 1) / copies;
    return s * uses + g / copies;
}

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(23 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(23 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(23 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(23 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(23 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

struct RingCursor {
    uint32_t pos = 0;
    uint32_t units = 0;

    template <uint32_t E>
    uint32_t place() {
        constexpr uint32_t block = Entry<E>::block_bytes;
        if (pos + block > ring_bytes) {
            units += (ring_bytes - pos) / kUnit;
            pos = 0;
        }
        const uint32_t at = pos;
        pos += block;
        units += block / kUnit;
        return at;
    }
};

}  // namespace resident
