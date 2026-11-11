// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma-4-31B resident decode prototype (step 1, attention stubbed): shared compile-time schedule of the
// streamer cores. Adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/kernels/streamer_common.hpp
// (Apache-2.0); the weight ring and its deterministic block placement are that design.
//
// One launch runs one decode step of M rows (users or verify rows, 1 <= M <= 32): `layers` decoder layers, then the
// LM head. Activations are R x 32 tiles (R = 1, 2, 4, 8: row r of every tile is row r; R = 32: full tiles, rows past
// M zero); the weights stream once per launch for all rows. Per layer (layer l is full attention when l % 6 == 5):
//   mixer half: h (from the slices) @ W_qkv (this core's columns); attention STUB: o = q, the first Ot columns of the
//               projection row (the chip's 8 q heads = the o_proj input width), gathered from every streamer into
//               every streamer's o_in; partial = o_in @ W_o (this core's hidden columns) -> its slice and the hub
//   mlp half:   h @ [G | U] -> gelu_tanh(g) * u, gathered into every streamer's act; partial = act @ W_down
// The hub carries the partials over fabric to the peer chips' slices; K slice cores own the residual stream by
// hidden columns (post norm, residual, the next pre-norm) and multicast their columns of the next h.
#pragma once

#include <stdint.h>

namespace gres {

constexpr uint32_t kEntries = 7;
constexpr uint32_t kQkvS = 0, kQkvF = 1, kOS = 2, kOF = 3, kGU = 4, kDown = 5, kHead = 6;

// streamer CBs
constexpr uint32_t cb_w0 = 0;         // weight ring, entry e -> cb_w0 + e (0..6), all aliasing one allocation
constexpr uint32_t cb_consumed = 7;   // its tiles_acked register carries compute's consumed ring units
constexpr uint32_t cb_h = 8;          // h from the slices (Ht tiles), backed by an L1 tensor
constexpr uint32_t cb_q = 9;          // this core's projection columns (one block of kBlk tiles)
constexpr uint32_t cb_o_in = 10;      // the gathered o_proj input (kAct tiles), backed by the act tensor
constexpr uint32_t cb_pout = 11;      // partial sent to the slice and the hub (nd_max tiles)
constexpr uint32_t cb_g = 12;         // gate columns (R x 32 tiles) ...
constexpr uint32_t cb_g_full = 13;    // ... and the same memory as 32x32 views
constexpr uint32_t cb_u = 14;
constexpr uint32_t cb_u_full = 15;
constexpr uint32_t cb_aslice = 16;    // gelu_tanh(g) * u of this core
constexpr uint32_t cb_aslice_full = 17;
constexpr uint32_t cb_act = 18;       // the gathered MLP activation (kAct tiles), backed by the act tensor
constexpr uint32_t cb_logits = 19;    // LM head columns of this core, backed by the act tensor

constexpr uint32_t kUnit = 64;

// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 num_streamers, 2 Ht, 3 It, 4 chip, 5 ring_bytes, 6 ng_max, 7 nd_max, 8 nq_max, 9 nv_max, 10 kBlk,
//   11 kAct (tiles of the shared o_in / act / logits buffer), 12 sem_slots, 13 sem_o, 14 sem_act, 15 Ot sliding,
//   16 Ot full, 17 R (tile rows), 18 M (rows), 19 K (slices), then per entry e at 20 + 5e: Kt, sb, pages, page_size,
//   block_bytes
constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t num_streamers = get_compile_time_arg_val(1);
constexpr uint32_t Ht = get_compile_time_arg_val(2);
constexpr uint32_t It = get_compile_time_arg_val(3);
constexpr uint32_t chip = get_compile_time_arg_val(4);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(5);
constexpr uint32_t ng_max = get_compile_time_arg_val(6);
constexpr uint32_t nd_max = get_compile_time_arg_val(7);
constexpr uint32_t nq_max = get_compile_time_arg_val(8);
constexpr uint32_t nv_max = get_compile_time_arg_val(9);
constexpr uint32_t kBlk = get_compile_time_arg_val(10);
constexpr uint32_t kAct = get_compile_time_arg_val(11);
constexpr uint32_t sem_slots = get_compile_time_arg_val(12);
constexpr uint32_t sem_o = get_compile_time_arg_val(13);
constexpr uint32_t sem_act = get_compile_time_arg_val(14);
constexpr uint32_t kOtS = get_compile_time_arg_val(15);
constexpr uint32_t kOtF = get_compile_time_arg_val(16);
constexpr uint32_t R = get_compile_time_arg_val(17);
constexpr uint32_t M = get_compile_time_arg_val(18);
constexpr uint32_t K = get_compile_time_arg_val(19);
static_assert(R == 1 || R == 2 || R == 4 || R == 8 || R == 32, "tile rows");
static_assert(M >= 1 && M <= R, "rows");
constexpr bool kCustomMm = R <= 8;              // M <= 8: custom_mm on R x 32 tiles; else matmul on 32 x 32 tiles
constexpr uint32_t kTileBytes = 64 * R;          // an R x 32 bf16 tile
constexpr uint32_t kBlkBytes = kBlk * kTileBytes;
constexpr uint32_t kViewTiles = 32 / R;          // tiles per 32x32 view (R <= 32)
static_assert(kBlk % kViewTiles == 0, "a column block is whole 32x32 views");
static_assert(nq_max <= kBlk && ng_max <= kBlk, "column blocks");
constexpr uint32_t views_of(uint32_t n) { return (n * R + 31) / 32; }
constexpr uint32_t kMw = (M + 1) / 2;            // the writer scans logit rows [0, kMw), the reader [kMw, M)
constexpr uint32_t kRecStride = (R * 8 + 15) / 16 * 16;  // bytes of a core's argmax records on the hub (16 B aligned)

// optional timeline: per layer kTsWords words, reader [0, 8), writer [8, 16)
constexpr uint32_t kTsWords = 16;

FORCE_INLINE bool is_full(uint32_t l) { return l % 6 == 5; }

// byte offset of row r's 16 elements of face f (columns 16 f .. 16 f + 15) in an R x 32 bf16 tile
FORCE_INLINE uint32_t face_row_offset(uint32_t r, uint32_t f) {
    if constexpr (R == 32) {
        return (r / 16) * 1024 + f * 512 + (r % 16) * 32;
    } else {
        return f * R * 32 + r * 32;
    }
}

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
FORCE_INLINE uint32_t key_of(uint32_t b) { return (b & 0x8000) ? (~b & 0xffff) : (b | 0x8000); }

// rows [r0, r1) of the logits (nv tiles at `base`): per row the largest (order key, vocab index), the first on ties,
// written as 2 words per row at rec[2 r]
FORCE_INLINE void argmax_rows(
    uint32_t base, uint32_t nv, uint32_t first_index, uint32_t r0, uint32_t r1, volatile tt_l1_ptr uint32_t* rec) {
    for (uint32_t r = r0; r < r1; r++) {
        uint32_t best = 0, best_i = 0;
        for (uint32_t t = 0; t < nv; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                volatile tt_l1_ptr uint32_t* row =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * kTileBytes + face_row_offset(r, f));
                const uint32_t col0 = t * 32 + f * 16;
                for (uint32_t w = 0; w < 8; w++) {
                    const uint32_t v = row[w];
                    const uint32_t k0 = key_of(v & 0xffff), k1 = key_of(v >> 16);
                    if (k0 > best) {
                        best = k0;
                        best_i = col0 + 2 * w;
                    }
                    if (k1 > best) {
                        best = k1;
                        best_i = col0 + 2 * w + 1;
                    }
                }
            }
        }
        rec[2 * r] = best;
        rec[2 * r + 1] = first_index + best_i;
    }
}
#endif

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(20 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(20 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(20 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(20 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(20 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

// Deterministic block placement in the weight ring: a block that does not fit before the end starts over at the
// ring base, so the reader and compute derive every block's position independently. units counts 64 B units
// accounted so far (skipped ring tails included), mod 2^16.
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

}  // namespace gres
