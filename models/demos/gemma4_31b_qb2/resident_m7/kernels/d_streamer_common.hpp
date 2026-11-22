// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter (M6b, bench/resident/m6/M6A_DESIGN.md): shared compile-time schedule of the D-program's
// streamer cores. The weight ring, the deterministic block placement and the per-bank streaming are streamer_common.hpp's
// (the target's), unchanged; the entries and the per-layer flow are the drafter's (Qwen3 pre-norm layers with DFlash2's
// dynamic block convs). One launch: `layers` drafter layers on R = 32 rows (U user blocks of Mb rows), then the
// LM head. Per layer:
//   h  (input_norm(x), from the slices) @ Kp_attn -> the conv coefficients, this core's coefficient tiles to every slice
//   xin (conv_in(h), from the slices) @ W_qkv; attention STUB: o = q (the chip's 8 q heads, the o_proj input), gathered
//      into every streamer's o_in; partial = o_in @ W_o -> its slice and the hub (all-reduce round 2 l)
//   h2 (post_norm(x)) @ Kp_mlp -> coefficient tiles to the slices
//   m_in (conv_in(h2)) @ [G | U] -> silu(g) * u, gathered into every streamer's act; partial = act @ W_down
//      (all-reduce round 2 l + 1)
// then logits = final_norm(x) @ W_head (the target's LM head columns of this core) and the per-row argmax records
// (the hub reduces them, as the target's).
#pragma once

#include <stdint.h>

namespace dres {

constexpr uint32_t kEntries = 7;
constexpr uint32_t kKpA = 0, kQkv = 1, kO = 2, kKpM = 3, kGU = 4, kDown = 5, kHead = 6;

// streamer CBs
constexpr uint32_t cb_w0 = 0;         // weight ring, entry e -> cb_w0 + e (0..6), all aliasing one allocation
constexpr uint32_t cb_consumed = 7;   // its tiles_acked register carries compute's consumed ring units
constexpr uint32_t cb_h = 8;          // the activation from the slices (Ht tiles): h, xin, h2, m_in, final h in turn
constexpr uint32_t cb_q = 9;          // this core's q|k|v columns (one block of kBlk tiles)
constexpr uint32_t cb_o_in = 10;      // the gathered o_proj input (kAct tiles), backed by the act buffer
constexpr uint32_t cb_pout = 11;      // partial sent to the slice and the hub (nd_max tiles)
constexpr uint32_t cb_g = 12;
constexpr uint32_t cb_u = 14;
constexpr uint32_t cb_aslice = 16;    // silu(g) * u of this core
constexpr uint32_t cb_act = 18;       // the gathered MLP activation (kAct tiles), backed by the act buffer
constexpr uint32_t cb_logits = 19;    // LM head columns of this core, backed by the act buffer
constexpr uint32_t cb_coef = 20;      // this core's conv-coefficient columns (kBlkC tiles)

constexpr uint32_t kUnit = 64;

// Compile-time args (identical on the three RISCs of a streamer):
//   0 layers, 1 num_streamers, 2 Ht, 3 It, 4 chip, 5 ring_bytes, 6 ng_max, 7 nd_max, 8 nq_max, 9 nv_max, 10 kBlk,
//   11 kAct, 12 sem_slots, 13 sem_o, 14 sem_act, 15 Ot (q tiles of the chip: the o_proj input width), 16 nc_max,
//   17 R (32), 18 M (rows), 19 K (slices), then per entry e at 20 + 5e: Kt, sb, pages, page_size, block_bytes;
//   55 sem_coef (the slices' coefficient-arrival semaphore), 56 kBlkC (coefficient block tiles), 57 coefficient tiles
//   per slice block (48: the 4 sections). (Teacher forcing runs drafter layers [off, off + layers): the host points
//   every per-layer entry's base at layer off.)
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
constexpr uint32_t kOt = get_compile_time_arg_val(15);
constexpr uint32_t nc_max = get_compile_time_arg_val(16);
constexpr uint32_t R = get_compile_time_arg_val(17);
constexpr uint32_t M = get_compile_time_arg_val(18);
constexpr uint32_t K = get_compile_time_arg_val(19);
constexpr uint32_t sem_coef = get_compile_time_arg_val(20 + 5 * kEntries);
constexpr uint32_t kBlkC = get_compile_time_arg_val(21 + 5 * kEntries);
constexpr uint32_t kCoefTiles = get_compile_time_arg_val(22 + 5 * kEntries);
static_assert(R == 32, "the D-program runs R = 32 activation tiles (32 x 32): every projection matmul_block, BF8 at HiFi2");
static_assert(M >= 1 && M <= R, "rows");
constexpr uint32_t kTileBytes = 64 * R;
constexpr uint32_t kMw = (M + 1) / 2;  // the writer scans logit rows [0, kMw), the reader [kMw, M)
constexpr uint32_t kRecStride = (R * 8 + 15) / 16 * 16;

// byte offset of row r's 16 elements of face f in a 32 x 32 bf16 tile
FORCE_INLINE uint32_t face_row_offset(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
FORCE_INLINE uint32_t key_of(uint32_t b) { return (b & 0x8000) ? (~b & 0xffff) : (b | 0x8000); }

// rows [r0, r1) of the logits (nv tiles at `base`): per row the largest (order key, vocab index), the first on ties
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

// streamer_common.hpp's ring cursor (deterministic block placement; the reader and compute agree on every block)
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

}  // namespace dres
