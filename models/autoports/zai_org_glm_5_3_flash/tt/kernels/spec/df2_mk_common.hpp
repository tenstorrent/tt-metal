// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 in-trace drafter, rung 3 (tt/dflash2_mk.py): the drafter's layer / ctx programs as MkLayer-style
// megakernels. ONE generic_op per program: the DS bank readers (kernels/mk/reader_{w,a,c}.cpp, 7a's matmul phases)
// stream every matmul weight; 64 OWNER cores run the glue phases below between the matmuls; 2 fabric lane cores
// (kernels/fabric_ar) all-reduce the o / down partials (and all-gather the ctx fc output) in-program. Every hand-off
// is a semaphore (kernels/mk/mk_common.hpp conventions): the owners signal the readers' SEM_R (count 64) after an
// output landed, the readers signal the owners' SEM_CF / SEM_QKV / SEM_UG (count = all readers), the lanes signal the
// owners' SEM_AR / SEM_AR2 (count 2).
//
// Owner k holds the hidden column tiles {2k, 2k+1} of the [32, 4096] activations (rows = the block's 32 rows, K+1 <= 8
// valid; the rest garbage that stays finite). Owner phases (PH_*):
//   RMS      x (DRAM bf16 [32, H]) -> y = x * rsqrt(mean(x^2) + eps) * w (one bf16 rounding, as ttnn.rms_norm): the
//            partial sums of squares all-gathered over the 64 owners (kernels/head's pattern); y -> DRAM, SEM_R
//   RMS_AR   the same on the fabric all-reduce's result (packed rows 0..7 in the lanes' result buffer; rows >= 8 = 0)
//   CONV     the rung-2 grouped dynamic conv (kernels/spec/df2_conv_*: the same tile arithmetic) of x (DRAM) with the
//            kp readers' cf (SEM_CF) -> DRAM, SEM_R
//   FIN      res <- res + conv_side1(bf16 AR result) (the DF2_CONV_RES fold on the all-reduced partial): in place
//   HEADS    head h = owner k (q heads 0..NQH-1, k heads, v heads of the readers' q|k|v output, SEM_QKV): q / k: the
//            per-row RMS over the head's 4 tiles * w (bf16), RoPE (rotate-half, the PREP-written cos / sin tiles of the
//            rows' positions); q -> the SDPA input tiles (DRAM, token-row tiles: tt/dflash2_mk.py's layout trick), k /
//            v
//            -> the paged K / V cache rows at pos[t] (a read-modify-write of the destination tiles by the NCRISC: no
//            two owners share a tile, so no race)
//   SWI      m = up * silu(gate) of the ug readers' output (SEM_UG) -> DRAM, SEM_R
// RISC split: BRISC (dm0) reads the DRAM inputs and the lanes' packed rows, writes the DRAM outputs, signals the
// readers; NCRISC (dm1) the RMS partial-sum gather and the cache RMW; TRISC the arithmetic.
#pragma once

#include <cstdint>

namespace df2mk {

constexpr uint32_t TB = 2048;    // bf16 tile bytes
constexpr uint32_t TB32 = 4096;  // fp32 tile bytes
constexpr uint32_t N_OWNERS = 64;
constexpr uint32_t REC_BYTES = 128;  // one owner's partial sums: 32 fp32

// phase types
constexpr uint32_t PH_RMS = 0, PH_RMS_AR = 1, PH_CONV = 2, PH_FIN = 3, PH_HEADS = 4, PH_SWI = 5;

// owner CBs
constexpr uint32_t CB_X = 0;      // bf16 2: the RMS / CONV / FIN input tiles of columns c0, c0 + 1
constexpr uint32_t CB_W = 1;      // fp32 2: the norm weight row tiles (row 0 = w)
constexpr uint32_t CB_P = 2;      // fp32 1: this owner's partial sums of squares (column 0)
constexpr uint32_t CB_G = 3;      // fp32 2: the 64 owners' partials (owner o: column o % 32 of tile o / 32)
constexpr uint32_t CB_OUT = 4;    // bf16 4: the phase outputs (RMS 2, CONV 2, FIN 2, HEADS 4, SWI NTS)
constexpr uint32_t CB_LAND = 5;   // 64 x 128 B: the partials landing area (written by every owner's NCRISC)
constexpr uint32_t CB_REC = 6;    // 128 B: this owner's record
constexpr uint32_t CB_CF = 7;     // bf16 4: the conv cf tiles (2 per column tile: tap 0, tap 1)
constexpr uint32_t CB_E = 8;      // bf16 4: the conv selection tiles (2 per column tile)
constexpr uint32_t CB_J = 9;      // bf16 2: J[2 side], J[2 side + 1]
constexpr uint32_t CB_B = 10;     // bf16 2: the base tiles of columns c0, c0 + 1
constexpr uint32_t CB_SH = 11;    // bf16 1: the row shift tile
constexpr uint32_t CB_ZERO = 12;  // bf16 1: a zero tile
constexpr uint32_t CB_RES = 13;   // bf16 2: the residual tiles (FIN)
constexpr uint32_t CB_QKV = 14;   // bf16 4: a q / k head's 4 column tiles
constexpr uint32_t CB_HW = 15;    // fp32 4: the head norm weight row tiles
constexpr uint32_t CB_COS = 16;   // bf16 4: cos tiles of the rows' positions
constexpr uint32_t CB_SIN = 17;   // bf16 4
constexpr uint32_t CB_INV = 18;   // fp32 1: the head rows' 1 / rms (column 0)
constexpr uint32_t CB_HN = 19;    // bf16 4: the normed head tiles
constexpr uint32_t CB_V = 20;     // bf16 4: a v head's tiles (BRISC -> NCRISC)
constexpr uint32_t CB_SCR = 21;   // raw: pos (128) + page-table row + one tile + the owner table (512)
constexpr uint32_t CB_GU = 22;    // bf16 2 NTS: up tiles [0, NTS), gate tiles [NTS, 2 NTS)
constexpr uint32_t CB_KOUT = 23;  // bf16 HT: a k head's tiles after norm + RoPE (TRISC -> the NCRISC's cache write)
constexpr uint32_t N_CBS = 24;
// ONE consumer RISC and ONE producer RISC per CB over the whole program (the 2026-10-23 01:52Z silicon taps: the
// block's k heads read stale CB_OUT pages): cb_wait_front snapshots the shared pages_acked once and spins on
// pages_received, so a second consumer that arrives early (the NCRISC at HEADS, before the BRISC popped the RMS /
// CONV outputs) is released by the OTHER consumer's phases' pushes. CB_OUT is the BRISC's (RMS, CONV, FIN, SWI,
// the q heads), CB_KOUT the NCRISC's (the k heads), CB_V the NCRISC's (the v heads, BRISC-produced).
// CB discipline (the 2026-10-23 01:02Z watcher assert, dataflow_api.h:270 fifo_rd_ptr <= fifo_limit on an owner
// BRISC): a CB's chunks (reserve / push / wait / pop counts) differ between phases (CB_OUT: 2 RMS, 1 + 1 CONV, HT
// HEADS, NTS SWI) and a chunk must never straddle the CB's end, so EVERY RISC rewinds every owner CB to its base at
// the end of each phase (kernels/mk/mk_common.hpp cb_rewind: the local pointers only; every chunk a RISC pushed or
// popped in a phase is complete at its end, and the consumers of a phase's output pop it before the next phase's
// inputs are pushed). The host model tt/dflash2_mk.py cb_model asserts the balance and the no-straddle rule.

// semaphores: owners 0..5, readers 6
constexpr uint32_t SEM_G = 0;    // owners: the partial-sum gather (64)
constexpr uint32_t SEM_CF = 1;   // readers -> owners: the kp output landed (all readers)
constexpr uint32_t SEM_QKV = 2;  // readers -> owners: the q|k|v output landed
constexpr uint32_t SEM_AR = 3;   // lanes -> owners: an even all-reduce's result (2)
constexpr uint32_t SEM_AR2 = 4;  // lanes -> owners: an odd all-reduce's result (2)
constexpr uint32_t SEM_UG = 5;   // readers -> owners: the up|gate output landed
constexpr uint32_t SEM_R = 6;    // owners -> readers: a matmul input landed (64)

// compile-time args (every owner kernel): the geometry, then per phase (TYPE, FLAG) at CT_PH0 + 2 P
constexpr uint32_t CT_NPH = 0, CT_NRD = 1, CT_NOWN = 2, CT_NQH = 3, CT_NKVH = 4, CT_HT = 5, CT_GPT = 6, CT_NG = 7,
                   CT_IT = 8, CT_NTS = 9, CT_EPS = 10, CT_INVH = 11, CT_INVHD = 12, CT_NBLK = 13, CT_FABV = 14,
                   CT_FABCH = 15, CT_PH0 = 16;
// FLAG: CONV = side; FIN = the all-reduce ordinal (0: SEM_AR, 1: SEM_AR2) | side << 4; RMS_AR = the ordinal;
// HEADS = i | (last << 8): the i-th q|k|v hand-off of the program (the wait is cumulative: NRD x (i + 1)), reset
// after the last
// runtime args, dm0 (BRISC): 0 k, 1 c0, 2 E, 3 J, 4 SH, 5..8 lane 0 / 1 NoC (x, y), 9 the lanes' result address,
//   10 .. 10 + 2 NRD the readers' NoC (x, y), then the phase blocks of RT_W words at RT_PB0 = 10 + 2 NRD
// runtime args, dm1 (NCRISC): 0 k, 1 c0, 2 the owner table address (DRAM [64] x (x, y) uint32), phase blocks at 3
// runtime args, compute: 0 k
constexpr uint32_t RT_W = 24;
// phase block fields: RMS / RMS_AR: 0 src, 1 w, 2 out; CONV: 0 x, 1 cf, 2 out, 3 base4; FIN: 0 res, 1 cf, 3 base4;
// HEADS: 0 qkv, 1 q_norm w, 2 k_norm w, 3 cos, 4 sin, 5 pos, 6 page table, 7 page-table row bytes, 8 + g q' of kv
// group g, 8 + NKVH + g the K cache of kv head g, 8 + 2 NKVH + g the V cache; SWI: 0 up|gate, 1 m
constexpr uint32_t PB_Q = 8;
// taps (GLM53_DF2_TAPS=1, the per-phase device taps of tt/dflash2_mk.py; 0 = off): a phase's output tiles are also
// written to a [1, 1, 32, W] bf16 tap tensor (pages = the same column tiles): RT_TAP = the tap of the phase's output
// (RMS / CONV / FIN / SWI: the BRISC), RT_TAP2 = FIN's assembled all-reduce rows (bf16 X tiles) / HEADS: the K head
// tiles after norm + RoPE (the NCRISC, pages g HT + j), RT_TAP3 = HEADS: the V head tiles
constexpr uint32_t RT_TAP = 20, RT_TAP2 = 21, RT_TAP3 = 22;

// element offset of tile element (r, c) in the face-major layout of a 32 x 32 tile
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
// byte offset of row r's left face row (16 bf16) in a bf16 tile (the right face row is + 512)
constexpr uint32_t row_off(uint32_t r) { return (r >> 4) * 1024u + (r & 15u) * 32u; }

}  // namespace df2mk
