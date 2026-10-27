// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_bw: compile-time geometry and tile-address math shared by the reader,
// writer and compute kernels.  The compile-time argument block (indices 0..CT_ACC_BASE-1) is
// identical across the three binaries; the dataflow kernels append their TensorAccessorArgs after
// it.  Every value here is solved on the host (chunk_gated_delta_rule_bw_program_factory.cpp) and
// the two sides must agree exactly: a CB pushed with a block size the host did not size it for
// wraps its fifo, which is a hang rather than an error.
//
// Layout: every tensor is token-major with the heads folded into the last dim, so a page is
// [32 tokens x 32 dims] of ONE head and every block moves as whole pages:
//   q, k        [B, 1, T, HK*K]      v, do   [B, 1, T, HV*V]      g, beta [B, 1, T, HV]
//   (rank-3 [B, T, X] pages identically; do may instead be head-major [B*HV, 1, T, V])
//   dq, dk      [G*B, 1, T, HK*K]    plane g holds value head hk*G + g; the host sums the G planes
//   dv          [B, 1, T, HV*V]      dg, dbeta [B, 1, T, HV]
//   h0, dht, dh0 [B, HV, K, V]
//   saved h [B*HV, NC, K, V], v_new [B*HV, 1, T, V], A [B*HV, 1, T, C] (f32): item wi's block
//   starts at tile wi * ST_S / ST_VNEW / ST_ATTN, the stride of the scratch block it replaces.
// The work item is a VALUE head h; its key head is h / G.

#pragma once

#include <cstdint>

constexpr uint32_t gB = get_compile_time_arg_val(0);
constexpr uint32_t gT = get_compile_time_arg_val(1);
constexpr uint32_t gH = get_compile_time_arg_val(2);  // value-head count HV
constexpr uint32_t gK = get_compile_time_arg_val(3);
constexpr uint32_t gV = get_compile_time_arg_val(4);
constexpr uint32_t Ct = get_compile_time_arg_val(5);  // tiles per chunk
constexpr uint32_t Kt = get_compile_time_arg_val(6);
constexpr uint32_t Vt = get_compile_time_arg_val(7);
constexpr uint32_t Vb = get_compile_time_arg_val(8);   // V tiles per compute pass
constexpr uint32_t NVB = get_compile_time_arg_val(9);  // Vt / Vb
constexpr uint32_t NC = get_compile_time_arg_val(10);  // chunks per sequence
constexpr uint32_t BH = get_compile_time_arg_val(11);
constexpr uint32_t NEUMANN_STEPS = get_compile_time_arg_val(12);
constexpr uint32_t ESZ = get_compile_time_arg_val(13);  // input element bytes
constexpr uint32_t DEST_LIMIT = get_compile_time_arg_val(14);
constexpr uint32_t HAS_H0 = get_compile_time_arg_val(15);
constexpr uint32_t HAS_DHT = get_compile_time_arg_val(16);
constexpr uint32_t Tt = get_compile_time_arg_val(17);

// Tile-index bases of the per-item blocks inside the f32 scratch tensor.
constexpr uint32_t SC_ATTN = get_compile_time_arg_val(18);
constexpr uint32_t SC_KCD = get_compile_time_arg_val(19);
constexpr uint32_t SC_P = get_compile_time_arg_val(20);
constexpr uint32_t SC_VEC = get_compile_time_arg_val(21);
constexpr uint32_t SC_VCORR = get_compile_time_arg_val(22);
constexpr uint32_t SC_U = get_compile_time_arg_val(23);
constexpr uint32_t SC_C = get_compile_time_arg_val(24);
constexpr uint32_t SC_S = get_compile_time_arg_val(25);
constexpr uint32_t SC_DS = get_compile_time_arg_val(26);
constexpr uint32_t SC_VNEW = get_compile_time_arg_val(27);
constexpr uint32_t SC_DVNEW = get_compile_time_arg_val(28);
constexpr uint32_t SEM_PREP = get_compile_time_arg_val(29);
constexpr uint32_t SEM_SCAN = get_compile_time_arg_val(30);

// Uniform CB block sizes (one push size per CB) and named-slot counts.
constexpr uint32_t NCONST = get_compile_time_arg_val(31);    // cb_const pages
constexpr uint32_t NCOL = get_compile_time_arg_val(32);      // cb_colones pages
constexpr uint32_t MAXV = get_compile_time_arg_val(33);      // cb_vin block
constexpr uint32_t LVB = get_compile_time_arg_val(34);       // cb_load_vb block
constexpr uint32_t LITEM = get_compile_time_arg_val(35);     // cb_load_item block
constexpr uint32_t MAXBLK = get_compile_time_arg_val(36);    // cb_egr block
constexpr uint32_t MAXBLK_G = get_compile_time_arg_val(37);  // cb_gegr block
constexpr uint32_t NUM_CONST_MASKS = get_compile_time_arg_val(38);
constexpr uint32_t VECA_BLOCKS = get_compile_time_arg_val(39);
constexpr uint32_t VECB_BLOCKS = get_compile_time_arg_val(40);
constexpr uint32_t gHK = get_compile_time_arg_val(41);  // key-head count
constexpr uint32_t gG = get_compile_time_arg_val(42);   // value heads per key head
// The forward's h / v_new / A are inputs: Tinv, S_i and v_new are read, not rebuilt, and the
// forward half of the stage-S scan does not run.
constexpr uint32_t HAS_SAVED = get_compile_time_arg_val(43);
constexpr uint32_t DO_HEAD_MAJOR = get_compile_time_arg_val(44);  // d_out is [B*HV, 1, T, V]

constexpr uint32_t CT_ACC_BASE = 45;

constexpr uint32_t CHUNK = Ct * 32;
constexpr uint32_t CtCt = Ct * Ct;
constexpr uint32_t CtKt = Ct * Kt;
constexpr uint32_t CtVb = Ct * Vb;
constexpr uint32_t KtVb = Kt * Vb;
constexpr uint32_t CtVt = Ct * Vt;
constexpr uint32_t KtVt = Kt * Vt;

// Per-item scratch strides; must mirror ScratchMap on the host.
constexpr uint32_t ST_ATTN = CtCt;
constexpr uint32_t ST_KCD = CtKt;
constexpr uint32_t ST_P = CtKt;
constexpr uint32_t ST_VEC = 4 * Ct;  // decay, beta, dc1, w
constexpr uint32_t ST_VCORR = CtVt;
constexpr uint32_t ST_U = CtVt;
constexpr uint32_t ST_C = KtVt;
constexpr uint32_t ST_S = KtVt;
constexpr uint32_t ST_DS = KtVt;
constexpr uint32_t ST_VNEW = CtVt;
constexpr uint32_t ST_DVNEW = CtVt;

// Circular buffers; must mirror the kCb* constants on the host.
constexpr uint32_t cb_const = 0;
constexpr uint32_t cb_colones = 1;
constexpr uint32_t cb_stage = 2;  // dataflow-local staging for the g/beta column gather and scatter
constexpr uint32_t cb_qin = 3;
constexpr uint32_t cb_kin = 4;
constexpr uint32_t cb_vin = 5;
constexpr uint32_t cb_doin = 6;
constexpr uint32_t cb_gatein = 7;
constexpr uint32_t cb_load_item = 8;
constexpr uint32_t cb_load_vb = 9;
constexpr uint32_t cb_ka = 10;
constexpr uint32_t cb_kb = 11;
constexpr uint32_t cb_kc = 12;
constexpr uint32_t cb_kd = 13;
constexpr uint32_t cb_ke = 14;
constexpr uint32_t cb_kf = 15;
constexpr uint32_t cb_ktr = 16;
constexpr uint32_t cb_va = 17;
constexpr uint32_t cb_vb = 18;
constexpr uint32_t cb_vc = 19;
constexpr uint32_t cb_ca = 20;
constexpr uint32_t cb_cbL = 21;
constexpr uint32_t cb_cc = 22;
constexpr uint32_t cb_cd = 23;
constexpr uint32_t cb_ce = 24;
constexpr uint32_t cb_sa = 25;
constexpr uint32_t cb_sb = 26;
constexpr uint32_t cb_veca = 27;
constexpr uint32_t cb_vecb = 28;
constexpr uint32_t cb_vecc = 29;
constexpr uint32_t cb_egr = 30;
constexpr uint32_t cb_gegr = 31;

// cb_const block offsets (tiles): NUM_CONST_MASKS [C,C] mask blocks, then the scale, zero and
// row-of-ones tiles.
constexpr uint32_t CST_LT = 0;          // lower-triangular ones, diagonal included
constexpr uint32_t CST_NSTRICT = CtCt;  // -1 on the strictly-lower triangle
constexpr uint32_t CST_UT = 2 * CtCt;   // upper-triangular ones, diagonal included
constexpr uint32_t CST_EYE = 3 * CtCt;
constexpr uint32_t CST_BIAS = 4 * CtCt;  // 0 on tril-inclusive, -1e4 on strict upper
constexpr uint32_t CST_SUT = 5 * CtCt;   // strictly-upper ones (exclusive reverse cumsum)
static_assert(NUM_CONST_MASKS == 6, "cb_const mask layout above assumes six [C,C] mask blocks");
constexpr uint32_t CST_SCALE = NUM_CONST_MASKS * CtCt;        // column 0 == scale (stride-0 COL vector)
constexpr uint32_t CST_ZERO = NUM_CONST_MASKS * CtCt + 1;     // all zeros
constexpr uint32_t CST_ROWONES = NUM_CONST_MASKS * CtCt + 2;  // Ct tiles, row 0 == 1 (outer-product operand)

// cb_veca: the per-item decay column list.  Exactly VECA_BLOCKS single-block pushes per item, all
// popped at the end, so the fifo cycles back to its base and no block straddles the wrap point.
constexpr uint32_t VA_DECAY = 0;
constexpr uint32_t VA_GAMMA = Ct;
constexpr uint32_t VA_W = 2 * Ct;
constexpr uint32_t VA_BETA = 3 * Ct;
constexpr uint32_t VA_DC1 = 4 * Ct;  // tile VA_DC1 has [0,0] == log(Gamma)
static_assert(VECA_BLOCKS == 5, "cb_veca slot layout above assumes five column slots");

// cb_vecb: the stage-G column accumulators, same wrap rule as cb_veca.
constexpr uint32_t VB_DBETA_V = 0;   // rowsum_V(dv_beta . v), summed over V blocks
constexpr uint32_t VB_S1 = Ct;       // rowsum(ndU . U)
constexpr uint32_t VB_DGG = 2 * Ct;  // d(gamma) * gamma
constexpr uint32_t VB_RMC = 3 * Ct;  // rowsum(R - R^T)
constexpr uint32_t VB_DWW = 4 * Ct;  // d(w) * w
constexpr uint32_t VB_DBETA = 5 * Ct;
constexpr uint32_t VB_GAMMA = 6 * Ct;  // Gamma tile
constexpr uint32_t VB_DGGAM = 7 * Ct;  // LT_ones @ (dGamma * Gamma), as a column
constexpr uint32_t VB_UTE = 8 * Ct;    // UT_ones @ (rowsum(R - R^T) + dgg)
static_assert(VECB_BLOCKS == 9, "cb_vecb slot layout above assumes nine column slots");

// Element offset of (row r, col c) inside a 32x32 tile stored as four 16x16 faces in
// (face_r, face_c) row-major order.
FORCE_INLINE uint32_t tile_elem_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + ((r & 15u) << 4) + (c & 15u);
}

// Page index of tile (token-tile tt, head h, d-tile j) of a token-major [N,1,T,Hx*D] grid whose
// tile row holds hx_dt tiles (Hx * D/32).  `n` is the outermost index (batch, or plane*B + batch).
FORCE_INLINE uint32_t flat_page(uint32_t n, uint32_t tt, uint32_t hx_dt, uint32_t h, uint32_t dt, uint32_t j) {
    return (n * Tt + tt) * hx_dt + h * dt + j;
}
