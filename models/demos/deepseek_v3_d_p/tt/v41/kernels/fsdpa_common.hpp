// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The decode step's flash-decode SDPA (tt/v41/flash_sdpa.py): the compile-time argument layout,
// the CB ids and the work assignment shared by fsdpa_reader.cpp, fsdpa_writer.cpp and fsdpa_compute.cpp. The Python
// side (flash_sdpa.py's ``common`` list and ``CB`` dict) is the mirror; keep the two in step.

#pragma once

#include <stdint.h>

namespace fsdpa {

namespace ct {
enum : uint32_t {
    H = 0,             // query heads (one tile: 32)
    S,                 // tile rows (users)
    W,                 // index row width (keys a row at most)
    D,                 // head dim (K width)
    V,                 // V width (the output width)
    KPU,               // keys a unit
    UMAX,              // units a row at most (ceil(W / KPU))
    N,                 // cores
    SCALED_KV,         // 1: the packed scaled-FP8 rows; 0: bf16 rows
    PACKED_ROW_BYTES,  // scaled: the packed row's physical (page-aligned) stride
    Q_ELEM_BYTES,
    KV_ELEM_BYTES,
    SCALE,  // the softmax scale, fp32 bits
    USE_SINK,
    MATH_APPROX,
    NSEM,          // semaphores a core (merge units a core at most); the partial CBs hold NSEM * UMAX pages
    VT,            // output column tiles (V / 32)
    SCALE_BLOCKS,  // scaled: V / 128
    KV_IS_FP8,
    SLOT_TILES,     // tiles a partial slot: 2 + VT
    K_ROW_BYTES,    // the KV row's logical bytes
    LIVE_SENTINEL,  // the live page's word of a row that is not live
    HQ,             // the query heads read (the rest of the tile's heads are zero: not read, zeroed in L1)
    DEBUG_SCRATCH,  // 1: the writer also copies every partial to the DRAM scratch (the test's diagnostic)
    END,
};
}  // namespace ct

constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
constexpr uint16_t NEG_INF_BF16 = 0xFF80u;

// CB ids (flash_sdpa.py's CB dict)
enum CB : uint32_t {
    cb_q_rm = 0,
    cb_q_in,
    cb_k_rm,
    cb_k_in,
    cb_neginf,
    cb_mask_part,
    cb_scale,
    cb_qk_im,
    cb_max,
    cb_sum,
    cb_out,
    cb_ctrl_c,
    cb_ctrl_w,
    cb_col_identity,
    cb_k_rope_rm,
    cb_k_scale_bcast,
    cb_k_latent_tile,
    cb_k_rope_tile,
    cb_sink,
    cb_pm,
    cb_ps,
    cb_pa,
    cb_mM,
    cb_mw,
    cb_mws,
    cb_mS,
    cb_mrecip,
    cb_mA,
    cb_max_w,  // the unit's max tile for the writer (cb_max is the compute's own copy)
    cb_o_rm,
    cb_rscratch,
    cb_wscratch,
};

// The plan page the reader hands the compute and the writer: word 0 the live row count, words 16.. the live rows.
constexpr uint32_t PLAN_N_LIVE = 0;
constexpr uint32_t PLAN_LIVE = 16;

// Unit u of the live rows' UMAX units each runs on core u % N; merge unit mu (VT a live row) on core
// N - 1 - mu % N, its semaphore and partial slot mu / N. A unit's partial (max, sum, acc_v) goes straight into the
// merge core's L1: page (slot * UMAX + j) of its cb_pm / cb_ps / cb_pa (sdpa_decode's reducer exchange: no DRAM
// round trip, and the writer's barrier before the semaphore increment orders the data on the one NoC).
inline uint32_t merge_core(uint32_t mu, uint32_t n_cores) { return n_cores - 1 - (mu % n_cores); }
inline uint32_t merge_sem(uint32_t mu, uint32_t n_cores) { return mu / n_cores; }

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace fsdpa
