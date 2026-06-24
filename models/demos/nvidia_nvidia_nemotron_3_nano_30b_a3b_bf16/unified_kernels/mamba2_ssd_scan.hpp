// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mamba2_ssd_scan.hpp — Fused Mamba2 SSD chunked-scan for NemotronH-30B prefill.
//
// Replaces the Python chunk loop in mamba2_prefill.py:
//   for c in range(n_chunks):  # 4096 iterations at ISL=256K
//       y_c, h_prev = _mamba2_ssd_chunk(...)
//
// Grid: 8×8 = 64 cores, one core per SSM head.
// Each core processes all n_chunks sequentially (sequential state carry).
//
// Inputs must be pre-transposed to head-first layout (Python side):
//   x_dt : [H=64, n_chunks, C=64, D=64]
//   B    : [N_GROUPS=8, n_chunks, C=64, N=128]
//   C    : [N_GROUPS=8, n_chunks, C=64, N=128]
//   x    : [H=64, n_chunks, C=64, D=64]
//   logd : [H=64, n_chunks*C]                 (log-decay, flattened)
//   h_in : [H=64, D=64, N=128]
//   D_skip:[H=64, 1]

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/reg_api.h"
#include "api/compute/transpose_wh_dest.h"
#endif

namespace nemotron30b_ops {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr uint32_t NUM_HEADS = 64;
static constexpr uint32_t HEAD_DIM = 64;         // D
static constexpr uint32_t SSM_STATE_SIZE = 128;  // N
static constexpr uint32_t N_GROUPS = 8;
static constexpr uint32_t HEADS_PER_GROUP = 8;
static constexpr uint32_t CHUNK_SIZE = 64;  // C

static constexpr uint32_t TILE = 32;
static constexpr uint32_t XDT_TILES_R = CHUNK_SIZE / TILE;                        // 2
static constexpr uint32_t XDT_TILES_C = HEAD_DIM / TILE;                          // 2
static constexpr uint32_t XDT_TILES = XDT_TILES_R * XDT_TILES_C;                  // 4
static constexpr uint32_t B_TILES_R = CHUNK_SIZE / TILE;                          // 2
static constexpr uint32_t B_TILES_C = SSM_STATE_SIZE / TILE;                      // 4
static constexpr uint32_t B_TILES = B_TILES_R * B_TILES_C;                        // 8
static constexpr uint32_t H_TILES = (HEAD_DIM * SSM_STATE_SIZE) / (TILE * TILE);  // 8
static constexpr uint32_t L_TILES = (CHUNK_SIZE * CHUNK_SIZE) / (TILE * TILE);    // 4
static constexpr uint32_t GAMMA_TILES = CHUNK_SIZE / TILE;                        // 2

// Face sub-tile geometry (for NCRISC logd extraction)
static constexpr uint32_t HALF = 16;           // face height/width
static constexpr uint32_t FACE = HALF * HALF;  // 256 elements per face

// BF16 -inf sentinel for log-L upper triangle
static constexpr uint16_t NEG_INF_BF16 = 0xFF80u;

// CB IDs
static constexpr uint32_t CB_X_DT = 0;         // x_dt [C,D], double-buffered
static constexpr uint32_t CB_B = 1;            // B [C,N], double-buffered
static constexpr uint32_t CB_C = 2;            // C [C,N], double-buffered
static constexpr uint32_t CB_X = 3;            // x [C,D], double-buffered
static constexpr uint32_t CB_LOGL = 4;         // log_L [C,C] = 4 tiles, double-buffered
static constexpr uint32_t CB_H = 5;            // h_state [D,N] = 8 tiles, double-buffered
static constexpr uint32_t CB_Y = 6;            // y output [C,D]
static constexpr uint32_t CB_QK = 7;           // general scratch 4 tiles
static constexpr uint32_t CB_HOUT = 8;         // final h_out
static constexpr uint32_t CB_DSKIP = 9;        // D_skip scalar (1 tile)
static constexpr uint32_t CB_LOGGAMMA = 10;    // log_gamma [C] col-0 = 2 tiles, double-buf
static constexpr uint32_t CB_LOGDELTA = 11;    // log_delta [C] col-0 = 2 tiles, double-buf
static constexpr uint32_t CB_LOGGSCALAR = 12;  // log(gamma_last) scalar = 1 tile, double-buf
static constexpr uint32_t CB_L_EXP = 13;       // L after exp [C,C] scratch
static constexpr uint32_t CB_YCROSS = 14;      // y_cross and x_dt_scaled^T scratch
static constexpr uint32_t CB_YINTRA = 15;      // y_intra scratch (not exposed to BRISC)

// ---------------------------------------------------------------------------
// Struct args
// ---------------------------------------------------------------------------
struct CTArgs {
    static constexpr uint32_t GRID_R = 8;
    static constexpr uint32_t GRID_C = 8;
};

struct ReaderArgs {
    uint32_t x_dt_addr;
    uint32_t B_addr;
    uint32_t C_addr;
    uint32_t x_addr;
    uint32_t logd_addr;
    uint32_t h_in_addr;
    uint32_t D_skip_addr;
    uint32_t n_chunks;
};

struct WriterArgs {
    uint32_t y_addr;
    uint32_t h_out_addr;
    uint32_t n_chunks;
};

struct ComputeArgs {
    uint32_t n_chunks;
};

using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

// ---------------------------------------------------------------------------
// NCRISC helpers (soft-FP; no hardware float on rv32im)
// ---------------------------------------------------------------------------
#if defined(COMPILE_FOR_NCRISC)

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    // Round-to-nearest-even
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

static float sw_expf(float x) {
    if (x >= 88.0f) {
        return 3.40282347e+38f;
    }
    if (x <= -88.0f) {
        return 0.0f;
    }
    const float LOG2E = 1.4426950408f;
    float z = x * LOG2E;
    int n = (int32_t)z;
    if (z < 0.0f && z != (float)n) {
        n--;
    }
    float frac = z - (float)n;
    float p = 1.0f + frac * (0.693147180f + frac * (0.240226507f + frac * 0.055715f));
    uint32_t bits;
    __builtin_memcpy(&bits, &p, 4);
    int32_t exp_field = (int32_t)((bits >> 23) & 0xFFu) + n;
    if (exp_field <= 0) {
        return 0.0f;
    }
    if (exp_field >= 255) {
        return 3.40282347e+38f;
    }
    bits = (bits & 0x807FFFFFu) | ((uint32_t)exp_field << 23);
    __builtin_memcpy(&p, &bits, 4);
    return p;
}

#endif  // COMPILE_FOR_NCRISC

// ---------------------------------------------------------------------------
// Main Op
// ---------------------------------------------------------------------------
template <typename CTArgs, bool IsActiveCore>
class Op {
public:
    void operator()(const RTArgs& args) {
        if constexpr (IsActiveCore) {
            impl(args);
        }
        // Idle cores do nothing
    }

private:
    void impl([[maybe_unused]] const RTArgs& args) {
// ==========================================================================
// NCRISC — Reader
// ==========================================================================
#if defined(COMPILE_FOR_NCRISC)
        {
            const uint32_t n_chunks = args.n_chunks;

            const uint32_t core_x = my_x[0];
            const uint32_t core_y = my_y[0];
            const uint32_t head_h = core_y * CTArgs::GRID_C + core_x;
            if (head_h >= NUM_HEADS) {
                return;
            }

            const uint32_t group_g = head_h / HEADS_PER_GROUP;

            // Head's row within its tile (for logd extraction)
            const uint32_t r_in_tile = head_h & (TILE - 1);  // head_h % 32
            const uint32_t rf = r_in_tile >> 4;              // face row (0 or 1)
            const uint32_t ri = r_in_tile & 15;              // row within face (0..15)
            // Tile row in logd tensor for this head
            const uint32_t logd_tile_row = head_h / TILE;  // 0 or 1

            // Chunk strides (bytes)
            const uint32_t xdt_chunk_bytes = CHUNK_SIZE * HEAD_DIM * sizeof(uint16_t);      // 8192
            const uint32_t b_chunk_bytes = CHUNK_SIZE * SSM_STATE_SIZE * sizeof(uint16_t);  // 16384
            // logd tile layout: [H/TILE, n_chunks*2] tiles
            // Each tile is TILE*TILE*2 = 2048 bytes
            // A chunk occupies 2 consecutive tile columns: 2*2048 = 4096 bytes
            const uint32_t logd_chunk_bytes = 2u * TILE * TILE * sizeof(uint16_t);  // 4096

            // Base L1 byte offsets for this head/group in DRAM
            const uint32_t xdt_base = args.x_dt_addr + head_h * n_chunks * xdt_chunk_bytes;
            const uint32_t B_base = args.B_addr + group_g * n_chunks * b_chunk_bytes;
            const uint32_t C_base = args.C_addr + group_g * n_chunks * b_chunk_bytes;
            const uint32_t x_base = args.x_addr + head_h * n_chunks * xdt_chunk_bytes;
            // logd: tile_row = logd_tile_row, tile_col for chunk c = c*2
            // Byte offset for tile (logd_tile_row, 0): logd_tile_row * n_chunks*2 * 2048
            const uint32_t logd_row_base = args.logd_addr + logd_tile_row * n_chunks * logd_chunk_bytes;

            // ---- One-time: D_skip ----
            // D_skip is [H, 1] tiled as 2 tiles of [TILE, TILE]; tile logd_tile_row has D_skip[0..31] or [32..63]
            // Read the tile, extract scalar D_skip[head_h], put it in CB_DSKIP[0,0]
            {
                cb_reserve_back(CB_DSKIP, 1);
                uint32_t dskip_l1 = get_write_ptr(CB_DSKIP);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(
                        0, args.D_skip_addr + logd_tile_row * TILE * TILE * sizeof(uint16_t)),
                    dskip_l1,
                    TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();

                uint16_t* tile = reinterpret_cast<uint16_t*>(dskip_l1);
                // Extract D_skip[head_h]: face rf*2 (left half), row ri, col 0
                uint16_t val = tile[rf * 2u * FACE + ri * HALF + 0u];
                // Rebuild tile: zero all, place scalar at face0[0] for SCALAR bcast
                for (uint32_t i = 0; i < TILE * TILE; i++) {
                    tile[i] = 0;
                }
                tile[0] = val;  // face0, row0, col0 = [0,0]
                cb_push_back(CB_DSKIP, 1);
            }

            // ---- One-time: h_in ----
            {
                cb_reserve_back(CB_H, H_TILES);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(
                        0, args.h_in_addr + head_h * H_TILES * TILE * TILE * sizeof(uint16_t)),
                    get_write_ptr(CB_H),
                    H_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();
                cb_push_back(CB_H, H_TILES);
            }

            // ---- Per-chunk loop ----
            for (uint32_t c = 0; c < n_chunks; ++c) {
                const uint32_t off_xdt = c * xdt_chunk_bytes;
                const uint32_t off_b = c * b_chunk_bytes;
                const uint32_t off_logd = c * logd_chunk_bytes;

                // x_dt_c: [C, D] = XDT_TILES tiles
                cb_reserve_back(CB_X_DT, XDT_TILES);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(0, xdt_base + off_xdt),
                    get_write_ptr(CB_X_DT),
                    XDT_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();
                cb_push_back(CB_X_DT, XDT_TILES);

                // B_c: [C, N] = B_TILES tiles
                cb_reserve_back(CB_B, B_TILES);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(0, B_base + off_b),
                    get_write_ptr(CB_B),
                    B_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();
                cb_push_back(CB_B, B_TILES);

                // C_c: [C, N] = B_TILES tiles
                cb_reserve_back(CB_C, B_TILES);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(0, C_base + off_b),
                    get_write_ptr(CB_C),
                    B_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();
                cb_push_back(CB_C, B_TILES);

                // x_c: [C, D] = XDT_TILES tiles
                cb_reserve_back(CB_X, XDT_TILES);
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(0, x_base + off_xdt),
                    get_write_ptr(CB_X),
                    XDT_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();
                cb_push_back(CB_X, XDT_TILES);

                // ---- logd processing: build log_L, log_gamma, log_delta, log_gscalar ----
                // Use CB_LOGL write area as scratch: read 2 raw logd tiles, then overwrite with log_L

                cb_reserve_back(CB_LOGL, L_TILES);
                uint32_t logl_l1 = get_write_ptr(CB_LOGL);

                // Read the 2 logd tiles for this chunk into the first 2 tile slots of CB_LOGL
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(0, logd_row_base + off_logd),
                    logl_l1,
                    2u * TILE * TILE * sizeof(uint16_t));
                noc_async_read_barrier();

                // Extract head_h's cumulative sum of log_decay from the 2 raw tiles
                // Tile layout: face 0 = rows 0..15, cols 0..15; face 1 = rows 0..15, cols 16..31
                //              face 2 = rows 16..31, cols 0..15; face 3 = rows 16..31, cols 16..31
                const uint16_t* tile0 = reinterpret_cast<const uint16_t*>(logl_l1);
                const uint16_t* tile1 = tile0 + TILE * TILE;

                float cum[CHUNK_SIZE];
                float acc = 0.0f;
                // tile0, face rf*2 (left half of rows rf*16..(rf+1)*16-1), row ri, cols 0..15
                for (uint32_t i = 0; i < HALF; i++) {
                    acc += bf16_to_fp32(tile0[rf * 2u * FACE + ri * HALF + i]);
                    cum[i] = acc;
                }
                // tile0, face rf*2+1 (right half), row ri, cols 16..31 of chunk
                for (uint32_t i = 0; i < HALF; i++) {
                    acc += bf16_to_fp32(tile0[(rf * 2u + 1u) * FACE + ri * HALF + i]);
                    cum[HALF + i] = acc;
                }
                // tile1, face rf*2, row ri, cols 32..47 of chunk
                for (uint32_t i = 0; i < HALF; i++) {
                    acc += bf16_to_fp32(tile1[rf * 2u * FACE + ri * HALF + i]);
                    cum[2u * HALF + i] = acc;
                }
                // tile1, face rf*2+1, row ri, cols 48..63 of chunk
                for (uint32_t i = 0; i < HALF; i++) {
                    acc += bf16_to_fp32(tile1[(rf * 2u + 1u) * FACE + ri * HALF + i]);
                    cum[3u * HALF + i] = acc;
                }

                // Build log_L [C,C] = 4 tiles in CB_LOGL
                // log_L[i,s] = min(0, cum[i]-cum[s]) for s<=i; NEG_INF_BF16 for s>i
                uint16_t* logl_w = reinterpret_cast<uint16_t*>(logl_l1);
                for (uint32_t rt = 0; rt < 2u; rt++) {
                    for (uint32_t ct = 0; ct < 2u; ct++) {
                        uint16_t* tile = logl_w + (rt * 2u + ct) * TILE * TILE;
                        for (uint32_t rf2 = 0; rf2 < 2u; rf2++) {
                            for (uint32_t cf2 = 0; cf2 < 2u; cf2++) {
                                uint16_t* face = tile + (rf2 * 2u + cf2) * FACE;
                                for (uint32_t ri2 = 0; ri2 < HALF; ri2++) {
                                    const uint32_t row_i = rt * TILE + rf2 * HALF + ri2;
                                    for (uint32_t ci2 = 0; ci2 < HALF; ci2++) {
                                        const uint32_t col_s = ct * TILE + cf2 * HALF + ci2;
                                        if (col_s > row_i) {
                                            face[ri2 * HALF + ci2] = NEG_INF_BF16;
                                        } else {
                                            float logL = cum[row_i] - cum[col_s];
                                            if (logL > 0.0f) {
                                                logL = 0.0f;
                                            }
                                            face[ri2 * HALF + ci2] = fp32_to_bf16(logL);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                cb_push_back(CB_LOGL, L_TILES);

                // Build log_gamma: 2 tiles, each with gamma[rt*32..] in col-0 for col-bcast mul
                // gamma[i] = exp(cum[i]); stored as BF16 in face col-0 for BroadcastType::COL
                cb_reserve_back(CB_LOGGAMMA, GAMMA_TILES);
                {
                    uint16_t* g_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(CB_LOGGAMMA));
                    for (uint32_t t = 0; t < GAMMA_TILES; t++) {
                        uint16_t* gtile = g_ptr + t * TILE * TILE;
                        for (uint32_t i = 0; i < TILE * TILE; i++) {
                            gtile[i] = 0;
                        }
                        // Row r of tile t → face rf2, row ri2 within face
                        for (uint32_t r = 0; r < TILE; r++) {
                            uint32_t rf2 = r >> 4;
                            uint32_t ri2 = r & 15u;
                            // Col 0 lives in face rf2*2 (left half), position ri2*16+0
                            gtile[(rf2 * 2u) * FACE + ri2 * HALF + 0u] = fp32_to_bf16(cum[t * TILE + r]);
                        }
                    }
                }
                cb_push_back(CB_LOGGAMMA, GAMMA_TILES);

                // Build log_delta: 2 tiles, delta[s] = exp(cum[C-1] - cum[s])
                cb_reserve_back(CB_LOGDELTA, GAMMA_TILES);
                {
                    uint16_t* d_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(CB_LOGDELTA));
                    const float cum_last = cum[CHUNK_SIZE - 1u];
                    for (uint32_t t = 0; t < GAMMA_TILES; t++) {
                        uint16_t* dtile = d_ptr + t * TILE * TILE;
                        for (uint32_t i = 0; i < TILE * TILE; i++) {
                            dtile[i] = 0;
                        }
                        for (uint32_t r = 0; r < TILE; r++) {
                            uint32_t rf2 = r >> 4;
                            uint32_t ri2 = r & 15u;
                            dtile[(rf2 * 2u) * FACE + ri2 * HALF + 0u] = fp32_to_bf16(cum_last - cum[t * TILE + r]);
                        }
                    }
                }
                cb_push_back(CB_LOGDELTA, GAMMA_TILES);

                // Build log_gscalar: 1 tile, gamma_last = exp(cum[C-1]) at [0,0]
                cb_reserve_back(CB_LOGGSCALAR, 1);
                {
                    uint16_t* s_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(CB_LOGGSCALAR));
                    for (uint32_t i = 0; i < TILE * TILE; i++) {
                        s_ptr[i] = 0;
                    }
                    s_ptr[0] = fp32_to_bf16(cum[CHUNK_SIZE - 1u]);
                }
                cb_push_back(CB_LOGGSCALAR, 1);
            }
        }

// ==========================================================================
// BRISC — Writer
// ==========================================================================
#elif defined(COMPILE_FOR_BRISC)
        {
            const uint32_t n_chunks = args.n_chunks;
            const uint32_t core_x = my_x[0];
            const uint32_t core_y = my_y[0];
            const uint32_t head_h = core_y * CTArgs::GRID_C + core_x;
            if (head_h >= NUM_HEADS) {
                return;
            }

            const uint32_t y_chunk_bytes = CHUNK_SIZE * HEAD_DIM * sizeof(uint16_t);
            const uint32_t y_base = args.y_addr + head_h * n_chunks * y_chunk_bytes;

            for (uint32_t c = 0; c < n_chunks; ++c) {
                cb_wait_front(CB_Y, XDT_TILES);
                noc_async_write(
                    get_read_ptr(CB_Y),
                    get_noc_addr_from_bank_id<true>(0, y_base + c * y_chunk_bytes),
                    XDT_TILES * TILE * TILE * sizeof(uint16_t));
                noc_async_write_barrier();
                cb_pop_front(CB_Y, XDT_TILES);
            }

            cb_wait_front(CB_HOUT, H_TILES);
            noc_async_write(
                get_read_ptr(CB_HOUT),
                get_noc_addr_from_bank_id<true>(0, args.h_out_addr + head_h * H_TILES * TILE * TILE * sizeof(uint16_t)),
                H_TILES * TILE * TILE * sizeof(uint16_t));
            noc_async_write_barrier();
            cb_pop_front(CB_HOUT, H_TILES);
        }

// ==========================================================================
// TRISC — Compute
// ==========================================================================
#elif defined(COMPILE_FOR_TRISC)
        {
            const uint32_t n_chunks = args.n_chunks;

            // Wait for one-time inputs
            cb_wait_front(CB_H, H_TILES);  // h_in from NCRISC
            cb_wait_front(CB_DSKIP, 1);    // D_skip scalar from NCRISC

            // ---- Per-chunk loop ----
            for (uint32_t c = 0; c < n_chunks; ++c) {
                // Wait for all per-chunk CBs from NCRISC
                cb_wait_front(CB_LOGL, L_TILES);
                cb_wait_front(CB_X_DT, XDT_TILES);
                cb_wait_front(CB_B, B_TILES);
                cb_wait_front(CB_C, B_TILES);
                cb_wait_front(CB_X, XDT_TILES);
                cb_wait_front(CB_LOGGAMMA, GAMMA_TILES);
                cb_wait_front(CB_LOGDELTA, GAMMA_TILES);
                cb_wait_front(CB_LOGGSCALAR, 1);

                // --------------------------------------------------------
                // Step A: L = exp(log_L) → CB_L_EXP
                // --------------------------------------------------------
                cb_reserve_back(CB_L_EXP, L_TILES);
                copy_tile_to_dst_init_short(CB_LOGL);
                exp_tile_init();
                tile_regs_acquire();
                for (uint32_t t = 0; t < L_TILES; ++t) {
                    copy_tile(CB_LOGL, t, t);
                    exp_tile(t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < L_TILES; ++t) {
                    pack_tile(t, CB_L_EXP);
                }
                tile_regs_release();
                cb_pop_front(CB_LOGL, L_TILES);
                cb_push_back(CB_L_EXP, L_TILES);

                // --------------------------------------------------------
                // Step B: Q_K = C @ B^T  [C,N] @ [N,C] → [C,C]
                // --------------------------------------------------------
                cb_reserve_back(CB_QK, L_TILES);
                mm_init(CB_C, CB_B, CB_QK, 1 /*transpose B*/);
                tile_regs_acquire();
                for (uint32_t it = 0; it < XDT_TILES_R; ++it) {
                    for (uint32_t jt = 0; jt < XDT_TILES_R; ++jt) {
                        for (uint32_t k = 0; k < B_TILES_C; ++k) {
                            matmul_tiles(CB_C, CB_B, it * B_TILES_C + k, jt * B_TILES_C + k, it * XDT_TILES_R + jt);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < L_TILES; ++t) {
                    pack_tile(t, CB_QK);
                }
                tile_regs_release();
                cb_push_back(CB_QK, L_TILES);

                // --------------------------------------------------------
                // Step C: L_QK = L ⊙ Q_K → CB_L_EXP (reuse after pop)
                // --------------------------------------------------------
                cb_wait_front(CB_L_EXP, L_TILES);
                cb_wait_front(CB_QK, L_TILES);
                cb_reserve_back(CB_YCROSS, L_TILES);  // temp output for L_QK
                mul_tiles_init(CB_L_EXP, CB_QK);
                tile_regs_acquire();
                for (uint32_t t = 0; t < L_TILES; ++t) {
                    mul_tiles(CB_L_EXP, CB_QK, t, t, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < L_TILES; ++t) {
                    pack_tile(t, CB_YCROSS);
                }
                tile_regs_release();
                cb_pop_front(CB_L_EXP, L_TILES);
                cb_pop_front(CB_QK, L_TILES);
                cb_push_back(CB_YCROSS, L_TILES);  // CB_YCROSS = L_QK

                // --------------------------------------------------------
                // Step D: y_intra = L_QK @ x_dt  [C,C] @ [C,D] → [C,D]
                // Stored in CB_YINTRA (not CB_Y) to prevent BRISC from consuming
                // y_intra before Step G-a reads it.  Only Step G-c pushes to CB_Y.
                // --------------------------------------------------------
                cb_reserve_back(CB_YINTRA, XDT_TILES);
                cb_wait_front(CB_YCROSS, L_TILES);
                mm_init(CB_YCROSS, CB_X_DT, CB_YINTRA, 0);
                tile_regs_acquire();
                for (uint32_t it = 0; it < XDT_TILES_R; ++it) {
                    for (uint32_t jt = 0; jt < XDT_TILES_C; ++jt) {
                        for (uint32_t k = 0; k < XDT_TILES_R; ++k) {
                            matmul_tiles(
                                CB_YCROSS, CB_X_DT, it * XDT_TILES_R + k, k * XDT_TILES_C + jt, it * XDT_TILES_C + jt);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_YINTRA);
                }
                tile_regs_release();
                cb_pop_front(CB_YCROSS, L_TILES);  // L_QK consumed
                // CB_X_DT NOT popped yet — needed for delta scaling in Step H
                cb_push_back(CB_YINTRA, XDT_TILES);

                // --------------------------------------------------------
                // Step E: y_cross = C @ h_prev^T  [C,N] @ [D,N]^T → [C,D]
                // --------------------------------------------------------
                cb_reserve_back(CB_L_EXP, XDT_TILES);
                mm_init(CB_C, CB_H, CB_L_EXP, 1 /*transpose h*/);
                tile_regs_acquire();
                for (uint32_t it = 0; it < XDT_TILES_R; ++it) {
                    for (uint32_t jt = 0; jt < XDT_TILES_C; ++jt) {
                        for (uint32_t k = 0; k < B_TILES_C; ++k) {
                            matmul_tiles(CB_C, CB_H, it * B_TILES_C + k, jt * B_TILES_C + k, it * XDT_TILES_C + jt);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_L_EXP);
                }
                tile_regs_release();
                cb_pop_front(CB_C, B_TILES);  // C consumed
                // CB_H NOT popped — still needed for Step H state update
                cb_push_back(CB_L_EXP, XDT_TILES);  // CB_L_EXP = y_cross (unscaled)

                // --------------------------------------------------------
                // Step F-a: gamma = exp(log_gamma) → CB_QK (2 tiles)
                // --------------------------------------------------------
                cb_reserve_back(CB_QK, GAMMA_TILES);
                copy_tile_to_dst_init_short(CB_LOGGAMMA);
                exp_tile_init();
                tile_regs_acquire();
                for (uint32_t t = 0; t < GAMMA_TILES; ++t) {
                    copy_tile(CB_LOGGAMMA, t, t);
                    exp_tile(t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < GAMMA_TILES; ++t) {
                    pack_tile(t, CB_QK);
                }
                tile_regs_release();
                cb_pop_front(CB_LOGGAMMA, GAMMA_TILES);
                cb_push_back(CB_QK, GAMMA_TILES);

                // --------------------------------------------------------
                // Step F-b: y_cross_scaled = y_cross * gamma (col-bcast)
                // gamma tile rt broadcasts to rows rt*32..(rt+1)*32-1 of y_cross
                // --------------------------------------------------------
                cb_wait_front(CB_L_EXP, XDT_TILES);  // y_cross
                cb_wait_front(CB_QK, GAMMA_TILES);   // gamma
                cb_reserve_back(CB_YCROSS, XDT_TILES);
                mul_bcast_cols_init_short(CB_L_EXP, CB_QK);
                tile_regs_acquire();
                for (uint32_t rt = 0; rt < XDT_TILES_R; ++rt) {
                    for (uint32_t ct = 0; ct < XDT_TILES_C; ++ct) {
                        mul_tiles_bcast<BroadcastType::COL>(
                            CB_L_EXP, CB_QK, rt * XDT_TILES_C + ct, rt, rt * XDT_TILES_C + ct);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_YCROSS);
                }
                tile_regs_release();
                cb_pop_front(CB_L_EXP, XDT_TILES);
                cb_pop_front(CB_QK, GAMMA_TILES);
                cb_push_back(CB_YCROSS, XDT_TILES);  // CB_YCROSS = y_cross_scaled

                // --------------------------------------------------------
                // Step G-a: y = y_intra + y_cross_scaled
                // --------------------------------------------------------
                cb_wait_front(CB_YINTRA, XDT_TILES);  // y_intra (from CB_YINTRA, not CB_Y)
                cb_wait_front(CB_YCROSS, XDT_TILES);  // y_cross_scaled
                cb_reserve_back(CB_L_EXP, XDT_TILES);
                add_tiles_init(CB_YINTRA, CB_YCROSS);
                tile_regs_acquire();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    add_tiles(CB_YINTRA, CB_YCROSS, t, t, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_L_EXP);
                }
                tile_regs_release();
                cb_pop_front(CB_YINTRA, XDT_TILES);
                cb_pop_front(CB_YCROSS, XDT_TILES);
                cb_push_back(CB_L_EXP, XDT_TILES);  // CB_L_EXP = y_intra + y_cross_scaled

                // --------------------------------------------------------
                // Step G-b: y += D_skip * x_c (scalar bcast)
                // --------------------------------------------------------
                cb_reserve_back(CB_QK, XDT_TILES);
                mul_tiles_bcast_scalar_init_short(CB_X, CB_DSKIP);
                tile_regs_acquire();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    mul_tiles_bcast<BroadcastType::SCALAR>(CB_X, CB_DSKIP, t, 0, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_QK);
                }
                tile_regs_release();
                cb_pop_front(CB_X, XDT_TILES);
                cb_push_back(CB_QK, XDT_TILES);  // CB_QK = D_skip * x_c

                // --------------------------------------------------------
                // Step G-c: y = (y_intra + y_cross_scaled) + D_skip*x → CB_Y for BRISC
                // --------------------------------------------------------
                cb_wait_front(CB_L_EXP, XDT_TILES);
                cb_wait_front(CB_QK, XDT_TILES);
                cb_reserve_back(CB_Y, XDT_TILES);
                add_tiles_init(CB_L_EXP, CB_QK);
                tile_regs_acquire();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    add_tiles(CB_L_EXP, CB_QK, t, t, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_Y);
                }
                tile_regs_release();
                cb_pop_front(CB_L_EXP, XDT_TILES);
                cb_pop_front(CB_QK, XDT_TILES);
                cb_push_back(CB_Y, XDT_TILES);  // BRISC will write this

                // --------------------------------------------------------
                // Step H-a: delta = exp(log_delta) → CB_QK (2 tiles)
                // --------------------------------------------------------
                cb_reserve_back(CB_QK, GAMMA_TILES);
                copy_tile_to_dst_init_short(CB_LOGDELTA);
                exp_tile_init();
                tile_regs_acquire();
                for (uint32_t t = 0; t < GAMMA_TILES; ++t) {
                    copy_tile(CB_LOGDELTA, t, t);
                    exp_tile(t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < GAMMA_TILES; ++t) {
                    pack_tile(t, CB_QK);
                }
                tile_regs_release();
                cb_pop_front(CB_LOGDELTA, GAMMA_TILES);
                cb_push_back(CB_QK, GAMMA_TILES);

                // --------------------------------------------------------
                // Step H-b: x_dt_scaled = x_dt * delta (col-bcast) → CB_L_EXP
                // --------------------------------------------------------
                cb_wait_front(CB_X_DT, XDT_TILES);  // x_dt (held since Step D)
                cb_wait_front(CB_QK, GAMMA_TILES);  // delta
                cb_reserve_back(CB_L_EXP, XDT_TILES);
                mul_bcast_cols_init_short(CB_X_DT, CB_QK);
                tile_regs_acquire();
                for (uint32_t rt = 0; rt < XDT_TILES_R; ++rt) {
                    for (uint32_t ct = 0; ct < XDT_TILES_C; ++ct) {
                        mul_tiles_bcast<BroadcastType::COL>(
                            CB_X_DT, CB_QK, rt * XDT_TILES_C + ct, rt, rt * XDT_TILES_C + ct);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_L_EXP);
                }
                tile_regs_release();
                cb_pop_front(CB_X_DT, XDT_TILES);   // x_dt fully consumed
                cb_pop_front(CB_QK, GAMMA_TILES);   // delta consumed
                cb_push_back(CB_L_EXP, XDT_TILES);  // CB_L_EXP = x_dt_scaled [C,D]

                // --------------------------------------------------------
                // Step H-c: Transpose x_dt_scaled [C,D] → [D,C] in CB_YCROSS
                // [C,D] source tiles: 0=(C0,D0), 1=(C0,D1), 2=(C1,D0), 3=(C1,D1)
                // [D,C] output order: 0=(D0,C0)=src0^T, 1=(D0,C1)=src2^T,
                //                     2=(D1,C0)=src1^T, 3=(D1,C1)=src3^T
                // Load into DST in output order so pack can run sequentially
                // (avoids non-sequential ifrom_dst which may stall the pack engine).
                // --------------------------------------------------------
                cb_wait_front(CB_L_EXP, XDT_TILES);
                cb_reserve_back(CB_YCROSS, XDT_TILES);
                copy_tile_to_dst_init_short(CB_L_EXP);
                transpose_wh_dest_init_short();
                tile_regs_acquire();
                copy_tile(CB_L_EXP, 0, 0);
                transpose_wh_dest(0);  // DST[0] = src0^T = (D0,C0)
                copy_tile(CB_L_EXP, 2, 1);
                transpose_wh_dest(1);  // DST[1] = src2^T = (D0,C1)
                copy_tile(CB_L_EXP, 1, 2);
                transpose_wh_dest(2);  // DST[2] = src1^T = (D1,C0)
                copy_tile(CB_L_EXP, 3, 3);
                transpose_wh_dest(3);  // DST[3] = src3^T = (D1,C1)
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < XDT_TILES; ++t) {
                    pack_tile(t, CB_YCROSS);  // sequential: slots 0,1,2,3
                }
                tile_regs_release();
                cb_pop_front(CB_L_EXP, XDT_TILES);
                cb_push_back(CB_YCROSS, XDT_TILES);  // CB_YCROSS = x_dt_scaled^T [D,C]

                // --------------------------------------------------------
                // Step H-d: gamma_last = exp(log_gscalar) → CB_QK (1 tile)
                // --------------------------------------------------------
                cb_reserve_back(CB_QK, 1);
                copy_tile_to_dst_init_short(CB_LOGGSCALAR);
                exp_tile_init();
                tile_regs_acquire();
                copy_tile(CB_LOGGSCALAR, 0, 0);
                exp_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_QK, 0);
                tile_regs_release();
                cb_pop_front(CB_LOGGSCALAR, 1);
                cb_push_back(CB_QK, 1);

                // --------------------------------------------------------
                // Step H-e: h_prev_scaled = gamma_last * h_prev (scalar-bcast)
                // Reads from CB_H front, writes to CB_H back (double-buffered)
                // --------------------------------------------------------
                cb_wait_front(CB_H, H_TILES);
                cb_wait_front(CB_QK, 1);
                cb_reserve_back(CB_H, H_TILES);
                mul_tiles_bcast_scalar_init_short(CB_H, CB_QK);
                tile_regs_acquire();
                for (uint32_t t = 0; t < H_TILES; ++t) {
                    mul_tiles_bcast<BroadcastType::SCALAR>(CB_H, CB_QK, t, 0, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < H_TILES; ++t) {
                    pack_tile(t, CB_H);
                }
                tile_regs_release();
                cb_pop_front(CB_H, H_TILES);  // consume old h_prev
                cb_pop_front(CB_QK, 1);       // consume gamma_last scalar
                cb_push_back(CB_H, H_TILES);  // push gamma_last * h_prev

                // --------------------------------------------------------
                // Step H-f: h_next = gamma_last*h_prev + x_dt_scaled^T @ B
                //
                // The packer zeros DST tiles after each pack (SyncFull default), so
                // we cannot rely on DST retaining gamma_last*h_prev from Step H-e.
                // Split into two acquire blocks:
                //   Part 1: x_new = x_dt_scaled^T @ B  (matmul into CB_HOUT scratch)
                //   Part 2: h_next = gamma_last*h_prev + x_new  (add → CB_H)
                //
                // CB_HOUT (capacity H_TILES=8) is used as scratch; BRISC only reads
                // it after the chunk loop (for the final h_out write), so reuse here
                // is safe.
                // --------------------------------------------------------

                // Part 1: x_new = x_dt_scaled^T @ B → CB_HOUT
                // CB_YCROSS = x_dt_scaled^T [D,C] = 2×2 tiles
                // CB_B = B [C,N] = 2×4 tiles; output [D,N] = 2×4 = H_TILES
                cb_wait_front(CB_YCROSS, XDT_TILES);
                cb_wait_front(CB_B, B_TILES);
                cb_reserve_back(CB_HOUT, H_TILES);
                // mm_init_short OUTSIDE acquire: llk_unpack_AB_matmul_init spins
                // waiting for unpack idle — deadlocks if held inside acquire.
                mm_init_short(CB_YCROSS, CB_B, 0 /*no transpose*/);
                tile_regs_acquire();
                for (uint32_t dt = 0; dt < XDT_TILES_C; ++dt) {
                    for (uint32_t nt = 0; nt < B_TILES_C; ++nt) {
                        for (uint32_t kt = 0; kt < XDT_TILES_R; ++kt) {
                            matmul_tiles(
                                CB_YCROSS, CB_B, dt * XDT_TILES_R + kt, kt * B_TILES_C + nt, dt * B_TILES_C + nt);
                        }
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < H_TILES; ++t) {
                    pack_tile(t, CB_HOUT);
                }
                tile_regs_release();
                cb_pop_front(CB_YCROSS, XDT_TILES);
                cb_pop_front(CB_B, B_TILES);
                cb_push_back(CB_HOUT, H_TILES);  // x_new ready

                // Part 2: h_next = gamma_last*h_prev + x_new
                // CB_H front = gamma_last*h_prev (from H-e), CB_HOUT front = x_new
                cb_wait_front(CB_H, H_TILES);
                cb_wait_front(CB_HOUT, H_TILES);
                cb_reserve_back(CB_H, H_TILES);
                add_tiles_init(CB_H, CB_HOUT);
                tile_regs_acquire();
                for (uint32_t t = 0; t < H_TILES; ++t) {
                    add_tiles(CB_H, CB_HOUT, t, t, t);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < H_TILES; ++t) {
                    pack_tile(t, CB_H);
                }
                tile_regs_release();
                cb_pop_front(CB_H, H_TILES);     // consume gamma_last*h_prev
                cb_pop_front(CB_HOUT, H_TILES);  // consume x_new scratch
                cb_push_back(CB_H, H_TILES);     // h_next for the next chunk

            }  // end chunk loop

            // ---- Copy final h_state → CB_HOUT for BRISC to write ----
            cb_wait_front(CB_H, H_TILES);
            cb_reserve_back(CB_HOUT, H_TILES);
            copy_tile_to_dst_init_short(CB_H);
            tile_regs_acquire();
            for (uint32_t t = 0; t < H_TILES; ++t) {
                copy_tile(CB_H, t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < H_TILES; ++t) {
                pack_tile(t, CB_HOUT);
            }
            tile_regs_release();
            cb_pop_front(CB_H, H_TILES);
            cb_push_back(CB_HOUT, H_TILES);

            cb_pop_front(CB_DSKIP, 1);
        }
#endif  // COMPILE_FOR_TRISC
    }  // impl
};  // class Op

}  // namespace nemotron30b_ops

// ---------------------------------------------------------------------------
// Kernel entry point (called by Metal firmware brisck.cc / ncrisck.cc / trisck.cc)
// ---------------------------------------------------------------------------
void kernel_main() {
    using namespace nemotron30b_ops;

    // Determine active core: head_h = core_y * GRID_C + core_x
    // (core coordinates from dataflow_api my_x[0]/my_y[0] on DM cores;
    //  for TRISC the grid is implied by how Metal assigns work)

    // Read runtime args (common_runtime_args, indexed from 0)
#if defined(COMPILE_FOR_NCRISC)
    ReaderArgs args{};
    args.x_dt_addr = get_common_arg_val<uint32_t>(0);
    args.B_addr = get_common_arg_val<uint32_t>(1);
    args.C_addr = get_common_arg_val<uint32_t>(2);
    args.x_addr = get_common_arg_val<uint32_t>(3);
    args.logd_addr = get_common_arg_val<uint32_t>(4);
    args.h_in_addr = get_common_arg_val<uint32_t>(5);
    args.D_skip_addr = get_common_arg_val<uint32_t>(6);
    args.n_chunks = get_common_arg_val<uint32_t>(7);
    const uint32_t head_h = my_y[0] * CTArgs::GRID_C + my_x[0];
    if (head_h < NUM_HEADS) {
        Op<CTArgs, true> op;
        op(args);
    }
#elif defined(COMPILE_FOR_BRISC)
    WriterArgs args{};
    args.y_addr = get_common_arg_val<uint32_t>(0);
    args.h_out_addr = get_common_arg_val<uint32_t>(1);
    args.n_chunks = get_common_arg_val<uint32_t>(2);
    Op<CTArgs, true> op;
    op(args);
#elif defined(COMPILE_FOR_TRISC)
    ComputeArgs args{};
    args.n_chunks = get_common_arg_val<uint32_t>(0);
    Op<CTArgs, true> op;
    op(args);
#endif
}
