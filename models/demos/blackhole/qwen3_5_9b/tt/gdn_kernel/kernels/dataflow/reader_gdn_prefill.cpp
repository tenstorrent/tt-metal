// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill GDN reader kernel — reads N tokens per pair with state loaded once.
//
// Key difference from reader_gdn_fused.cpp (decode):
//   - Decode: 1 token per pair, reads row `batch_idx` from conv_out
//   - Prefill: N tokens per pair, reads rows 0..N-1 from conv_out
//   - State is read once at the start (not per-token)
//
// Input tensor layout (TILE_LAYOUT, 32x32 bfloat16 tiles):
//   conv_out: [1, N, qkv_dim_tp] → tile grid: ceil(N/32) rows × (qkv_dim_tp/32) cols
//   a:        [1, N, Nv_TP]      → tile grid: ceil(N/32) rows × ceil(Nv_TP/32) cols
//   b:        [1, N, Nv_TP]      → same as a
//
// For token t:
//   tile_row = t / 32
//   row_in_tile = t % 32
//   tile_id = tile_row * tiles_per_row + col_offset

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

// ---- NOC read helpers (issue only, no barrier) ----

template <bool is_dram>
FORCE_INLINE void issue_row_reads(
    const InterleavedAddrGenFast<is_dram>& addr_gen, uint32_t tile_id, uint32_t row, uint32_t scratch_slot) {
    uint32_t face_base = (row < 16) ? 0 : 1024;
    uint32_t aligned_row = (row % 16) & ~1u;

    uint64_t src0 = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src0, scratch_slot, 64);

    uint64_t src1 = addr_gen.get_noc_addr(tile_id, face_base + 512 + aligned_row * 32);
    noc_async_read(src1, scratch_slot + 64, 64);
}

template <bool is_dram>
FORCE_INLINE void issue_scalar_read(
    const InterleavedAddrGenFast<is_dram>& addr_gen,
    uint32_t tile_id,
    uint32_t row,
    uint32_t col,
    uint32_t scratch_slot) {
    uint32_t face_base = (row < 16 ? 0 : 1024) + (col < 16 ? 0 : 512);
    uint32_t aligned_row = (row % 16) & ~1u;

    uint64_t src = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src, scratch_slot, 64);
}

// ---- Local copy helpers (call AFTER barrier) ----

FORCE_INLINE void copy_row_to_tile(uint32_t row, uint32_t scratch_slot, uint32_t dest_l1) {
    uint32_t row_offset = ((row % 16) & 1u) * 32;

    volatile uint32_t* dst0 = reinterpret_cast<volatile uint32_t*>(dest_l1);
    volatile uint32_t* s0 = reinterpret_cast<volatile uint32_t*>(scratch_slot + row_offset);
    for (uint32_t i = 0; i < 8; i++) {
        dst0[i] = s0[i];
    }

    volatile uint32_t* dst1 = reinterpret_cast<volatile uint32_t*>(dest_l1 + 512);
    volatile uint32_t* s1 = reinterpret_cast<volatile uint32_t*>(scratch_slot + 64 + row_offset);
    for (uint32_t i = 0; i < 8; i++) {
        dst1[i] = s1[i];
    }
}

FORCE_INLINE void copy_scalar_to_tile(uint32_t row, uint32_t col, uint32_t scratch_slot, uint32_t dest_l1) {
    uint32_t row_offset = ((row % 16) & 1u) * 32;
    uint32_t face_col = col % 16;

    volatile uint16_t* src_val = reinterpret_cast<volatile uint16_t*>(scratch_slot + row_offset + face_col * 2);
    volatile uint16_t* dst_val = reinterpret_cast<volatile uint16_t*>(dest_l1);
    *dst_val = *src_val;
}

FORCE_INLINE void zero_tile(uint32_t l1_addr, uint32_t tile_bytes) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < tile_bytes / 4; i++) {
        ptr[i] = 0;
    }
}

// Extract row `row` from a full 32x32 tile at `src_tile_l1` and write it into row 0
// (left+right faces) of destination tile at `dst_tile_l1`. Same output contract as
// copy_row_to_tile, but source is a whole tile instead of a 2-row stub.
FORCE_INLINE void extract_row_from_full_tile(uint32_t src_tile_l1, uint32_t row, uint32_t dst_tile_l1) {
    // Tile faces: face 0 (top-left, 0..511), face 1 (top-right, 512..1023),
    //             face 2 (bot-left, 1024..1535), face 3 (bot-right, 1536..2047).
    // Within a face, row r (0..15) is at byte offset r*32 (16 bf16 cols * 2 bytes).
    uint32_t top_bottom_off = (row < 16) ? 0 : 1024;
    uint32_t row_in_face = row & 15u;
    uint32_t row_byte_off = row_in_face * 32;

    volatile uint32_t* src_L = reinterpret_cast<volatile uint32_t*>(src_tile_l1 + top_bottom_off + row_byte_off);
    volatile uint32_t* dst_L = reinterpret_cast<volatile uint32_t*>(dst_tile_l1);
    for (uint32_t i = 0; i < 8; i++) {
        dst_L[i] = src_L[i];
    }

    volatile uint32_t* src_R = reinterpret_cast<volatile uint32_t*>(src_tile_l1 + top_bottom_off + 512 + row_byte_off);
    volatile uint32_t* dst_R = reinterpret_cast<volatile uint32_t*>(dst_tile_l1 + 512);
    for (uint32_t i = 0; i < 8; i++) {
        dst_R[i] = src_R[i];
    }
}

// Extract single bf16 element at (row, col) from a full 32x32 tile.
FORCE_INLINE void extract_scalar_from_full_tile(
    uint32_t src_tile_l1, uint32_t row, uint32_t col, uint32_t dst_tile_l1) {
    uint32_t face_off = ((row < 16) ? 0u : 1024u) + ((col < 16) ? 0u : 512u);
    uint32_t row_in_face = row & 15u;
    uint32_t col_in_face = col & 15u;
    uint32_t byte_off = row_in_face * 32 + col_in_face * 2;

    volatile uint16_t* src = reinterpret_cast<volatile uint16_t*>(src_tile_l1 + face_off + byte_off);
    volatile uint16_t* dst = reinterpret_cast<volatile uint16_t*>(dst_tile_l1);
    *dst = *src;
}

void kernel_main() {
    // Runtime args
    uint32_t conv_out_addr = get_arg_val<uint32_t>(0);   // [1, N, qkv_dim_tp]
    uint32_t a_addr = get_arg_val<uint32_t>(1);          // [1, N, Nv_TP]
    uint32_t b_addr = get_arg_val<uint32_t>(2);          // [1, N, Nv_TP]
    uint32_t neg_exp_A_addr = get_arg_val<uint32_t>(3);  // [1, 1, Nv_TP]
    uint32_t dt_bias_addr = get_arg_val<uint32_t>(4);    // [1, 1, Nv_TP]
    uint32_t norm_w_addr = get_arg_val<uint32_t>(5);     // [1, 1, Dv]
    uint32_t scale_addr = get_arg_val<uint32_t>(6);      // [1 tile]
    uint32_t rms_scale_addr = get_arg_val<uint32_t>(7);  // [1 tile]
    uint32_t state_addr = get_arg_val<uint32_t>(8);      // [num_pairs, Dk, Dv]
    uint32_t rms_eps_addr = get_arg_val<uint32_t>(9);    // [1 tile]
    uint32_t pair_start = get_arg_val<uint32_t>(10);
    uint32_t num_pairs = get_arg_val<uint32_t>(11);

    // Compile-time args
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t Nv_TP = get_compile_time_arg_val(5);
    constexpr uint32_t Nk_TP = get_compile_time_arg_val(6);
    constexpr uint32_t repeat_factor = get_compile_time_arg_val(7);
    constexpr uint32_t key_tile_offset = get_compile_time_arg_val(8);
    constexpr uint32_t v_tile_offset = get_compile_time_arg_val(9);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(10);          // N
    constexpr uint32_t conv_tiles_per_row = get_compile_time_arg_val(11);  // qkv_dim_tp / 32
    constexpr uint32_t ab_tiles_per_row = get_compile_time_arg_val(12);    // ceil(Nv_TP / 32)
    constexpr uint32_t state_tiles = Kt * Vt;

    // Scratch layout: whole tiles for Q/K/V/a/b (loaded once per 32-token tile-row),
    // plus per-pair scalar stubs for neg_exp_A/dt_bias (loaded once per pair).
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t SCRATCH_Q = 0;
    constexpr uint32_t SCRATCH_K = Kt * TILE_BYTES;
    constexpr uint32_t SCRATCH_V = 2 * Kt * TILE_BYTES;
    constexpr uint32_t SCRATCH_A = (2 * Kt + Vt) * TILE_BYTES;
    constexpr uint32_t SCRATCH_B = SCRATCH_A + TILE_BYTES;
    constexpr uint32_t SCRATCH_SCALAR = SCRATCH_B + TILE_BYTES;  // neg_exp_A + dt_bias (64 B each)

    // CB indices
    constexpr uint32_t cb_q_raw = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_raw = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_a = tt::CBIndex::c_9;
    constexpr uint32_t cb_b = tt::CBIndex::c_10;
    constexpr uint32_t cb_neg_exp_A = tt::CBIndex::c_12;
    constexpr uint32_t cb_dt_bias = tt::CBIndex::c_13;
    constexpr uint32_t cb_norm_w = tt::CBIndex::c_14;
    constexpr uint32_t cb_scale = tt::CBIndex::c_15;
    constexpr uint32_t cb_state = tt::CBIndex::c_6;
    constexpr uint32_t cb_rms_scale = tt::CBIndex::c_31;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_19;
    constexpr uint32_t cb_rms_eps = tt::CBIndex::c_20;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_21;

    constexpr bool is_dram = true;

    // Address generators
    const InterleavedAddrGenFast<is_dram> conv_rd = {
        .bank_base_address = conv_out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> a_rd = {
        .bank_base_address = a_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> b_rd = {
        .bank_base_address = b_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> neg_exp_A_rd = {
        .bank_base_address = neg_exp_A_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> dt_bias_rd = {
        .bank_base_address = dt_bias_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> norm_w_rd = {
        .bank_base_address = norm_w_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> scale_rd = {
        .bank_base_address = scale_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> rms_scale_rd = {
        .bank_base_address = rms_scale_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> rms_eps_rd = {
        .bank_base_address = rms_eps_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_rd = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // Get scratch L1 address
    cb_reserve_back(cb_scratch, 1);
    uint32_t scratch_l1 = get_write_ptr(cb_scratch);

    // ---- Read constants (once, persistent) ----
    cb_reserve_back(cb_norm_w, Vt);
    cb_reserve_back(cb_scale, 1);
    cb_reserve_back(cb_rms_scale, 1);
    cb_reserve_back(cb_rms_eps, 1);

    uint32_t wp = get_write_ptr(cb_norm_w);
    for (uint32_t vt = 0; vt < Vt; vt++) {
        noc_async_read_tile(vt, norm_w_rd, wp);
        wp += tile_bytes;
    }
    noc_async_read_tile(0, scale_rd, get_write_ptr(cb_scale));
    noc_async_read_tile(0, rms_scale_rd, get_write_ptr(cb_rms_scale));
    noc_async_read_tile(0, rms_eps_rd, get_write_ptr(cb_rms_eps));

    noc_async_read_barrier();

    cb_push_back(cb_norm_w, Vt);
    cb_push_back(cb_scale, 1);
    cb_push_back(cb_rms_scale, 1);
    cb_push_back(cb_rms_eps, 1);

    generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);

    // ---- Per-pair processing ----
    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        uint32_t v_head = p % Nv_TP;
        uint32_t k_head = v_head / repeat_factor;

        // Pre-read neg_exp_A and dt_bias for this head (constant across tokens)
        // We'll read them once and cache in scratch for reuse
        issue_scalar_read(neg_exp_A_rd, 0, 0, v_head, scratch_l1 + SCRATCH_SCALAR + 128);
        issue_scalar_read(dt_bias_rd, 0, 0, v_head, scratch_l1 + SCRATCH_SCALAR + 192);

        // Load state ONCE for this pair (stays in cb_state for all tokens)
        cb_reserve_back(cb_state, state_tiles);
        uint32_t wp_st = get_write_ptr(cb_state);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_read_tile(p * state_tiles + s, state_rd, wp_st + s * tile_bytes);
        }

        noc_async_read_barrier();
        cb_push_back(cb_state, state_tiles);

        // Cache neg_exp_A and dt_bias scratch locations (constant across all tokens for this pair)
        uint32_t neg_scratch = scratch_l1 + SCRATCH_SCALAR + 128;
        uint32_t dt_scratch = scratch_l1 + SCRATCH_SCALAR + 192;

        // ---- Stream N tokens for this pair, grouped by tile-row (32 tokens per tile) ----
        // Each tile-row: 1 NOC read per Q/K/V/a/b tile (amortized across up to 32 tokens),
        // instead of 1 partial-tile NOC read per token as in the original implementation.
        uint32_t num_tile_rows = (num_tokens + 31) / 32;

        for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
            uint32_t base = tr * 32;
            uint32_t rows_here = (base + 32 <= num_tokens) ? 32 : (num_tokens - base);

            // ---- Whole-tile NOC reads for this tile-row's Q/K/V/a/b ----
            for (uint32_t kt = 0; kt < Kt; kt++) {
                uint32_t q_tile = tr * conv_tiles_per_row + k_head * Kt + kt;
                noc_async_read_tile(q_tile, conv_rd, scratch_l1 + SCRATCH_Q + kt * TILE_BYTES);
            }
            for (uint32_t kt = 0; kt < Kt; kt++) {
                uint32_t k_tile = tr * conv_tiles_per_row + key_tile_offset + k_head * Kt + kt;
                noc_async_read_tile(k_tile, conv_rd, scratch_l1 + SCRATCH_K + kt * TILE_BYTES);
            }
            for (uint32_t vt = 0; vt < Vt; vt++) {
                uint32_t v_tile = tr * conv_tiles_per_row + v_tile_offset + v_head * Vt + vt;
                noc_async_read_tile(v_tile, conv_rd, scratch_l1 + SCRATCH_V + vt * TILE_BYTES);
            }
            // a, b: [1, N, Nv_TP] tiles — one tile per tile-row when Nv_TP <= 32.
            uint32_t ab_tile = tr * ab_tiles_per_row;
            noc_async_read_tile(ab_tile, a_rd, scratch_l1 + SCRATCH_A);
            noc_async_read_tile(ab_tile, b_rd, scratch_l1 + SCRATCH_B);

            noc_async_read_barrier();  // ONE barrier per 32 tokens

            // ---- Extract per-token tiles from scratch (local L1 copies, no NOC) ----
            for (uint32_t r = 0; r < rows_here; r++) {
                cb_reserve_back(cb_q_raw, Kt);
                cb_reserve_back(cb_k_raw, Kt);
                cb_reserve_back(cb_v, Vt);
                cb_reserve_back(cb_a, 1);
                cb_reserve_back(cb_b, 1);
                cb_reserve_back(cb_neg_exp_A, 1);
                cb_reserve_back(cb_dt_bias, 1);

                uint32_t wp_q = get_write_ptr(cb_q_raw);
                uint32_t wp_k = get_write_ptr(cb_k_raw);
                uint32_t wp_v = get_write_ptr(cb_v);
                uint32_t wp_a = get_write_ptr(cb_a);
                uint32_t wp_b = get_write_ptr(cb_b);
                uint32_t wp_neg = get_write_ptr(cb_neg_exp_A);
                uint32_t wp_dt = get_write_ptr(cb_dt_bias);

                // Q: extract row r from each of Kt whole tiles
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    extract_row_from_full_tile(scratch_l1 + SCRATCH_Q + kt * TILE_BYTES, r, wp_q + kt * tile_bytes);
                }
                // K: zero then extract (outer product requires other rows to be zero)
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    uint32_t dst = wp_k + kt * tile_bytes;
                    zero_tile(dst, tile_bytes);
                    extract_row_from_full_tile(scratch_l1 + SCRATCH_K + kt * TILE_BYTES, r, dst);
                }
                // V
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    extract_row_from_full_tile(scratch_l1 + SCRATCH_V + vt * TILE_BYTES, r, wp_v + vt * tile_bytes);
                }

                // Scalars: extract element (r, v_head) for a/b; (0, v_head) for per-pair constants
                extract_scalar_from_full_tile(scratch_l1 + SCRATCH_A, r, v_head, wp_a);
                extract_scalar_from_full_tile(scratch_l1 + SCRATCH_B, r, v_head, wp_b);
                copy_scalar_to_tile(0, v_head, neg_scratch, wp_neg);
                copy_scalar_to_tile(0, v_head, dt_scratch, wp_dt);

                cb_push_back(cb_q_raw, Kt);
                cb_push_back(cb_k_raw, Kt);
                cb_push_back(cb_v, Vt);
                cb_push_back(cb_a, 1);
                cb_push_back(cb_b, 1);
                cb_push_back(cb_neg_exp_A, 1);
                cb_push_back(cb_dt_bias, 1);
            }
        }
    }

    // Release scratch
    cb_push_back(cb_scratch, 1);
    cb_pop_front(cb_scratch, 1);
}
