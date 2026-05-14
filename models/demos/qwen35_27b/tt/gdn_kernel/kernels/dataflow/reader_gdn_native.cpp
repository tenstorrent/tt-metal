// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Native-input reader for the fused GDN recurrence kernel (TensorAccessor variant).
//
// Reads kernel inputs at their natural upstream shapes via sub-tile reads — no
// per-pair retile required in the Python upstream chain. Mirrors the sub-tile read
// pattern from reader_gdn_fused.cpp but for the recurrence-only kernel.
//
//   - q from q_normed at (B, Nk_TP, Dk)         : row k_head of tile (batch_idx*Kt + kt)
//   - k from k_normed at (B, Nk_TP, Dk)         : row k_head of tile (batch_idx*Kt + kt); tile zeroed
//   - v from v_h     at (B, Nv_TP, Dv)          : row v_head of tile (batch_idx*Vt + vt)
//   - g scalar from g_pre   at (1, B, Nv_TP)    : scalar at (batch_idx, v_head) of tile 0
//   - beta scalar from beta_tt at (1, B, Nv_TP) : scalar at (batch_idx, v_head) of tile 0
//   - scale tile (1,1,1)                        : full tile, persistent across pairs
//   - state from rec_states at (num_pairs,Dk,Dv): full tile reads (unchanged)
//
// Pair mapping (matches the production Q-and-K repeat_interleave: each k_head
// services `repeat_factor` consecutive v_heads):
//   batch_idx = p / Nv_TP
//   v_head    = p % Nv_TP
//   k_head    = v_head / repeat_factor

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// ---- Sub-tile read helpers (templated on address-gen type so they work for
//      either InterleavedAddrGenFast or TensorAccessor) ----

template <typename AddrGen>
FORCE_INLINE void issue_row_reads(
    const AddrGen& addr_gen,
    uint32_t tile_id,
    uint32_t row,
    uint32_t scratch_slot  // 128-byte scratch area (two 64-byte face halves)
) {
    uint32_t face_base = (row < 16) ? 0 : 1024;
    uint32_t aligned_row = (row % 16) & ~1u;

    uint64_t src0 = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src0, scratch_slot, 64);

    uint64_t src1 = addr_gen.get_noc_addr(tile_id, face_base + 512 + aligned_row * 32);
    noc_async_read(src1, scratch_slot + 64, 64);
}

template <typename AddrGen>
FORCE_INLINE void issue_scalar_read(
    const AddrGen& addr_gen,
    uint32_t tile_id,
    uint32_t row,
    uint32_t col,
    uint32_t scratch_slot  // 64-byte scratch area
) {
    uint32_t face_base = (row < 16 ? 0 : 1024) + (col < 16 ? 0 : 512);
    uint32_t aligned_row = (row % 16) & ~1u;

    uint64_t src = addr_gen.get_noc_addr(tile_id, face_base + aligned_row * 32);
    noc_async_read(src, scratch_slot, 64);
}

// Copy a row from the 128-byte scratch slot into row 0 of the destination tile.
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

// Copy a single bf16 scalar from the 64-byte scratch slot into tile position [0,0].
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

void kernel_main() {
    // Runtime args
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    uint32_t g_addr = get_arg_val<uint32_t>(3);
    uint32_t beta_addr = get_arg_val<uint32_t>(4);
    uint32_t scale_addr = get_arg_val<uint32_t>(5);
    uint32_t state_addr = get_arg_val<uint32_t>(6);
    uint32_t pair_start = get_arg_val<uint32_t>(7);
    uint32_t num_pairs = get_arg_val<uint32_t>(8);

    // Compile-time args: Kt, Vt, tile_bytes, Nv_TP, Nk_TP, repeat_factor, then 7 TensorAccessorArgs
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t Nv_TP = get_compile_time_arg_val(3);
    constexpr uint32_t Nk_TP = get_compile_time_arg_val(4);
    constexpr uint32_t repeat_factor = get_compile_time_arg_val(5);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();

    // Scratch layout (per-pair; reused across all pairs)
    //   [0..511]     Q (Kt × 128 bytes per tile; two face halves of one row-pair)
    //   [512..1023]  K (Kt × 128 bytes)
    //   [1024..1535] V (Vt × 128 bytes)
    //   [1536..1599] g  scalar (64 bytes)
    //   [1600..1663] β  scalar (64 bytes)
    constexpr uint32_t SCRATCH_Q = 0;
    constexpr uint32_t SCRATCH_K = Kt * 128;
    constexpr uint32_t SCRATCH_V = 2 * Kt * 128;
    constexpr uint32_t SCRATCH_SCALAR = 3 * Kt * 128;

    // CB indices
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;
    constexpr uint32_t cb_state = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale = tt::CBIndex::c_15;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_21;

    // Address generators
    const auto q_rd = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_rd = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_rd = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto g_rd = TensorAccessor(g_args, g_addr, tile_bytes);
    const auto beta_rd = TensorAccessor(beta_args, beta_addr, tile_bytes);
    const auto scale_rd = TensorAccessor(scale_args, scale_addr, tile_bytes);
    const auto state_rd = TensorAccessor(state_args, state_addr, tile_bytes);

    // Reserve scratch CB (1 tile of L1 scratch)
    cb_reserve_back(cb_scratch, 1);
    uint32_t scratch_l1 = get_write_ptr(cb_scratch);

    // ---- Read scale tile once, persistent across all pairs ----
    cb_reserve_back(cb_scale, 1);
    noc_async_read_page(0, scale_rd, get_write_ptr(cb_scale));
    noc_async_read_barrier();
    cb_push_back(cb_scale, 1);

    // ---- Per-pair reads ----
    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;
        uint32_t batch_idx = p / Nv_TP;
        uint32_t v_head = p % Nv_TP;
        uint32_t k_head = v_head / repeat_factor;

        // Reserve all per-pair CBs upfront
        cb_reserve_back(cb_q, Kt);
        cb_reserve_back(cb_k, Kt);
        cb_reserve_back(cb_v, Vt);
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_beta, 1);
        cb_reserve_back(cb_state, state_tiles);

        uint32_t wp_q = get_write_ptr(cb_q);
        uint32_t wp_k = get_write_ptr(cb_k);
        uint32_t wp_v = get_write_ptr(cb_v);
        uint32_t wp_g = get_write_ptr(cb_g);
        uint32_t wp_beta = get_write_ptr(cb_beta);
        uint32_t wp_st = get_write_ptr(cb_state);

        // ==== Issue all NoC reads (batched, single barrier) ====

        // Q: Kt tiles × 2 face-half reads = 2*Kt reads → scratch[SCRATCH_Q..]
        for (uint32_t kt = 0; kt < Kt; kt++) {
            issue_row_reads(q_rd, batch_idx * Kt + kt, k_head, scratch_l1 + SCRATCH_Q + kt * 128);
        }
        // K: Kt tiles × 2 face-half reads = 2*Kt reads → scratch[SCRATCH_K..]
        for (uint32_t kt = 0; kt < Kt; kt++) {
            issue_row_reads(k_rd, batch_idx * Kt + kt, k_head, scratch_l1 + SCRATCH_K + kt * 128);
        }
        // V: Vt tiles × 2 face-half reads = 2*Vt reads → scratch[SCRATCH_V..]
        for (uint32_t vt = 0; vt < Vt; vt++) {
            issue_row_reads(v_rd, batch_idx * Vt + vt, v_head, scratch_l1 + SCRATCH_V + vt * 128);
        }
        // Scalars: 2 reads → scratch[SCRATCH_SCALAR..]
        issue_scalar_read(g_rd, 0, batch_idx, v_head, scratch_l1 + SCRATCH_SCALAR);
        issue_scalar_read(beta_rd, 0, batch_idx, v_head, scratch_l1 + SCRATCH_SCALAR + 64);

        // State: full tile reads directly into cb_state
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_read_page(p * state_tiles + s, state_rd, wp_st + s * tile_bytes);
        }

        // ==== Single barrier for all reads ====
        noc_async_read_barrier();

        // ==== Local copy phase (no NoC; L1 ops) ====

        // Q: copy rows to tile row 0 (other rows are stale; compute uses only row 0)
        for (uint32_t kt = 0; kt < Kt; kt++) {
            copy_row_to_tile(k_head, scratch_l1 + SCRATCH_Q + kt * 128, wp_q + kt * tile_bytes);
        }
        // K: zero tiles first (compute does transpose → outer-product matmul needs zeros)
        for (uint32_t kt = 0; kt < Kt; kt++) {
            uint32_t tile_addr = wp_k + kt * tile_bytes;
            zero_tile(tile_addr, tile_bytes);
            copy_row_to_tile(k_head, scratch_l1 + SCRATCH_K + kt * 128, tile_addr);
        }
        // V: copy rows to tile row 0
        for (uint32_t vt = 0; vt < Vt; vt++) {
            copy_row_to_tile(v_head, scratch_l1 + SCRATCH_V + vt * 128, wp_v + vt * tile_bytes);
        }
        // Scalars: copy to tile position [0,0]
        copy_scalar_to_tile(batch_idx, v_head, scratch_l1 + SCRATCH_SCALAR, wp_g);
        copy_scalar_to_tile(batch_idx, v_head, scratch_l1 + SCRATCH_SCALAR + 64, wp_beta);

        // ==== Push all CBs to compute ====
        cb_push_back(cb_q, Kt);
        cb_push_back(cb_k, Kt);
        cb_push_back(cb_v, Vt);
        cb_push_back(cb_g, 1);
        cb_push_back(cb_beta, 1);
        cb_push_back(cb_state, state_tiles);
    }

    // Release scratch
    cb_push_back(cb_scratch, 1);
    cb_pop_front(cb_scratch, 1);
}
