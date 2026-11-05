// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for QkvHeadsNormRope. A core's work = (user u, tensors t0 .. t0 + t_count - 1 of {q, k, v}; one tensor
// per core when the grid holds 3 B cores, the three in sequence otherwise): gathers user u's row of every head
// of each tensor from the fused QKV projection output ([1, 1, B, N] bf16 TILE, L1 interleaved: row u =
// user u, column tiles [q heads | k heads | v heads] x (D / 32)) into the head-row tiles of the norm + RoPE
// compute (row h of CB tile c = user u's columns of head h, column tile c: the nlp_create_qkv_heads_decode
// layout without that program); then the tensor's gamma column tiles (q / k) and the user's cos / sin row (row 0
// of a tile per column) as reader_qkv_norm_rope.cpp. 32-byte face-row reads: L1 sources only (a DRAM source
// trips the Blackhole DRAM-read alignment rule, tt-metal #16667). A tile row r lives in faces (r/16)*2 (columns
// 0-15) and +1 (columns 16-31) at byte offset (r%16)*32.
// compile-time args: [cb_x, cb_g, cb_cos, cb_sin, cb_scaler, cols_t, hq, hkv]
//                    + TensorAccessorArgs(x, gq, gk, cos, sin)
// runtime args: [x_addr, gq_addr, gk_addr, cos_addr, sin_addr, user, t0, t_count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gq_addr = get_arg_val<uint32_t>(1);
    const uint32_t gk_addr = get_arg_val<uint32_t>(2);
    const uint32_t cos_addr = get_arg_val<uint32_t>(3);
    const uint32_t sin_addr = get_arg_val<uint32_t>(4);
    const uint32_t user = get_arg_val<uint32_t>(5);
    const uint32_t t0 = get_arg_val<uint32_t>(6);
    const uint32_t t_count = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cols_t = get_compile_time_arg_val(5);
    constexpr uint32_t hq = get_compile_time_arg_val(6);
    constexpr uint32_t hkv = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto gq_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;

    const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto s_gq = TensorAccessor(gq_args, gq_addr, tile_bytes);
    const auto s_gk = TensorAccessor(gk_args, gk_addr, tile_bytes);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, tile_bytes);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, tile_bytes);
    // the user's row inside a source tile: face pair (u/16)*2, byte offset (u%16)*32 inside each face
    const uint32_t src_off = (user >> 4) * 2 * face_bytes + (user & 15) * 32;
    const uint32_t src_face = (user >> 4) * 2;
    const uint32_t src_word = (user & 15) * 8;
    // unit scaler for the row reduce: bf16 1.0 in the first row of every face, zeros elsewhere; the 2 x cols_t
    // pages after it stage the cos / sin tiles of a tensor (never pushed)
    CircularBuffer cs(cb_scaler);
    cs.reserve_back(1 + 2 * cols_t);
    const uint32_t scaler_addr = get_write_ptr(cb_scaler);
    const uint32_t stage_addr = scaler_addr + tile_bytes;
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_addr);
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t j = 0; j < 8; ++j) {
                p[f * 128 + j] = 0x3F803F80u;
            }
        }
    }
    cs.push_back(1);
    CircularBuffer cx(cb_x), cg(cb_g), ccos(cb_cos), csin(cb_sin);
    for (uint32_t t = t0; t < t0 + t_count; ++t) {
        const uint32_t heads = t == 0 ? hq : hkv;
        const uint32_t base = t == 0 ? 0 : (t == 1 ? hq * cols_t : (hq + hkv) * cols_t);
        cx.reserve_back(cols_t);
        const uint32_t dst_base = get_write_ptr(cb_x);
        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t dst_off = (h >> 4) * 2 * face_bytes + (h & 15) * 32;
            for (uint32_t col = 0; col < cols_t; ++col) {
                const uint64_t src = s_x.get_noc_addr(base + h * cols_t + col) + src_off;
                const uint32_t dst = dst_base + col * tile_bytes + dst_off;
                noc_async_read(src, dst, 32);                            // columns 0-15 of the row
                noc_async_read(src + face_bytes, dst + face_bytes, 32);  // columns 16-31
            }
        }
        if (t < 2) {  // q / k: gamma + the user's cos / sin row
            cg.reserve_back(cols_t);
            const uint32_t a_g = get_write_ptr(cb_g);
            const auto& s_g = t == 0 ? s_gq : s_gk;
            for (uint32_t c = 0; c < cols_t; ++c) {
                noc_async_read(s_g.get_noc_addr(c), a_g + c * tile_bytes, tile_bytes);
            }
        }
        noc_async_read_barrier();  // the head rows (and the gamma) are in L1
        cx.push_back(cols_t);
        if (t < 2) {
            cg.push_back(cols_t);
            ccos.reserve_back(cols_t);
            csin.reserve_back(cols_t);
            // the user's cos / sin row into row 0 of each destination tile (the row the bcast-rows multiply
            // reads): every cos / sin tile staged whole (a DRAM source: whole-tile reads only, tt-metal #16667),
            // all 2 x cols_t reads in flight, one barrier, then the local row copies
            const uint32_t a_cos = get_write_ptr(cb_cos), a_sin = get_write_ptr(cb_sin);
            for (uint32_t which = 0; which < 2; ++which) {
                const auto& acc = which == 0 ? s_cos : s_sin;
                for (uint32_t c = 0; c < cols_t; ++c) {
                    noc_async_read(acc.get_noc_addr(c), stage_addr + (which * cols_t + c) * tile_bytes, tile_bytes);
                }
            }
            noc_async_read_barrier();
            for (uint32_t which = 0; which < 2; ++which) {
                const uint32_t dst_base2 = which == 0 ? a_cos : a_sin;
                for (uint32_t c = 0; c < cols_t; ++c) {
                    volatile tt_l1_ptr uint32_t* stage =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_addr + (which * cols_t + c) * tile_bytes);
                    volatile tt_l1_ptr uint32_t* dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_base2 + c * tile_bytes);
                    for (uint32_t j = 0; j < 8; ++j) {
                        dst[j] = stage[src_face * 128 + src_word + j];              // columns 0-15
                        dst[128 + j] = stage[(src_face + 1) * 128 + src_word + j];  // columns 16-31
                    }
                }
            }
            ccos.push_back(cols_t);
            csin.push_back(cols_t);
        }
    }
}
