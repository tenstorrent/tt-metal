// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DFlash drafter q / k prep, reader (NCRISC) (dflash/qk_prep.py), one core per draft row r: row r's Q and K tiles
// (heads in tile rows), its V tiles straight into this core's shard of the V output, the q / k RMSNorm weights (full
// tiles, every row = the weight), the RoPE rotation matrix T, a reduce scaler, and row r's cos / sin rows (rope cache
// row pos[r], Meta layout) as row-broadcast tiles (row 0 of every face).
//
// runtime args: [q, k, v, gamma, cos, sin, rot_pos, trans, v_out, r (, kv write core noc x, y with PREP_NOTIFY)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// PREP_SEM_BASE: the fused KV write's semaphores, PREP_SEM_BASE (K shards) and PREP_SEM_BASE + 1 (V shards)

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t cos_addr = get_arg_val<uint32_t>(4);
    const uint32_t sin_addr = get_arg_val<uint32_t>(5);
    const uint32_t rp_addr = get_arg_val<uint32_t>(6);
    const uint32_t t_addr = get_arg_val<uint32_t>(7);
    const uint32_t v_out = get_arg_val<uint32_t>(8);
    const uint32_t r = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);       // q gamma 0, 1, k gamma 0, 1
    constexpr uint32_t cb_cs = get_compile_time_arg_val(3);      // cos 0, 1, sin 0, 1 (row broadcast)
    constexpr uint32_t cb_t = get_compile_time_arg_val(4);       // T
    constexpr uint32_t cb_scl = get_compile_time_arg_val(5);     // reduce scaler
    constexpr uint32_t cb_scr = get_compile_time_arg_val(6);     // 3 x 128 bytes: positions, cos row, sin row
    constexpr uint32_t row_bytes = get_compile_time_arg_val(7);  // one rope cache row (head_dim BF16)
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto rp_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<rp_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;

    const auto s_q = TensorAccessor(q_args, q_addr, tile);
    const auto s_k = TensorAccessor(k_args, k_addr, tile);
    const auto s_v = TensorAccessor(v_args, v_addr, tile);
    const auto s_g = TensorAccessor(g_args, g_addr, tile);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, row_bytes);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, row_bytes);
    const auto s_rp = TensorAccessor(rp_args, rp_addr, 128);
    const auto s_t = TensorAccessor(t_args, t_addr, tile);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_rp.get_noc_addr(0), scr, 128);
    cb_reserve_back(cb_q, 2);
    cb_reserve_back(cb_k, 2);
    for (uint32_t d = 0; d < 2; ++d) {
        noc_async_read(s_q.get_noc_addr(2 * r + d), get_write_ptr(cb_q) + d * tile, tile);
        noc_async_read(s_k.get_noc_addr(2 * r + d), get_write_ptr(cb_k) + d * tile, tile);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
#ifdef PREP_JOINT
    // q and k in one tile: k's head rows 0 .. NKV - 1 -> rows NQ .. NQ + NKV - 1 of the q tile (NQ + NKV <= 16: faces
    // 0 / 1, 32 bytes per face row); compute normalizes and rotates both at once (per-row weights)
    for (uint32_t d = 0; d < 2; ++d) {
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t h = 0; h < NKV; ++h) {
                volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_k) + d * tile + f * 512 + h * 32);
                volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_write_ptr(cb_q) + d * tile + f * 512 + (NQ + h) * 32);
                for (uint32_t i = 0; i < 8; ++i) {
                    dst[i] = src[i];
                }
            }
        }
    }
#endif
    const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[r];
    noc_async_read(s_cos.get_noc_addr(p), scr + 128, row_bytes);
    noc_async_read(s_sin.get_noc_addr(p), scr + 256, row_bytes);
    cb_push_back(cb_q, 2);
#ifndef PREP_JOINT
    cb_push_back(cb_k, 2);
#endif
    for (uint32_t d = 0; d < 2; ++d) {
        noc_async_read(s_v.get_noc_addr(2 * r + d), v_out + d * tile, tile);
    }
    cb_reserve_back(cb_scl, 1);
    cb_reserve_back(cb_cs, 4);
    // scaler: 1.0 in row 0 of each face, zeros elsewhere
    const uint32_t scl = get_write_ptr(cb_scl), cs = get_write_ptr(cb_cs);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, scl + z, MEM_ZEROS_SIZE);
    }
#ifdef PREP_JOINT
    constexpr uint32_t n_g = 2;  // per-row weights [q_norm rows | k_norm rows] as two full tiles
#else
    constexpr uint32_t n_g = 4;
#endif
    cb_reserve_back(cb_g, n_g);
    cb_reserve_back(cb_t, 1);
    for (uint32_t i = 0; i < n_g; ++i) {
        noc_async_read(s_g.get_noc_addr(i), get_write_ptr(cb_g) + i * tile, tile);
    }
    noc_async_read(s_t.get_noc_addr(0), get_write_ptr(cb_t), tile);
    noc_async_read_barrier();
    invalidate_l1_cache();
#ifdef PREP_NOTIFY
    // V shard written: tell the fused KV write (semaphore PREP_SEM_BASE + 1 on its core)
    noc_semaphore_inc(
        get_noc_addr(get_arg_val<uint32_t>(10), get_arg_val<uint32_t>(11), get_semaphore(PREP_SEM_BASE + 1)), 1);
    noc_async_atomic_barrier();  // the increment lands before the kernel ends
#endif
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scl);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t i = 0; i < 8; ++i) {
            sc[f * 128 + i] = 0x3F803F80;
        }
    }
    cb_push_back(cb_scl, 1);
    // cos / sin: columns 32 t .. 32 t + 31 -> row 0 of faces 0 / 1 (and 2 / 3) of tile t
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128);
    for (uint32_t s = 0; s < 2; ++s) {      // cos, sin
        for (uint32_t t = 0; t < 2; ++t) {  // 32-column tile
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs + (2 * s + t) * tile);
            for (uint32_t h = 0; h < 2; ++h) {  // 16-column half
                for (uint32_t i = 0; i < 8; ++i) {
                    const uint32_t w = src[s * 32 + t * 16 + h * 8 + i];
                    o[h * 128 + i] = w;
                    o[(2 + h) * 128 + i] = w;
                }
            }
        }
    }
    cb_push_back(cb_cs, 4);
    cb_push_back(cb_g, n_g);
    cb_push_back(cb_t, 1);
}
