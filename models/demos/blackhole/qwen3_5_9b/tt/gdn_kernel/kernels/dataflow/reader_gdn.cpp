// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for fused GDN recurrence.
// Reads q, k_row, k_col, v, g, beta, and state from DRAM into L1 CBs.
// Uses TensorAccessor API for Blackhole compatibility.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_row_addr = get_arg_val<uint32_t>(1);
    uint32_t k_col_addr = get_arg_val<uint32_t>(2);
    uint32_t v_addr = get_arg_val<uint32_t>(3);
    uint32_t g_addr = get_arg_val<uint32_t>(4);
    uint32_t beta_addr = get_arg_val<uint32_t>(5);
    uint32_t state_addr = get_arg_val<uint32_t>(6);
    uint32_t pair_start = get_arg_val<uint32_t>(7);
    uint32_t num_pairs = get_arg_val<uint32_t>(8);

    // Compile-time args: Kt, Vt, tile_bytes, then 7 TensorAccessorArgs (one per tensor)
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t state_tiles = Kt * Vt;

    // TensorAccessorArgs at compile-time arg offsets 3..9 (1 arg each for interleaved DRAM)
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto kr_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto kc_args = TensorAccessorArgs<kr_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;
    constexpr uint32_t cb_state = tt::CBIndex::c_6;

    const auto q_rd = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto kr_rd = TensorAccessor(kr_args, k_row_addr, tile_bytes);
    const auto kc_rd = TensorAccessor(kc_args, k_col_addr, tile_bytes);
    const auto v_rd = TensorAccessor(v_args, v_addr, tile_bytes);
    const auto g_rd = TensorAccessor(g_args, g_addr, tile_bytes);
    const auto beta_rd = TensorAccessor(beta_args, beta_addr, tile_bytes);
    const auto state_rd = TensorAccessor(state_args, state_addr, tile_bytes);

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Read state [Kt*Vt tiles]
        cb_reserve_back(cb_state, state_tiles);
        uint32_t wp = get_write_ptr(cb_state);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_read_page(p * state_tiles + s, state_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_state, state_tiles);

        // Read q [Kt tiles]
        cb_reserve_back(cb_q, Kt);
        wp = get_write_ptr(cb_q);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(p * Kt + kt, q_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Kt);

        // Read k_row [Kt tiles]
        cb_reserve_back(cb_k_row, Kt);
        wp = get_write_ptr(cb_k_row);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(p * Kt + kt, kr_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_row, Kt);

        // Read k_col [Kt tiles]
        cb_reserve_back(cb_k_col, Kt);
        wp = get_write_ptr(cb_k_col);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(p * Kt + kt, kc_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_col, Kt);

        // Read v [Vt tiles]
        cb_reserve_back(cb_v, Vt);
        wp = get_write_ptr(cb_v);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc_async_read_page(p * Vt + vt, v_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Vt);

        // Read g [1 tile]
        cb_reserve_back(cb_g, 1);
        wp = get_write_ptr(cb_g);
        noc_async_read_page(p, g_rd, wp);
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);

        // Read beta [1 tile]
        cb_reserve_back(cb_beta, 1);
        wp = get_write_ptr(cb_beta);
        noc_async_read_page(p, beta_rd, wp);
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);
    }
}
