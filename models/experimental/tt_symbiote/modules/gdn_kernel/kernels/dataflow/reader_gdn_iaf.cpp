// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel for fused GDN recurrence.
// Uses InterleavedAddrGenFast for DRAM (inputs) and optionally L1 (state).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_row_addr = get_arg_val<uint32_t>(1);
    uint32_t k_col_addr = get_arg_val<uint32_t>(2);
    uint32_t v_addr = get_arg_val<uint32_t>(3);
    uint32_t g_addr = get_arg_val<uint32_t>(4);
    uint32_t beta_addr = get_arg_val<uint32_t>(5);
    uint32_t state_addr = get_arg_val<uint32_t>(6);
    uint32_t pair_start = get_arg_val<uint32_t>(7);
    uint32_t num_pairs = get_arg_val<uint32_t>(8);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);  // 1 = state in L1, 0 = DRAM
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;
    constexpr uint32_t cb_state = tt::CBIndex::c_6;

    // Inputs always in DRAM
    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> q_rd = {
        .bank_base_address = q_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> kr_rd = {
        .bank_base_address = k_row_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> kc_rd = {
        .bank_base_address = k_col_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> v_rd = {
        .bank_base_address = v_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> g_rd = {
        .bank_base_address = g_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<is_dram> beta_rd = {
        .bank_base_address = beta_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // State can be in L1 or DRAM (compile-time switch)
    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_rd = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Read state [Kt*Vt tiles]
        cb_reserve_back(cb_state, state_tiles);
        uint32_t wp = get_write_ptr(cb_state);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_read_tile(p * state_tiles + s, state_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_state, state_tiles);

        // Read q [Kt tiles]
        cb_reserve_back(cb_q, Kt);
        wp = get_write_ptr(cb_q);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_tile(p * Kt + kt, q_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Kt);

        // Read k_row [Kt tiles]
        cb_reserve_back(cb_k_row, Kt);
        wp = get_write_ptr(cb_k_row);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_tile(p * Kt + kt, kr_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_row, Kt);

        // Read k_col [Kt tiles]
        cb_reserve_back(cb_k_col, Kt);
        wp = get_write_ptr(cb_k_col);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_tile(p * Kt + kt, kc_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_col, Kt);

        // Read v [Vt tiles]
        cb_reserve_back(cb_v, Vt);
        wp = get_write_ptr(cb_v);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc_async_read_tile(p * Vt + vt, v_rd, wp);
            wp += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Vt);

        // Read g [1 tile]
        cb_reserve_back(cb_g, 1);
        wp = get_write_ptr(cb_g);
        noc_async_read_tile(p, g_rd, wp);
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);

        // Read beta [1 tile]
        cb_reserve_back(cb_beta, 1);
        wp = get_write_ptr(cb_beta);
        noc_async_read_tile(p, beta_rd, wp);
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);
    }
}
