// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads input tiles row by row for LayerNorm.
// Also fills persistent CBs: scaler (1/N), gamma, beta, eps.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t beta_addr = get_arg_val<uint32_t>(2);
    uint32_t Mt = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t eps_bits = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_gamma = 2;
    constexpr uint32_t cb_beta = 3;
    constexpr uint32_t cb_eps = 4;

    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto g_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const auto s = TensorAccessor(s_args, src_addr);
    const auto g = TensorAccessor(g_args, gamma_addr);
    const auto b = TensorAccessor(b_args, beta_addr);

    uint32_t N = Wt * 32;
    float recip_N = 1.0f / static_cast<float>(N);
    uint32_t recip_N_bits;
    __builtin_memcpy(&recip_N_bits, &recip_N, sizeof(recip_N_bits));
    uint16_t recip_N_bf16 = static_cast<uint16_t>(recip_N_bits >> 16);

    // Fill scaler CB with 1/N values (bfloat16)
    cb_reserve_back(cb_scaler, 1);
    uint32_t scaler_addr = get_write_ptr(cb_scaler);
    volatile tt_l1_ptr uint16_t* scaler_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scaler_addr);
    for (uint32_t i = 0; i < 32 * 32; i++) {
        scaler_ptr[i] = recip_N_bf16;
    }
    cb_push_back(cb_scaler, 1);

    // Fill eps CB
    uint16_t eps_bf16 = static_cast<uint16_t>(eps_bits);
    cb_reserve_back(cb_eps, 1);
    uint32_t eps_addr_l1 = get_write_ptr(cb_eps);
    volatile tt_l1_ptr uint16_t* eps_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(eps_addr_l1);
    for (uint32_t i = 0; i < 32 * 32; i++) {
        eps_ptr[i] = eps_bf16;
    }
    cb_push_back(cb_eps, 1);

    // Load gamma tiles (Wt tiles, persistent, broadcast-row format)
    for (uint32_t wt = 0; wt < Wt; wt++) {
        cb_reserve_back(cb_gamma, 1);
        uint32_t l1_addr = get_write_ptr(cb_gamma);
        noc_async_read_tile(wt, g, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_gamma, 1);
    }

    // Load beta tiles (Wt tiles, persistent, broadcast-row format)
    for (uint32_t wt = 0; wt < Wt; wt++) {
        cb_reserve_back(cb_beta, 1);
        uint32_t l1_addr = get_write_ptr(cb_beta);
        noc_async_read_tile(wt, b, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);
    }

    // Read input tiles row by row
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t wt = 0; wt < Wt; wt++) {
            uint32_t tile_idx = mt * Wt + wt;
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, s, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
