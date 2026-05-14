// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multicore layernorm reader: fills scaler/eps CBs, loads gamma/beta,
// then reads assigned rows of input.
// Runtime args: src_addr, gamma_addr, beta_addr, start_row, num_rows, Wt, eps_bits

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    uint32_t beta_addr = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);
    uint32_t num_rows = get_arg_val<uint32_t>(4);
    uint32_t Wt = get_arg_val<uint32_t>(5);
    uint32_t eps_bits = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_gamma = 2;
    constexpr uint32_t cb_beta = 3;
    constexpr uint32_t cb_eps = 4;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s_src = TensorAccessor(s_args, src_addr);
    constexpr auto g_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto s_gamma = TensorAccessor(g_args, gamma_addr);
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto s_beta = TensorAccessor(b_args, beta_addr);

    // Compute 1/N as bfloat16 (N = Wt * 32)
    uint32_t N = Wt * 32;
    float inv_n = 1.0f / static_cast<float>(N);
    uint32_t inv_n_bits;
    __builtin_memcpy(&inv_n_bits, &inv_n, sizeof(inv_n));
    uint16_t inv_n_bf16 = static_cast<uint16_t>(inv_n_bits >> 16);

    // Fill scaler CB with 1/N
    cb_reserve_back(cb_scaler, 1);
    auto* scaler_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < 1024; i++) {
        scaler_ptr[i] = inv_n_bf16;
    }
    cb_push_back(cb_scaler, 1);

    // Fill eps CB
    cb_reserve_back(cb_eps, 1);
    auto* eps_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_eps));
    uint16_t eps_bf16 = static_cast<uint16_t>(eps_bits & 0xFFFF);
    for (uint32_t i = 0; i < 1024; i++) {
        eps_ptr[i] = eps_bf16;
    }
    cb_push_back(cb_eps, 1);

    // Load gamma (Wt tiles, persistent)
    for (uint32_t w = 0; w < Wt; w++) {
        cb_reserve_back(cb_gamma, 1);
        uint32_t l1_addr = get_write_ptr(cb_gamma);
        noc_async_read_tile(w, s_gamma, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_gamma, 1);
    }

    // Load beta (Wt tiles, persistent)
    for (uint32_t w = 0; w < Wt; w++) {
        cb_reserve_back(cb_beta, 1);
        uint32_t l1_addr = get_write_ptr(cb_beta);
        noc_async_read_tile(w, s_beta, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_beta, 1);
    }

    // Read input tiles for assigned rows
    for (uint32_t mt = 0; mt < num_rows; mt++) {
        uint32_t row = start_row + mt;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            uint32_t tile_id = row * Wt + wt;
            cb_reserve_back(cb_in, 1);
            uint32_t l1_addr = get_write_ptr(cb_in);
            noc_async_read_tile(tile_id, s_src, l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}
