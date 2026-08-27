// SPDX-License-Identifier: Apache-2.0
// Multi-core, K-streaming reader for full-N-per-core fused LN. Each core owns an
// M-slice [m_start_tile, m_start_tile+per_core_M_tiles); reads full N, streams K
// in K_num_blocks blocks. Each core reads full weights from DRAM independently
// (AttnOut weight ~1MB; independent reads are cheap). Residual/gamma/beta per slice.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_resid = tt::CBIndex::c_4;
constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
constexpr uint32_t cb_beta = tt::CBIndex::c_6;
constexpr uint32_t cb_scaler = tt::CBIndex::c_7;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;

FORCE_INLINE void fill_tile(uint32_t cb_id, uint32_t packed) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 512; i++) {
        p[i] = packed;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t g_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(a++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(a++);
    const uint32_t m_start_tile = get_arg_val<uint32_t>(a++);

    constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_M_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t K_block = get_compile_time_arg_val(4);
    constexpr uint32_t a_tb = get_compile_time_arg_val(5);
    constexpr uint32_t w_tb = get_compile_time_arg_val(6);
    constexpr uint32_t r_tb = get_compile_time_arg_val(7);
    constexpr uint32_t gb_tb = get_compile_time_arg_val(8);
    constexpr uint32_t K_num_blocks = K_tiles / K_block;

    constexpr auto a_args = TensorAccessorArgs<9>();
    const auto a_acc = TensorAccessor(a_args, a_addr, a_tb);
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tb);
    constexpr auto r_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto r_acc = TensorAccessor(r_args, r_addr, r_tb);
    constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto g_acc = TensorAccessor(g_args, g_addr, gb_tb);
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_args, b_addr, gb_tb);

    wh_generate_reduce_scaler(cb_scaler, scaler_packed);
    fill_tile(cb_eps, eps_packed);
    cb_reserve_back(cb_gamma, N_tiles);
    uint32_t gw = get_write_ptr(cb_gamma);
    for (uint32_t n = 0; n < N_tiles; n++) {
        noc_async_read_page(n, g_acc, gw);
        gw += gb_tb;
    }
    cb_reserve_back(cb_beta, N_tiles);
    uint32_t bw = get_write_ptr(cb_beta);
    for (uint32_t n = 0; n < N_tiles; n++) {
        noc_async_read_page(n, b_acc, bw);
        bw += gb_tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, N_tiles);
    cb_push_back(cb_beta, N_tiles);

    for (uint32_t mb = 0; mb < per_core_M_blocks; mb++) {
        uint32_t m_tile = m_start_tile + mb * M_block;
        for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
            uint32_t k0 = kb * K_block;
            // in0 block [M_block, K_block]
            cb_reserve_back(cb_in0, M_block * K_block);
            uint32_t aw = get_write_ptr(cb_in0);
            for (uint32_t m = 0; m < M_block; m++) {
                for (uint32_t k = 0; k < K_block; k++) {
                    noc_async_read_page((m_tile + m) * K_tiles + (k0 + k), a_acc, aw);
                    aw += a_tb;
                }
            }
            // in1 block [K_block, N_tiles]
            cb_reserve_back(cb_in1, K_block * N_tiles);
            uint32_t ww = get_write_ptr(cb_in1);
            for (uint32_t k = 0; k < K_block; k++) {
                for (uint32_t n = 0; n < N_tiles; n++) {
                    noc_async_read_page((k0 + k) * N_tiles + n, w_acc, ww);
                    ww += w_tb;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in0, M_block * K_block);
            cb_push_back(cb_in1, K_block * N_tiles);
        }
        // residual block [M_block, N_tiles]
        cb_reserve_back(cb_resid, M_block * N_tiles);
        uint32_t rw = get_write_ptr(cb_resid);
        for (uint32_t m = 0; m < M_block; m++) {
            for (uint32_t n = 0; n < N_tiles; n++) {
                noc_async_read_page((m_tile + m) * N_tiles + n, r_acc, rw);
                rw += r_tb;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_resid, M_block * N_tiles);
    }
}
