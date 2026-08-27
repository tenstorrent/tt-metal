// SPDX-License-Identifier: Apache-2.0
// Single-core reader for the fused-LN validation harness. Reads, per M-block:
//   A block [M_block,K] (all K), W [K,N] (all K), residual [M_block,N], and once:
//   gamma[N], beta[N]. Also generates the reduce scaler (1/W) and eps tiles.
// DRAM interleaved, tile layout. Minimal (correctness-first), not the fast path.
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

FORCE_INLINE void generate_tile_with_value(uint32_t cb_id, uint32_t packed_value) {
    cb_reserve_back(cb_id, 1);
    uint32_t w = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w);
    // fill only the first face column region enough for reduce scaler / eps broadcast
    for (uint32_t i = 0; i < 512; i++) {
        ptr[i] = packed_value;
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t argidx = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t r_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t g_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t b_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t scaler_packed = get_arg_val<uint32_t>(argidx++);
    const uint32_t eps_packed = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t M_block = get_compile_time_arg_val(3);
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t r_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t gb_tile_bytes = get_compile_time_arg_val(7);

    constexpr auto a_args = TensorAccessorArgs<8>();
    const auto a_acc = TensorAccessor(a_args, a_addr, a_tile_bytes);
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
    constexpr auto r_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto r_acc = TensorAccessor(r_args, r_addr, r_tile_bytes);
    constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto g_acc = TensorAccessor(g_args, g_addr, gb_tile_bytes);
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto b_acc = TensorAccessor(b_args, b_addr, gb_tile_bytes);

    // scaler + eps (single tiles, reused)
    wh_generate_reduce_scaler(cb_scaler, scaler_packed);
    generate_tile_with_value(cb_eps, eps_packed);

    // gamma/beta once (N_tiles each)
    cb_reserve_back(cb_gamma, N_tiles);
    uint32_t gw = get_write_ptr(cb_gamma);
    for (uint32_t n = 0; n < N_tiles; n++) {
        noc_async_read_page(n, g_acc, gw);
        gw += gb_tile_bytes;
    }
    cb_reserve_back(cb_beta, N_tiles);
    uint32_t bw = get_write_ptr(cb_beta);
    for (uint32_t n = 0; n < N_tiles; n++) {
        noc_async_read_page(n, b_acc, bw);
        bw += gb_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_gamma, N_tiles);
    cb_push_back(cb_beta, N_tiles);

    const uint32_t num_m_blocks = M_tiles / M_block;
    for (uint32_t mb = 0; mb < num_m_blocks; mb++) {
        // A block: [M_block, K_tiles] in K-blocks of size K_tiles (single K block here)
        cb_reserve_back(cb_in0, M_block * K_tiles);
        uint32_t aw = get_write_ptr(cb_in0);
        for (uint32_t m = 0; m < M_block; m++) {
            for (uint32_t k = 0; k < K_tiles; k++) {
                noc_async_read_page((mb * M_block + m) * K_tiles + k, a_acc, aw);
                aw += a_tile_bytes;
            }
        }
        // W block: [K_tiles, N_tiles]
        cb_reserve_back(cb_in1, K_tiles * N_tiles);
        uint32_t ww = get_write_ptr(cb_in1);
        for (uint32_t k = 0; k < K_tiles; k++) {
            for (uint32_t n = 0; n < N_tiles; n++) {
                noc_async_read_page(k * N_tiles + n, w_acc, ww);
                ww += w_tile_bytes;
            }
        }
        // residual block [M_block, N_tiles]
        cb_reserve_back(cb_resid, M_block * N_tiles);
        uint32_t rw = get_write_ptr(cb_resid);
        for (uint32_t m = 0; m < M_block; m++) {
            for (uint32_t n = 0; n < N_tiles; n++) {
                noc_async_read_page((mb * M_block + m) * N_tiles + n, r_acc, rw);
                rw += r_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, M_block * K_tiles);
        cb_push_back(cb_in1, K_tiles * N_tiles);
        cb_push_back(cb_resid, M_block * N_tiles);
    }
}
