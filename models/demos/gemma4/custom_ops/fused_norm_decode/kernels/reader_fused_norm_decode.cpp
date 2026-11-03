// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for FusedNormDecode: this core's `per` column tiles of x0 (and x1), the matching full gamma tiles
// (rows replicated), the residual tiles and the unit reduce scaler; then it waits for the N partial-sum blocks the
// writers of all cores deposit in cb_stage (slot = sender, NIN x 32 fp32 per slot), scatters block j into column j
// of the fp32 receive tiles (tile i*RT + j/32 for input i; columns >= N stay zero) and publishes them, resetting the
// semaphore for the next replay.
// compile-time args: [cb_x0, cb_x1, cb_g0, cb_g1, cb_g2, cb_res, cb_scaler, cb_recv, cb_stage, per, N, RT, NIN,
//                     NOUT, has_res, gmask, sem_id] + TensorAccessorArgs(x0, x1, g0, g1, g2, res)
// runtime args: [x0_addr, x1_addr, g0_addr, g1_addr, g2_addr, res_addr, c0, gc0]  (c0 = first tile of this
// core's columns in x / residual, incl. the row-tile offset; gc0 = the same columns in the one-row-tile gammas)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x0_addr = get_arg_val<uint32_t>(0);
    const uint32_t x1_addr = get_arg_val<uint32_t>(1);
    const uint32_t g0_addr = get_arg_val<uint32_t>(2);
    const uint32_t g1_addr = get_arg_val<uint32_t>(3);
    const uint32_t g2_addr = get_arg_val<uint32_t>(4);
    const uint32_t res_addr = get_arg_val<uint32_t>(5);
    const uint32_t c0 = get_arg_val<uint32_t>(6);
    const uint32_t gc0 = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_x0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_g2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_res = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(7);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(8);
    constexpr uint32_t per = get_compile_time_arg_val(9);
    constexpr uint32_t N = get_compile_time_arg_val(10);
    constexpr uint32_t RT = get_compile_time_arg_val(11);
    constexpr uint32_t NIN = get_compile_time_arg_val(12);
    constexpr uint32_t NOUT = get_compile_time_arg_val(13);
    constexpr bool has_res = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t gmask = get_compile_time_arg_val(15);
    constexpr uint32_t sem_id = get_compile_time_arg_val(16);
    constexpr auto x0_args = TensorAccessorArgs<17>();
    constexpr auto x1_args = TensorAccessorArgs<x0_args.next_compile_time_args_offset()>();
    constexpr auto g0_args = TensorAccessorArgs<x1_args.next_compile_time_args_offset()>();
    constexpr auto g1_args = TensorAccessorArgs<g0_args.next_compile_time_args_offset()>();
    constexpr auto g2_args = TensorAccessorArgs<g1_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<g2_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t f32_tile_words = 1024;
    constexpr uint32_t n_gamma = NIN == 2 ? 2 : NOUT;

    CircularBuffer cx0(cb_x0), cx1(cb_x1), cg0(cb_g0), cg1(cb_g1), cg2(cb_g2), cr(cb_res), cs(cb_scaler),
        crecv(cb_recv);
    // the receive tiles: zero (columns >= N and the padding must read as zero), filled after the exchange
    crecv.reserve_back(NIN * RT);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_recv));
    for (uint32_t i = 0; i < NIN * RT * f32_tile_words; ++i) {
        recv[i] = 0;
    }
    auto read_tiles = [&](auto& acc, CircularBuffer& cb, uint32_t first) {
        cb.reserve_back(per);
        const uint32_t base = get_write_ptr(cb.get_cb_id());
        for (uint32_t c = 0; c < per; ++c) {
            noc_async_read(acc.get_noc_addr(first + c), base + c * tile_bytes, tile_bytes);
        }
    };
    const auto s_x0 = TensorAccessor(x0_args, x0_addr, tile_bytes);
    read_tiles(s_x0, cx0, c0);
    if constexpr (NIN == 2) {
        const auto s_x1 = TensorAccessor(x1_args, x1_addr, tile_bytes);
        read_tiles(s_x1, cx1, c0);
    }
    if constexpr ((gmask & 1) != 0) {
        const auto s_g0 = TensorAccessor(g0_args, g0_addr, tile_bytes);
        read_tiles(s_g0, cg0, gc0);
    }
    if constexpr ((gmask & 2) != 0 && n_gamma > 1) {
        const auto s_g1 = TensorAccessor(g1_args, g1_addr, tile_bytes);
        read_tiles(s_g1, cg1, gc0);
    }
    if constexpr ((gmask & 4) != 0 && n_gamma > 2) {
        const auto s_g2 = TensorAccessor(g2_args, g2_addr, tile_bytes);
        read_tiles(s_g2, cg2, gc0);
    }
    if constexpr (has_res) {
        const auto s_res = TensorAccessor(res_args, res_addr, tile_bytes);
        read_tiles(s_res, cr, c0);
    }
    // unit scaler for the FPU row reduce: bf16 1.0 in the first row of every face, zeros elsewhere
    cs.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
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
    noc_async_read_barrier();
    cx0.push_back(per);
    if constexpr (NIN == 2) {
        cx1.push_back(per);
    }
    if constexpr ((gmask & 1) != 0) {
        cg0.push_back(per);
    }
    if constexpr ((gmask & 2) != 0 && n_gamma > 1) {
        cg1.push_back(per);
    }
    if constexpr ((gmask & 4) != 0 && n_gamma > 2) {
        cg2.push_back(per);
    }
    if constexpr (has_res) {
        cr.push_back(per);
    }
    // the N partial-sum blocks -> columns of the receive tiles
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait_min(sem, N);
    noc_semaphore_set(sem, 0);
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stage));
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t jt = j >> 5, jc = j & 31;
        for (uint32_t i = 0; i < NIN; ++i) {
            volatile tt_l1_ptr uint32_t* blk = stage + (j * NIN + i) * 32;
            volatile tt_l1_ptr uint32_t* tile = recv + (i * RT + jt) * f32_tile_words;
            for (uint32_t r = 0; r < 32; ++r) {
                tile[((r >> 4) * 2 + (jc >> 4)) * 256 + (r & 15) * 16 + (jc & 15)] = blk[r];
            }
        }
    }
    crecv.push_back(NIN * RT);
}
