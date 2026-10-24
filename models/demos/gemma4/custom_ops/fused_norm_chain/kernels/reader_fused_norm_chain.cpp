// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for FusedNormChainDecode: this core's `per` column tiles of x, the residual and the two full gamma tiles
// (rows replicated), the unit reduce scaler; then two exchange rounds: the N partial-sum blocks of round r (the
// writers deposit them in cb_stage / cb_stage2, slot = sender, 32 fp32 per slot) are scattered into the columns
// of the fp32 receive tile of that round (columns >= N stay zero). The semaphore counts both rounds (N, then 2N)
// and is reset once at the end for the next replay.
// compile-time args: [cb_x, cb_g1, cb_g2, cb_res, cb_scaler, cb_recv, cb_recv2, cb_stage, cb_stage2, per, N, RT,
//                     sem_id] + TensorAccessorArgs(x, g1, g2, res)
// runtime args: [x_addr, g1_addr, g2_addr, res_addr, c0, gc0]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t g1_addr = get_arg_val<uint32_t>(1);
    const uint32_t g2_addr = get_arg_val<uint32_t>(2);
    const uint32_t res_addr = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t gc0 = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_res = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(5);
    constexpr uint32_t cb_recv2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(7);
    constexpr uint32_t cb_stage2 = get_compile_time_arg_val(8);
    constexpr uint32_t per = get_compile_time_arg_val(9);
    constexpr uint32_t N = get_compile_time_arg_val(10);
    constexpr uint32_t RT = get_compile_time_arg_val(11);
    constexpr uint32_t sem_id = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto g1_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g2_args = TensorAccessorArgs<g1_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<g2_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t f32_tile_words = 1024;

    CircularBuffer cx(cb_x), cg1(cb_g1), cg2(cb_g2), cr(cb_res), cs(cb_scaler), crecv(cb_recv), crecv2(cb_recv2);
    auto zero_recv = [&](CircularBuffer& cb, uint32_t cbid) {
        cb.reserve_back(RT);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cbid));
        for (uint32_t i = 0; i < RT * f32_tile_words; ++i) {
            p[i] = 0;
        }
    };
    zero_recv(crecv, cb_recv);
    zero_recv(crecv2, cb_recv2);
    auto read_tiles = [&](auto& acc, CircularBuffer& cb, uint32_t first) {
        cb.reserve_back(per);
        const uint32_t base = get_write_ptr(cb.get_cb_id());
        for (uint32_t c = 0; c < per; ++c) {
            noc_async_read(acc.get_noc_addr(first + c), base + c * tile_bytes, tile_bytes);
        }
    };
    const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto s_g1 = TensorAccessor(g1_args, g1_addr, tile_bytes);
    const auto s_g2 = TensorAccessor(g2_args, g2_addr, tile_bytes);
    const auto s_res = TensorAccessor(res_args, res_addr, tile_bytes);
    read_tiles(s_x, cx, c0);
    read_tiles(s_res, cr, c0);
    read_tiles(s_g1, cg1, gc0);
    read_tiles(s_g2, cg2, gc0);
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
    cx.push_back(per);
    cr.push_back(per);
    cg1.push_back(per);
    cg2.push_back(per);
    // the exchange rounds: N partial-sum blocks -> the columns of the round's receive tile(s)
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    auto scatter = [&](uint32_t stage_cb, uint32_t recv_cb) {
        volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(stage_cb));
        volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(recv_cb));
        for (uint32_t j = 0; j < N; ++j) {
            const uint32_t jt = j >> 5, jc = j & 31;
            volatile tt_l1_ptr uint32_t* blk = stage + j * 32;
            volatile tt_l1_ptr uint32_t* tile = recv + jt * f32_tile_words;
            for (uint32_t r = 0; r < 32; ++r) {
                tile[((r >> 4) * 2 + (jc >> 4)) * 256 + (r & 15) * 16 + (jc & 15)] = blk[r];
            }
        }
    };
    noc_semaphore_wait_min(sem, N);
    scatter(cb_stage, cb_recv);
    crecv.push_back(RT);
    noc_semaphore_wait_min(sem, 2 * N);
    noc_semaphore_set(sem, 0);
    scatter(cb_stage2, cb_recv2);
    crecv2.push_back(RT);
}
