// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for RmsNormDecode: this core's `per` x column tiles (pages c0..c0+per-1) into cb_x, the matching gamma
// tiles into cb_g, the unit reduce scaler; then it waits for the N partial-sum tiles the writers of all cores
// deposit in cb_recv (slot = sender) and publishes them, resetting the semaphore for the next replay.
// compile-time args: [cb_x, cb_g, cb_scaler, cb_recv, per, N, has_gamma, sem_id] + TensorAccessorArgs(x, gamma)
// runtime args: [x_addr, gamma_addr, c0]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t per = get_compile_time_arg_val(4);
    constexpr uint32_t N = get_compile_time_arg_val(5);
    constexpr bool has_gamma = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;

    CircularBuffer cx(cb_x), cg(cb_g), cs(cb_scaler), cr(cb_recv);
    cr.reserve_back(N);  // the remote writers land their partials at the base of cb_recv
    const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
    cx.reserve_back(per);
    const uint32_t ax = get_write_ptr(cb_x);
    for (uint32_t c = 0; c < per; ++c) {
        noc_async_read(s_x.get_noc_addr(c0 + c), ax + c * tile_bytes, tile_bytes);
    }
    if constexpr (has_gamma) {
        const auto s_g = TensorAccessor(g_args, g_addr, tile_bytes);
        cg.reserve_back(per);
        const uint32_t ag = get_write_ptr(cb_g);
        for (uint32_t c = 0; c < per; ++c) {
            noc_async_read(s_g.get_noc_addr(c0 + c), ag + c * tile_bytes, tile_bytes);
        }
    }
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
    if constexpr (has_gamma) {
        cg.push_back(per);
    }
    // the N partials
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait_min(sem, N);
    noc_semaphore_set(sem, 0);
    cr.push_back(N);
}
