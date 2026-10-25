// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 router matmul, writer.  K-slice cores (s > 0) send their fp32 partial tile into slot s-1 of the reducer
// core's cb_red and bump its semaphore; the reducer (s == 0, one per column tile n) loads its bias tile, waits for the
// S-1 partials, lets compute reduce + sqrt(softplus) + bias, and writes the score and rank tiles (page n).

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    DeviceZoneScopedN("RW_ALL");
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_red = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(3);
    constexpr uint32_t cb_score = get_compile_time_arg_val(4);
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr uint32_t SEM = get_compile_time_arg_val(6);
    constexpr auto b_args = TensorAccessorArgs<7>();
    constexpr auto r_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t rx = get_arg_val<uint32_t>(2);  // reducer NOC coordinates
    const uint32_t ry = get_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    experimental::CB cbpart(cb_part), cbred(cb_red), cbbias(cb_bias), cbrank(cb_rank), cbscore(cb_score);
    const uint32_t red_base = get_write_ptr(cb_red);
    const uint32_t sem_addr = get_semaphore(SEM);

    if (s != 0) {
        cbpart.wait_front(1);
        const uint32_t src = cbpart.get_read_ptr();
        noc_async_write(src, get_noc_addr(rx, ry, red_base + (s - 1) * TILE), TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, sem_addr), 1);
        noc_async_atomic_barrier();
        cbpart.pop_front(1);
    } else {
        const auto b_acc = TensorAccessor(b_args, b_addr, TILE);
        const auto r_acc = TensorAccessor(r_args, r_addr, TILE);
        const auto s_acc = TensorAccessor(s_args, s_addr, TILE);
        cbbias.reserve_back(1);
        noc.async_read(b_acc, cbbias, TILE, {.page_id = n, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbbias.push_back(1);
        if constexpr (S > 1) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            noc_semaphore_wait_min(sem, S - 1);
            *sem = 0;
            cbred.reserve_back(S - 1);
            cbred.push_back(S - 1);
        }
        cbscore.wait_front(1);
        cbrank.wait_front(1);
        noc.async_write(cbrank, r_acc, TILE, {.offset_bytes = 0}, {.page_id = n, .offset_bytes = 0});
        noc.async_write(cbscore, s_acc, TILE, {.offset_bytes = 0}, {.page_id = n, .offset_bytes = 0});
        noc.async_write_barrier();
        cbrank.pop_front(1);
        cbscore.pop_front(1);
    }
}
