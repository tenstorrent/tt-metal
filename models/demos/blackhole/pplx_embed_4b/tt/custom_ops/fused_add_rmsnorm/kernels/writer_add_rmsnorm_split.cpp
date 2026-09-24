// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Row-split add+RMSNorm writer: writes the sum slice, exchanges this core's partial mean-square with the
// other cores of its row (NoC write into slot k of their CB 8 + semaphore increment), then writes the
// normalised slice.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t sum_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wc = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(3);
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_sum = 16, cb_out = 17, cb_part = 7, cb_parts = 8;
    const auto ss = TensorAccessor(s_args, sum_addr);
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t ts = get_tile_size(cb_sum), to = get_tile_size(cb_out), tp = get_tile_size(cb_part);
    const uint32_t base = row * Wt + k * Wc;

    Noc noc;
    CircularBuffer cs(cb_sum), co(cb_out), cp(cb_part), cparts(cb_parts);

    // 1. residual sum slice
    cs.wait_front(Wc);
    for (uint32_t j = 0; j < Wc; ++j) {
        noc.async_write(cs, ss, ts, {.offset_bytes = j * ts}, {.page_id = base + j});
    }

    // 2. partial mean-square exchange within the row group (all R cores, including this one)
    cparts.reserve_back(R);  // CB 8 is used once per core: its write pointer is the CB base on every core
    const uint32_t parts_base = get_write_ptr(cb_parts);
    cp.wait_front(1);
    const uint32_t part_src = get_read_ptr(cb_part);
    const uint32_t sem_l1 = get_semaphore(SEM_ID);
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t vx = get_arg_val<uint32_t>(4 + 2 * j);
        const uint32_t vy = get_arg_val<uint32_t>(5 + 2 * j);
        noc_async_write(part_src, get_noc_addr(vx, vy, parts_base + k * tp), tp);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t vx = get_arg_val<uint32_t>(4 + 2 * j);
        const uint32_t vy = get_arg_val<uint32_t>(5 + 2 * j);
        noc_semaphore_inc(get_noc_addr(vx, vy, sem_l1), 1);
    }
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
    noc_semaphore_wait(sem_ptr, R);
    cparts.push_back(R);
    cp.pop_front(1);

    // 3. normalised slice
    co.wait_front(Wc);
    for (uint32_t j = 0; j < Wc; ++j) {
        noc.async_write(co, so, to, {.offset_bytes = j * to}, {.page_id = base + j});
    }
    noc.async_write_barrier();
    cs.pop_front(Wc);
    co.pop_front(Wc);
}
