// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Row-split add+RMSNorm writer: writes the sum slice, exchanges this core's partial mean-square with the
// other cores of its row (NoC write into slot k of their CB 8 + semaphore increment), then writes the
// normalised slice. The normalised output can be split by rows over up to four tensors of split_row tile-rows each:
// row goes to part row / split_row at row % split_row (split_row = 0xFFFFFFFF: one tensor).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t sum_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_waves = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
    const uint32_t row_stride = get_arg_val<uint32_t>(4);
    const uint32_t k = get_arg_val<uint32_t>(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wc = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t SEM_ID = get_compile_time_arg_val(3);
    const uint32_t split_row = get_arg_val<uint32_t>(6 + 2 * R);
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto o1_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto o2_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();
    constexpr auto o3_args = TensorAccessorArgs<o2_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_sum = 16, cb_out = 17, cb_part = 7, cb_parts = 8;
    const auto ss = TensorAccessor(s_args, sum_addr);
    const auto so = TensorAccessor(o_args, out_addr);
    const auto so1 = TensorAccessor(o1_args, get_arg_val<uint32_t>(7 + 2 * R));
    const auto so2 = TensorAccessor(o2_args, get_arg_val<uint32_t>(8 + 2 * R));
    const auto so3 = TensorAccessor(o3_args, get_arg_val<uint32_t>(9 + 2 * R));
    const uint32_t ts = get_tile_size(cb_sum), to = get_tile_size(cb_out), tp = get_tile_size(cb_part);
    Noc noc;
    CircularBuffer cs(cb_sum), co(cb_out), cp(cb_part), cparts(cb_parts);
    // CB 8 never wraps (waves_max * R slots, each used once), so slot (w, k) sits at a fixed offset from the
    // CB base on every core of the group; the semaphore counts every partial received and only grows.
    const uint32_t parts_base = get_write_ptr(cb_parts);
    const uint32_t sem_l1 = get_semaphore(SEM_ID);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);

    for (uint32_t w = 0; w < n_waves; ++w) {
        const uint32_t row = row0 + w * row_stride;
        const uint32_t base = row * Wt + k * Wc;

        // 1. residual sum slice
        cs.wait_front(Wc);
        for (uint32_t j = 0; j < Wc; ++j) {
            noc.async_write(cs, ss, ts, {.offset_bytes = j * ts}, {.page_id = base + j});
        }

        // 2. partial mean-square exchange within the row group (all R cores, including this one)
        cparts.reserve_back(R);
        cp.wait_front(1);
        const uint32_t part_src = get_read_ptr(cb_part);
        const uint32_t slot = parts_base + (w * R + k) * tp;
        for (uint32_t j = 0; j < R; ++j) {
            const uint32_t vx = get_arg_val<uint32_t>(6 + 2 * j);
            const uint32_t vy = get_arg_val<uint32_t>(7 + 2 * j);
            noc_async_write(part_src, get_noc_addr(vx, vy, slot), tp);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < R; ++j) {
            const uint32_t vx = get_arg_val<uint32_t>(6 + 2 * j);
            const uint32_t vy = get_arg_val<uint32_t>(7 + 2 * j);
            noc_semaphore_inc(get_noc_addr(vx, vy, sem_l1), 1);
        }
        noc_semaphore_wait_min(sem_ptr, R * (w + 1));
        cparts.push_back(R);
        cp.pop_front(1);

        // 3. normalised slice
        co.wait_front(Wc);
        const uint32_t part = row < split_row ? 0 : row / split_row;
        const uint32_t base_p = part == 0 ? base : (row - part * split_row) * Wt + k * Wc;
        for (uint32_t j = 0; j < Wc; ++j) {
            if (part == 0) {
                noc.async_write(co, so, to, {.offset_bytes = j * to}, {.page_id = base_p + j});
            } else if (part == 1) {
                noc.async_write(co, so1, to, {.offset_bytes = j * to}, {.page_id = base_p + j});
            } else if (part == 2) {
                noc.async_write(co, so2, to, {.offset_bytes = j * to}, {.page_id = base_p + j});
            } else {
                noc.async_write(co, so3, to, {.offset_bytes = j * to}, {.page_id = base_p + j});
            }
        }
        noc.async_write_barrier();
        cs.pop_front(Wc);
        co.pop_front(Wc);
    }
}
