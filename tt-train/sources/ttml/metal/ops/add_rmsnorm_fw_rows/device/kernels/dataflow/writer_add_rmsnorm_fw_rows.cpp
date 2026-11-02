// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rmsnorm_fw_rows writer: the sum slice goes to DRAM from the resident cb::sum as pass 1 produces it (read, not
// popped: the compute re-reads it in pass 2); this core's partial tile goes to slot s of cb::partials on every core
// of the row (itself included), each followed by a semaphore increment; slice 0 writes the row's rms tile; then the
// output blocks.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/device/kernels/add_rmsnorm_fw_rows_cbs.hpp"

namespace cb = add_rmsnorm_fw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t exchange_sem_id = get_compile_time_arg_val(4);
constexpr auto sum_args = TensorAccessorArgs<5>();
constexpr auto out_args = TensorAccessorArgs<sum_args.next_compile_time_args_offset()>();
constexpr auto rms_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t sum_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t rms_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);
    const uint32_t coords_base = arg;  // S pairs (noc x, noc y) of the row's cores

    const uint32_t tile_bytes = get_tile_size(cb::out);
    const uint32_t f32_tile_bytes = get_tile_size(cb::partial_out);
    const auto sum_acc = TensorAccessor(sum_args, sum_addr);
    const auto out_acc = TensorAccessor(out_args, out_addr);
    const auto rms_acc = TensorAccessor(rms_args, rms_addr);

    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
    const uint32_t page = r * Wt + col0;

    // The sum slice, written group by group as pass 1 produces it (cb::sum is resident: cumulative waits, no pops here;
    // the compute pops it at the very end).
    {
        uint32_t l1 = get_read_ptr(cb::sum);
        for (uint32_t c = 0; c < ncols; c += block) {  // the pass-1 group is the output block (4 tiles)
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            cb_wait_front(cb::sum, c + n);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_write_page(page + c + j, sum_acc, l1);
                l1 += tile_bytes;
            }
        }
    }

    // Partial exchange. cb::partials is never reserved by this RISC, so its write pointer is the buffer start.
    cb_wait_front(cb::partial_out, 1);
    const uint32_t src = get_read_ptr(cb::partial_out);
    const uint32_t slot_addr = get_write_ptr(cb::partials) + s * f32_tile_bytes;
    const uint64_t sem_local = get_semaphore(exchange_sem_id);
    for (uint32_t j = 0; j < S; ++j) {
        const uint32_t x = get_arg_val<uint32_t>(coords_base + 2U * j);
        const uint32_t y = get_arg_val<uint32_t>(coords_base + 2U * j + 1U);
        noc_async_write(src, get_noc_addr(x, y, slot_addr), f32_tile_bytes);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < S; ++j) {
        const uint32_t x = get_arg_val<uint32_t>(coords_base + 2U * j);
        const uint32_t y = get_arg_val<uint32_t>(coords_base + 2U * j + 1U);
        noc_semaphore_inc(get_noc_addr(x, y, sem_local), 1);
    }
    cb_pop_front(cb::partial_out, 1);

    if (s == 0) {
        cb_wait_front(cb::rms_out, 1);
        noc_async_write_page(r, rms_acc, get_read_ptr(cb::rms_out));
        noc_async_write_barrier();
        cb_pop_front(cb::rms_out, 1);
    }

    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
        cb_wait_front(cb::out, block);
        uint32_t l1 = get_read_ptr(cb::out);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write_page(page + c + j, out_acc, l1);
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb::out, block);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
