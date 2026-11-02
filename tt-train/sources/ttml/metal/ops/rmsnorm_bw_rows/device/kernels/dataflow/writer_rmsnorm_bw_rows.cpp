// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_bw_rows writer: this core's partial tile goes to slot s of cb::partials on every core of the row
// (itself included), each followed by a semaphore increment; then the dL_da blocks go to DRAM.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/rmsnorm_bw_rows_cbs.hpp"

namespace cb = rmsnorm_bw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t exchange_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t Ht = get_compile_time_arg_val(5);  // tile rows of the input: dL_da^T's tile-row stride
constexpr auto out_args = TensorAccessorArgs<6>();
#ifdef DX_TRANSPOSED
constexpr auto out_t_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t out_t_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t s = get_arg_val<uint32_t>(arg++);
    const uint32_t coords_base = arg;  // S pairs (noc x, noc y) of the row's cores

    const uint32_t tile_bytes = get_tile_size(cb::dx);
    const uint32_t f32_tile_bytes = get_tile_size(cb::partial_out);
    const auto out_acc = TensorAccessor(out_args, out_addr);
#ifdef DX_TRANSPOSED
    const auto out_t_acc = TensorAccessor(out_t_args, out_t_addr);
#endif

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

    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
        cb_wait_front(cb::dx, block);
        uint32_t l1 = get_read_ptr(cb::dx);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write_page(r * Wt + col0 + c + j, out_acc, l1);
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb::dx, block);
#ifdef DX_TRANSPOSED
        // dL_da^T: this block's tiles go to (col0 + c + j, r) of the [C, S] output (iteration 37)
        cb_wait_front(cb::dx_t, block);
        uint32_t l1_t = get_read_ptr(cb::dx_t);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write_page((col0 + c + j) * Ht + r, out_t_acc, l1_t);
            l1_t += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb::dx_t, block);
#endif
    }
    noc_async_atomic_barrier();
}
