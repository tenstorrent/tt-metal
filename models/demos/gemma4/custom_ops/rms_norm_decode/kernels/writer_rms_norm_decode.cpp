// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer for RmsNormDecode: sends this core's partial-sum tile to slot `me` of every core's cb_recv (itself
// included) and bumps their semaphores; then writes the normalized tiles to pages c0..c0+per-1.
// compile-time args: [cb_part, cb_recv, cb_out, per, N, sem_id] + TensorAccessorArgs(out)
// runtime args: [out_addr, c0, me, (noc_x, noc_y) x N]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t me = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t per = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t f32_tile = 4096;

    CircularBuffer cp(cb_part), co(cb_out);
    cp.wait_front(1);
    const uint32_t src = get_read_ptr(cb_part);
    const uint32_t recv_base = get_write_ptr(cb_recv);  // same L1 address on every core
    const uint32_t sem_addr = get_semaphore(sem_id);
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t nx = get_arg_val<uint32_t>(3 + 2 * j), ny = get_arg_val<uint32_t>(4 + 2 * j);
        noc_async_write(src, get_noc_addr(nx, ny, recv_base + me * f32_tile), f32_tile);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < N; ++j) {
        const uint32_t nx = get_arg_val<uint32_t>(3 + 2 * j), ny = get_arg_val<uint32_t>(4 + 2 * j);
        noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
    }
    noc_async_atomic_barrier();
    cp.pop_front(1);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_bytes);
    co.wait_front(per);
    const uint32_t ao = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < per; ++c) {
        noc_async_write(ao + c * tile_bytes, s_out.get_noc_addr(c0 + c), tile_bytes);
    }
    noc_async_write_barrier();
    co.pop_front(per);
}
