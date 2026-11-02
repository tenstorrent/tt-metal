// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer of the fused DSA decode state maintenance. Role A: the q_resid tiles of its tile row. Role B:
// the user's updated latent page, ring k/g blocks and pool block, in place.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_qout = 14, cb_lat_new = 15, cb_ring_new = 16, cb_pool_new = 17, cb_meta_w = 29;

void kernel_main() {
    constexpr uint32_t Lq_t = get_compile_time_arg_val(0);
    constexpr uint32_t L_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);

    constexpr auto q_a = TensorAccessorArgs<3>();
    constexpr auto lat_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto rk_a = TensorAccessorArgs<lat_a.next_compile_time_args_offset()>();
    constexpr auto rg_a = TensorAccessorArgs<rk_a.next_compile_time_args_offset()>();
    constexpr auto pool_a = TensorAccessorArgs<rg_a.next_compile_time_args_offset()>();

    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t idx = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t lat_addr = get_arg_val<uint32_t>(3);
    const uint32_t rk_addr = get_arg_val<uint32_t>(4);
    const uint32_t rg_addr = get_arg_val<uint32_t>(5);
    const uint32_t pool_addr = get_arg_val<uint32_t>(6);

    const uint32_t t16 = get_tile_size(cb_qout);
    const auto q_acc = TensorAccessor(q_a, q_addr, t16);
    const auto lat_acc = TensorAccessor(lat_a, lat_addr, t16);
    const auto rk_acc = TensorAccessor(rk_a, rk_addr, t16);
    const auto rg_acc = TensorAccessor(rg_a, rg_addr, t16);
    const auto pool_acc = TensorAccessor(pool_a, pool_addr, t16);

    Noc noc;
    auto write_tiles = [&](uint32_t cb_id, const auto& acc, uint32_t base, uint32_t n, uint32_t src_off = 0) {
        CircularBuffer cb(cb_id);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_write(cb, acc, t16, {.offset_bytes = (src_off + t) * t16}, {.page_id = base + t});
        }
        noc.async_write_barrier();
    };

    if (role == 0) {
        const uint32_t tb = idx;
        CircularBuffer q(cb_qout);
        for (uint32_t c = 0; c < Lq_t; c += 4) {
            q.wait_front(4);
            write_tiles(cb_qout, q_acc, tb * Lq_t + c, 4);
            q.pop_front(4);
        }
        return;
    }

    CircularBuffer mw(cb_meta_w);
    mw.wait_front(1);
    const CoreLocalMem<volatile uint32_t> w(mw.get_read_ptr());
    const uint32_t lat_page = w[0];
    const uint32_t b = w[1];
    const uint32_t pool_blk = w[2];

    CircularBuffer lat(cb_lat_new), ring(cb_ring_new), pool(cb_pool_new);
    lat.wait_front(L_t);
    write_tiles(cb_lat_new, lat_acc, lat_page * L_t, L_t);
    lat.pop_front(L_t);
    ring.wait_front(2 * D_t);
    write_tiles(cb_ring_new, rk_acc, b * D_t, D_t, 0);
    write_tiles(cb_ring_new, rg_acc, b * D_t, D_t, D_t);
    ring.pop_front(2 * D_t);
    pool.wait_front(D_t);
    write_tiles(cb_pool_new, pool_acc, pool_blk * D_t, D_t);
    pool.pop_front(D_t);
    mw.pop_front(1);
}
