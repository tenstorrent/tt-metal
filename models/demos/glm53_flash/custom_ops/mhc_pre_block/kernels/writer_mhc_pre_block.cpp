// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Writer (BRISC) of the fused pre-block: the send side of the cross-core sync. Every core sends its projection
// partial (an idle core the zero tile) into slot y of its column leader's gather buffer and counts it on the
// leader's ARRIVE1; a leader sends its column sum into slot x of the coordinator's gather buffer (ARRIVE2); the
// coordinator sends the pre tile to every leader (SEM_PRE; the leaders' readers fan it out) and writes the
// post / comb lane tiles to DRAM. The same again for the rows' sums of h^2 and the rsqrt tile (SEM_RS2).
// Finally a worker writes its bf16 hn tile.
void kernel_main() {
    const uint32_t post_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);
    const uint32_t hn_addr = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);  // column tiles of this core (0: idle)
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const bool worker = count != 0;
    const bool leader = get_arg_val<uint32_t>(5) != 0;
    const bool coordinator = get_arg_val<uint32_t>(6) != 0;
    const uint32_t leader_x = get_arg_val<uint32_t>(7);  // NoC coords of this column's leader
    const uint32_t leader_y = get_arg_val<uint32_t>(8);
    const uint32_t my_y = get_arg_val<uint32_t>(9);      // logical row: slot in the leader's gather buffer
    const uint32_t coord_x = get_arg_val<uint32_t>(10);  // NoC coords of the coordinator
    const uint32_t coord_y = get_arg_val<uint32_t>(11);
    const uint32_t my_x = get_arg_val<uint32_t>(12);  // logical column: slot in the coordinator's gather buffer

    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t cb_coef_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_hsq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rs2out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rs2 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gather1 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gather2 = get_compile_time_arg_val(9);
    constexpr uint32_t n = get_compile_time_arg_val(10);
    constexpr uint32_t GY = get_compile_time_arg_val(11);
    constexpr uint32_t GX = get_compile_time_arg_val(12);
    constexpr uint32_t Ct = get_compile_time_arg_val(13);
    constexpr uint32_t sem_arrive1 = get_compile_time_arg_val(14);
    constexpr uint32_t sem_arrive2 = get_compile_time_arg_val(15);
    constexpr uint32_t sem_pre = get_compile_time_arg_val(16);
    constexpr uint32_t sem_rs2 = get_compile_time_arg_val(17);
    constexpr uint32_t NOC_X_BASE = 18;           // NOC_X_BASE + x: the NoC x of logical column x
    constexpr uint32_t NOC_Y0 = NOC_X_BASE + GX;  // the NoC y of logical row 0 (the leaders)
    constexpr auto post_args = TensorAccessorArgs<NOC_Y0 + 1>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr auto hn_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_local_cb_interface(cb_part).fifo_page_size;
    const uint32_t out_page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);
    const auto s_hn = TensorAccessor(hn_args, hn_addr);

    Noc noc;
    CircularBuffer cbpart(cb_part), cbzero(cb_zero), cbcoef(cb_coef_out), cbhsq(cb_hsq), cbrs2out(cb_rs2out),
        cbo(cb_out);
    // the receive buffers' bases are the same on every core (identical CB layout, nothing reserved on this RISC)
    const uint32_t gather1_base = get_write_ptr(cb_gather1);
    const uint32_t gather2_base = get_write_ptr(cb_gather2);
    const uint32_t pre_base = get_write_ptr(cb_pre);
    const uint32_t rs2_base = get_write_ptr(cb_rs2);
    const uint32_t sem1_addr = get_semaphore(sem_arrive1);
    const uint32_t sem2_addr = get_semaphore(sem_arrive2);
    const uint32_t sem_pre_addr = get_semaphore(sem_pre);
    const uint32_t sem_rs2_addr = get_semaphore(sem_rs2);

    // this core's contribution (from `cb`, or the zero tile when idle) -> slot my_y of the leader's buffer
    auto send_to_leader = [&](uint32_t cb) {
        uint32_t src;
        if (worker) {
            CircularBuffer src_cb(cb);
            src_cb.wait_front(1);
            src = get_read_ptr(cb);
        } else {
            cbzero.wait_front(1);
            src = get_read_ptr(cb_zero);
        }
        noc_async_write(src, get_noc_addr(leader_x, leader_y, gather1_base + my_y * tile_bytes), tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, sem1_addr), 1);
        if (worker) {
            CircularBuffer src_cb(cb);
            src_cb.pop_front(1);
        }
    };
    // the column sum (cb_part) -> slot my_x of the coordinator's buffer
    auto send_to_coordinator = [&]() {
        cbpart.wait_front(1);
        noc_async_write(
            get_read_ptr(cb_part), get_noc_addr(coord_x, coord_y, gather2_base + my_x * tile_bytes), tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(coord_x, coord_y, sem2_addr), 1);
        cbpart.pop_front(1);
    };
    // one tile at l1 `src` -> every leader's `dst_base`, then their semaphore
    auto fan_out_to_leaders = [&](uint32_t src, uint32_t dst_base, uint32_t sem_addr) {
        for (uint32_t x = 0; x < GX; ++x) {
            const uint32_t nx = kernel_compile_time_args[NOC_X_BASE + x];
            noc_async_write(src, get_noc_addr(nx, get_compile_time_arg_val(NOC_Y0), dst_base), tile_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t x = 0; x < GX; ++x) {
            const uint32_t nx = kernel_compile_time_args[NOC_X_BASE + x];
            noc_semaphore_inc(get_noc_addr(nx, get_compile_time_arg_val(NOC_Y0), sem_addr), 1);
        }
    };

    // ---- round 1: projection partials
    send_to_leader(cb_part);
    if (leader) {
        send_to_coordinator();
    }
    if (coordinator) {
        cbcoef.wait_front(n + 2);  // pre, post, comb_0..n-1 lane tiles
        const uint32_t base = get_read_ptr(cb_coef_out);
        fan_out_to_leaders(base, pre_base, sem_pre_addr);
        noc.async_write(cbcoef, s_post, tile_bytes, {.offset_bytes = tile_bytes}, {.page_id = 0});
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_write(cbcoef, s_comb, tile_bytes, {.offset_bytes = (2 + j) * tile_bytes}, {.page_id = j});
        }
        noc.async_write_barrier();
        cbcoef.pop_front(n + 2);
    }
    // ---- round 2: rows' sums of h^2
    send_to_leader(cb_hsq);
    if (leader) {
        send_to_coordinator();
    }
    if (coordinator) {
        cbrs2out.wait_front(1);
        fan_out_to_leaders(get_read_ptr(cb_rs2out), rs2_base, sem_rs2_addr);
        cbrs2out.pop_front(1);
    }
    // ---- the normalised sublayer input
    for (uint32_t col = 0; col < count; ++col) {
        cbo.wait_front(1);
        noc.async_write(cbo, s_hn, out_page, {.offset_bytes = 0}, {.page_id = c0 + col});
        noc.async_write_barrier();
        cbo.pop_front(1);
    }
}
