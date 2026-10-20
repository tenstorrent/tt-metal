// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Reader (NCRISC) of the fused pre-block: this core's inputs (the 4 stream tiles and the 4 projection tiles of
// its column, the norm-weight tile, a zero tile, the Sinkhorn constants on the coordinator) and the receive
// side of the cross-core sync: a leader waits for its column's partials (semaphore ARRIVE1, counted by the
// senders) and hands them to the compute kernel; the coordinator waits for the leaders' sums (ARRIVE2); every
// core waits for the pre tile (SEM_PRE) and the rsqrt tile (SEM_RS2) -- fanned out by the coordinator to the
// leaders and by each leader (this kernel) to its column. Semaphores are reset by their waiter, so the program
// replays under trace.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(1);
    const uint32_t consts_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t mask_addr = get_arg_val<uint32_t>(9);
    const uint32_t count = get_arg_val<uint32_t>(4);  // column tiles of this core (0: idle)
    const uint32_t c0 = get_arg_val<uint32_t>(5);     // its first column tile
    const bool worker = count != 0;
    const bool leader = get_arg_val<uint32_t>(6) != 0;
    const bool coordinator = get_arg_val<uint32_t>(7) != 0;
    const uint32_t my_noc_x = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(1);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gather1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gather2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_w = get_compile_time_arg_val(7);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(8);
    constexpr uint32_t n = get_compile_time_arg_val(9);
    constexpr uint32_t GY = get_compile_time_arg_val(10);
    constexpr uint32_t GX = get_compile_time_arg_val(11);
    constexpr uint32_t Ct = get_compile_time_arg_val(12);
    constexpr uint32_t sem_arrive1 = get_compile_time_arg_val(13);
    constexpr uint32_t sem_arrive2 = get_compile_time_arg_val(14);
    constexpr uint32_t sem_pre = get_compile_time_arg_val(15);
    constexpr uint32_t sem_rs2 = get_compile_time_arg_val(16);
    constexpr uint32_t NOC_Y_BASE = 17;                                   // NOC_Y_BASE + y: the NoC y of logical row y
    constexpr uint32_t frep = get_compile_time_arg_val(NOC_Y_BASE + GY);  // fn tiles per stream tile (1 or 3)
    constexpr uint32_t cb_mask = get_compile_time_arg_val(NOC_Y_BASE + GY + 1);  // the lane-mask tile (frep == 3)
    constexpr auto x_args = TensorAccessorArgs<NOC_Y_BASE + GY + 2>();
    constexpr auto fn_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<consts_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t x_page = get_local_cb_interface(cb_x).fifo_page_size;
    const uint32_t fn_page = get_local_cb_interface(cb_fn).fifo_page_size;
    const uint32_t w_page = get_local_cb_interface(cb_w).fifo_page_size;
    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;
    const uint32_t tile_bytes = get_local_cb_interface(cb_pre).fifo_page_size;

    const auto s_x = TensorAccessor(x_args, x_addr);
    const auto s_fn = TensorAccessor(fn_args, fn_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_mask = TensorAccessor(mask_args, mask_addr);

    Noc noc;
    CircularBuffer cbx(cb_x), cbfn(cb_fn), cbzero(cb_zero), cbg1(cb_gather1), cbg2(cb_gather2), cbpre(cb_pre),
        cbrs2(cb_rs2), cbw(cb_w), cbc(cb_consts);

    volatile tt_l1_ptr uint32_t* p_arrive1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_arrive1));
    volatile tt_l1_ptr uint32_t* p_arrive2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_arrive2));
    volatile tt_l1_ptr uint32_t* p_pre = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pre));
    volatile tt_l1_ptr uint32_t* p_rs2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rs2));

    // ---- inputs
    if (worker) {
        cbx.reserve_back(n * count);
        cbfn.reserve_back(n * count * frep);
        cbw.reserve_back(count);
        for (uint32_t col = 0; col < count; ++col) {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t t = col * n + i;
                noc.async_read(s_x, cbx, x_page, {.page_id = i * Ct + c0 + col}, {.offset_bytes = t * x_page});
                for (uint32_t q = 0; q < frep; ++q) {
                    noc.async_read(
                        s_fn,
                        cbfn,
                        fn_page,
                        {.page_id = (i * Ct + c0 + col) * frep + q},
                        {.offset_bytes = (t * frep + q) * fn_page});
                }
            }
            noc.async_read(s_w, cbw, w_page, {.page_id = c0 + col}, {.offset_bytes = col * w_page});
        }
        if constexpr (frep > 1) {
            CircularBuffer cbm(cb_mask);
            cbm.reserve_back(1);
            noc.async_read(s_mask, cbm, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbm.push_back(1);
        }
        noc.async_read_barrier();
        cbx.push_back(n * count);
        cbfn.push_back(n * count * frep);
        cbw.push_back(count);
    }
    cbzero.reserve_back(1);
    noc.async_write_zeros(cbzero, tile_bytes);
    noc.write_zeros_l1_barrier();
    cbzero.push_back(1);
    if (coordinator) {
        cbc.reserve_back(8);
        for (uint32_t i = 0; i < 8; ++i) {
            noc.async_read(s_consts, cbc, consts_page, {.page_id = i}, {.offset_bytes = i * consts_page});
        }
        noc.async_read_barrier();
        cbc.push_back(8);
    }

    // a tile that arrived at this core (the sender wrote the cb's base) fanned out to the rest of the column
    auto forward_to_column = [&](uint32_t cb, uint32_t sem_id) {
        const uint32_t src = get_write_ptr(cb);  // the cb base (nothing reserved on this RISC yet)
        const uint32_t sem_addr = get_semaphore(sem_id);
        for (uint32_t y = 1; y < GY; ++y) {
            const uint32_t ny = kernel_compile_time_args[NOC_Y_BASE + y];
            noc_async_write(src, get_noc_addr(my_noc_x, ny, src), tile_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t y = 1; y < GY; ++y) {
            const uint32_t ny = kernel_compile_time_args[NOC_Y_BASE + y];
            noc_semaphore_inc(get_noc_addr(my_noc_x, ny, sem_addr), 1);
        }
    };

    // ---- round 1: partials -> leaders -> coordinator
    if (leader) {
        cbg1.reserve_back(GY);
        noc_semaphore_wait_min(p_arrive1, GY);
        noc_semaphore_set(p_arrive1, 0);
        cbg1.push_back(GY);
    }
    if (coordinator) {
        cbg2.reserve_back(GX);
        noc_semaphore_wait_min(p_arrive2, GX);
        noc_semaphore_set(p_arrive2, 0);
        cbg2.push_back(GX);
    }
    // ---- the pre tile: coordinator -> leaders (writer) -> column (this kernel)
    cbpre.reserve_back(1);
    noc_semaphore_wait_min(p_pre, 1);
    noc_semaphore_set(p_pre, 0);
    if (leader) {
        forward_to_column(cb_pre, sem_pre);
    }
    cbpre.push_back(1);
    // ---- round 2: sums of h^2 -> leaders -> coordinator
    if (leader) {
        cbg1.reserve_back(GY);
        noc_semaphore_wait_min(p_arrive1, GY);
        noc_semaphore_set(p_arrive1, 0);
        cbg1.push_back(GY);
    }
    if (coordinator) {
        cbg2.reserve_back(GX);
        noc_semaphore_wait_min(p_arrive2, GX);
        noc_semaphore_set(p_arrive2, 0);
        cbg2.push_back(GX);
    }
    // ---- the rsqrt tile
    cbrs2.reserve_back(1);
    noc_semaphore_wait_min(p_rs2, 1);
    noc_semaphore_set(p_rs2, 0);
    if (leader) {
        forward_to_column(cb_rs2, sem_rs2);
    }
    cbrs2.push_back(1);
}
