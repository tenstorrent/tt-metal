// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// skinny_contract reader. small^T's Kt tiles: read from DRAM once by the sender core (0, 0) and multicast into
// every core's cb_small (iteration 40: the 110-fold redundant read of the same 64 KB cost 15-18 us per call), with
// the receivers' handshake (each receiver invalidates its flag and counts itself in at the sender, the sender
// multicasts the data, then the VALID flag). Then per chunk of up to `cols` adjacent output columns [c, c + ncols)
// of the core's column range the K rows big[t, c .. c + ncols), t < Kt, one row of `cols` tiles per push (a
// shorter chunk still pushes cols).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_small = 0;
constexpr uint32_t cb_big = 1;
constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Nt = get_compile_time_arg_val(1);
constexpr uint32_t cols = get_compile_time_arg_val(2);
constexpr uint32_t prefetch = get_compile_time_arg_val(3);  // K rows per read batch
constexpr uint32_t sender_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(5);
constexpr uint32_t mcast_x_start = get_compile_time_arg_val(6);  // the grid rectangle, NoC coordinates
constexpr uint32_t mcast_y_start = get_compile_time_arg_val(7);
constexpr uint32_t mcast_x_end = get_compile_time_arg_val(8);
constexpr uint32_t mcast_y_end = get_compile_time_arg_val(9);
constexpr uint32_t num_receivers = get_compile_time_arg_val(10);  // every core of the rectangle but the sender
constexpr auto small_args = TensorAccessorArgs<11>();
constexpr auto big_args = TensorAccessorArgs<small_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t small_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t big_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t n_begin = get_arg_val<uint32_t>(arg++);
    const uint32_t n_count = get_arg_val<uint32_t>(arg++);
    const uint32_t tile_bytes = get_tile_size(cb_big);
    [[maybe_unused]] const auto small_acc = TensorAccessor(small_args, small_addr, tile_bytes);  // the sender's
    const auto big_acc = TensorAccessor(big_args, big_addr, tile_bytes);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sender_sem_id));
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(receiver_sem_id));

    cb_reserve_back(cb_small, Kt);
    const uint32_t l1_small = get_write_ptr(cb_small);
#ifdef MCAST_SENDER
    {
        uint32_t l1 = l1_small;
        for (uint32_t t = 0; t < Kt; ++t) {
            noc_async_read_page(t, small_acc, l1);
            l1 += tile_bytes;
        }
    }
    noc_async_read_barrier();
    // every receiver has its cb_small reserved and its flag INVALID
    noc_semaphore_wait(sender_sem_ptr, num_receivers);
    noc_semaphore_set(sender_sem_ptr, 0);
    noc_async_write_multicast(
        l1_small,
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, l1_small),
        Kt * tile_bytes,
        num_receivers);
#ifdef ARCH_BLACKHOLE
    noc_async_writes_flushed();
#endif
    noc_semaphore_set(receiver_sem_ptr, VALID);
    noc_semaphore_set_multicast(
        get_semaphore(receiver_sem_id),
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(receiver_sem_id)),
        num_receivers);
#else
    noc_semaphore_set(receiver_sem_ptr, INVALID);
    noc_semaphore_inc(get_noc_addr(mcast_x_start, mcast_y_start, get_semaphore(sender_sem_id)), 1);
    noc_semaphore_wait(receiver_sem_ptr, VALID);
#endif
    cb_push_back(cb_small, Kt);

    for (uint32_t c = n_begin; c < n_begin + n_count; c += cols) {
        const uint32_t ncols = (c + cols <= n_begin + n_count) ? cols : (n_begin + n_count - c);
        // `prefetch` K rows per reserve / barrier / push (the compute pops one row of `cols` tiles at a time; the
        // CB holds two batches, so the pops never straddle its end). A shorter last batch still pushes a full one.
        for (uint32_t t0 = 0; t0 < Kt; t0 += prefetch) {
            const uint32_t rows = (t0 + prefetch <= Kt) ? prefetch : (Kt - t0);
            cb_reserve_back(cb_big, prefetch * cols);
            uint32_t l1 = get_write_ptr(cb_big);
            for (uint32_t t = t0; t < t0 + rows; ++t) {
                for (uint32_t j = 0; j < ncols; ++j) {
#ifdef BIG_TRANSPOSED
                    noc_async_read_page((c + j) * Kt + t, big_acc, l1 + j * tile_bytes);
#else
                    noc_async_read_page(t * Nt + c + j, big_acc, l1 + j * tile_bytes);
#endif
                }
                l1 += cols * tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_big, prefetch * cols);
        }
    }
#ifdef MCAST_SENDER
    noc_async_write_barrier();
#endif
}
