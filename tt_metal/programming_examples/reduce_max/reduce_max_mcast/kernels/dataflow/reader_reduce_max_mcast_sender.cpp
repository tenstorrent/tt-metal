// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Sender reader: loads the scaler tile from DRAM, multicasts it to all receiver
 * cores, then streams its own input tiles from DRAM.
 *
 * Only ONE core in the grid runs this kernel.  All other cores receive the scaler
 * tile through the NoC multicast, so the DRAM bandwidth spent on this constant
 * tile is O(1) regardless of core count.
 *
 * Two-semaphore handshake (matching the production matmul pattern):
 *
 *   sem_sender   — lives on THIS core.  Each receiver atomically increments it
 *                  (via noc_semaphore_inc) once it has called cb_reserve_back on
 *                  the scaler CB, guaranteeing its L1 slot is allocated and ready
 *                  to receive the incoming write.  Sender waits until the count
 *                  reaches num_receivers before issuing the multicast.
 *
 *   sem_receiver — lives on ALL cores (allocated by CreateSemaphore on all_cores).
 *                  Sender pre-sets its local copy to VALID, then multicasts that
 *                  value to every receiver via noc_semaphore_set_multicast.
 *                  Receivers spin-wait on VALID before calling cb_push_back.
 *
 * Why pre-set sem_receiver = VALID at the top of kernel_main?
 *   noc_semaphore_set_multicast reads the current value at the local address and
 *   writes it to all remote addresses.  By setting the local copy to VALID before
 *   the multicast, we avoid a separate load-then-store sequence inside the hot
 *   path.  This is the same approach used in the matmul sender kernel.
 *
 * Runtime arguments:
 *   0:  src_addr         — DRAM base address of the input matrix.
 *   1:  scaler_addr      — DRAM address of the scaler tile (one 32×32 tile of 1.0).
 *   2:  mt_start         — First tile-row group assigned to this core.
 *   3:  mt_count         — Number of tile-row groups assigned to this core.
 *   4:  Nt               — Number of tile columns in the input matrix.
 *   5:  mcast_start_x    — NOC1 multicast rectangle start x (MAX physical x among receivers).
 *   6:  mcast_start_y    — NOC1 multicast rectangle start y (MAX physical y among receivers).
 *   7:  mcast_end_x      — NOC1 multicast rectangle end x   (MIN physical x among receivers).
 *   8:  mcast_end_y      — NOC1 multicast rectangle end y   (MIN physical y among receivers).
 *
 *   NOTE: This kernel runs on RISCV_1 (NOC1).  On NOC1, the multicast bounding
 *   box must be passed as (max_x, max_y) → (min_x, min_y), the reverse of the
 *   NOC0 convention.  The host must swap start/end accordingly.
 *   9:  num_receivers    — Number of receiver cores (NOT counting the sender).
 *   10: sem_sender_id    — Semaphore index (passed to get_semaphore()), on sender.
 *   11: sem_receiver_id  — Semaphore index (passed to get_semaphore()), on all cores.
 *
 * Compile-time arguments:
 *   [0..N)  TensorAccessorArgs for src_dram_buffer.
 *   [N..M)  TensorAccessorArgs for scaler_dram_buffer.
 */
void kernel_main() {
    uint32_t src_addr        = get_arg_val<uint32_t>(0);
    uint32_t scaler_addr     = get_arg_val<uint32_t>(1);
    uint32_t mt_start        = get_arg_val<uint32_t>(2);
    uint32_t mt_count        = get_arg_val<uint32_t>(3);
    uint32_t Nt              = get_arg_val<uint32_t>(4);
    uint32_t mcast_start_x   = get_arg_val<uint32_t>(5);
    uint32_t mcast_start_y   = get_arg_val<uint32_t>(6);
    uint32_t mcast_end_x     = get_arg_val<uint32_t>(7);
    uint32_t mcast_end_y     = get_arg_val<uint32_t>(8);
    uint32_t num_receivers   = get_arg_val<uint32_t>(9);
    uint32_t sem_sender_id   = get_arg_val<uint32_t>(10);
    uint32_t sem_receiver_id = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_id_in     = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_1;

    constexpr auto src_args    = TensorAccessorArgs<0>();
    const auto src             = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));
    constexpr auto scaler_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto scaler          = TensorAccessor(scaler_args, scaler_addr, get_tile_size(cb_id_scaler));

    // Pre-set our local sem_receiver to VALID.  noc_semaphore_set_multicast will
    // propagate this exact value to all receiver cores, signalling them that the
    // tile data has landed.
    uint32_t sem_receiver_l1 = get_semaphore(sem_receiver_id);
    volatile tt_l1_ptr uint32_t* sem_receiver_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_receiver_l1);
    *(sem_receiver_ptr) = VALID;

    volatile tt_l1_ptr uint32_t* sem_sender_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_sender_id));

    // -------------------------------------------------------------------------
    // Scaler: read from DRAM once, then multicast to all receivers.
    // -------------------------------------------------------------------------
    cb_reserve_back(cb_id_scaler, 1);
    uint32_t scaler_l1 = get_write_ptr(cb_id_scaler);

    noc_async_read_tile(0, scaler, scaler_l1);
    noc_async_read_barrier();

    if (num_receivers > 0) {
        // Wait until every receiver has called cb_reserve_back and incremented
        // sem_sender, guaranteeing its L1 slot is reserved and ready for the write.
        noc_semaphore_wait(sem_sender_ptr, num_receivers);
        noc_semaphore_set(sem_sender_ptr, 0);  // reset for potential future dispatch

        // Write-multicast the scaler tile to all receiver L1 slots.
        // num_receivers does NOT include the sender — the sender already has the
        // data from the DRAM read above.
        uint64_t mcast_data_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, scaler_l1);
        noc_async_write_multicast(scaler_l1, mcast_data_addr, get_tile_size(cb_id_scaler), num_receivers);

#ifdef ARCH_BLACKHOLE
        // On Blackhole, separate cmd buffer FIFOs may reorder writes; flush before
        // sending the semaphore multicast to preserve ordering.
        noc_async_writes_flushed();
#endif

        // Signal all receivers that their tile has arrived.
        // sem_receiver_l1 already holds VALID (set above), so this multicasts VALID
        // to every receiver's sem_receiver address.
        uint64_t mcast_sem_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, sem_receiver_l1);
        noc_semaphore_set_multicast(sem_receiver_l1, mcast_sem_addr, num_receivers);

        noc_async_write_barrier();
    }

    // Signal compute on this core that the scaler tile is in L1.
    cb_push_back(cb_id_scaler, 1);

    // -------------------------------------------------------------------------
    // Input tiles: same streaming loop as the multi-core reader.
    // -------------------------------------------------------------------------
    for (uint32_t mt = mt_start; mt < mt_start + mt_count; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            uint32_t tile_index = mt * Nt + nt;
            cb_reserve_back(cb_id_in, 1);
            noc_async_read_tile(tile_index, src, get_write_ptr(cb_id_in));
            noc_async_read_barrier();
            cb_push_back(cb_id_in, 1);
        }
    }
}
