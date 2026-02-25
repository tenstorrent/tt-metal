// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Receiver reader: waits for the scaler tile multicast from the sender core,
 * then streams its own input tiles from DRAM.
 *
 * All non-sender cores run this kernel.  The scaler tile arrives through the NoC
 * multicast (not a DRAM read), so only one DRAM access for the scaler is needed
 * across the entire core grid.
 *
 * Two-semaphore handshake — receiver side:
 *
 *   1. cb_reserve_back(scaler)
 *        Allocates the L1 slot.  The incoming multicast will write the tile
 *        directly to get_write_ptr(scaler), which is at the same L1 address on
 *        every core because CBs are laid out identically across the program.
 *
 *   2. noc_semaphore_set(sem_receiver, INVALID)
 *        Clears any stale VALID from a previous dispatch before signalling the
 *        sender.  Order matters: set INVALID *before* incrementing the sender
 *        semaphore, otherwise the sender could multicast VALID before we have
 *        cleared it, and we would miss the signal.
 *
 *   3. noc_semaphore_inc(sender's sem_sender, 1)
 *        Atomically increments the sender's counter.  Once the counter reaches
 *        num_receivers, the sender knows every core has a reserved L1 slot and
 *        issues the write-multicast.
 *
 *   4. noc_semaphore_wait(sem_receiver, VALID)
 *        Spin-waits until the sender has written the tile AND multicasted VALID
 *        into our local sem_receiver.  Because the data write and the semaphore
 *        write travel on the same NOC/VC, VALID arriving implies the tile data
 *        has already landed.
 *
 *   5. cb_push_back(scaler)
 *        The tile is in L1; signal the compute kernel.
 *
 * Runtime arguments:
 *   0: src_addr         — DRAM base address of the input matrix.
 *   1: mt_start         — First tile-row group assigned to this core.
 *   2: mt_count         — Number of tile-row groups assigned to this core.
 *   3: Nt               — Number of tile columns in the input matrix.
 *   4: sender_phys_x    — Physical NoC x of the sender core.
 *   5: sender_phys_y    — Physical NoC y of the sender core.
 *   6: sem_sender_id    — Semaphore index (get_semaphore()), lives on the sender.
 *   7: sem_receiver_id  — Semaphore index (get_semaphore()), lives on all cores.
 *
 * Compile-time arguments:
 *   [0..N)  TensorAccessorArgs for src_dram_buffer.
 */
void kernel_main() {
    uint32_t src_addr        = get_arg_val<uint32_t>(0);
    uint32_t mt_start        = get_arg_val<uint32_t>(1);
    uint32_t mt_count        = get_arg_val<uint32_t>(2);
    uint32_t Nt              = get_arg_val<uint32_t>(3);
    uint32_t sender_phys_x   = get_arg_val<uint32_t>(4);
    uint32_t sender_phys_y   = get_arg_val<uint32_t>(5);
    uint32_t sem_sender_id   = get_arg_val<uint32_t>(6);
    uint32_t sem_receiver_id = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in     = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_1;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src          = TensorAccessor(src_args, src_addr, get_tile_size(cb_id_in));

    // -------------------------------------------------------------------------
    // Scaler: wait for multicast from the sender core.
    // -------------------------------------------------------------------------

    // Reserve the L1 slot.  The multicast will write directly to this address.
    cb_reserve_back(cb_id_scaler, 1);

    uint32_t sem_receiver_l1 = get_semaphore(sem_receiver_id);
    volatile tt_l1_ptr uint32_t* sem_receiver_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_receiver_l1);

    // Set INVALID before signalling the sender so we cannot accidentally observe
    // a stale VALID from a previous dispatch.
    noc_semaphore_set(sem_receiver_ptr, INVALID);

    // Notify the sender that our L1 slot is reserved and we are ready to receive.
    uint64_t remote_sender_sem =
        get_noc_addr(sender_phys_x, sender_phys_y, get_semaphore(sem_sender_id));
    noc_semaphore_inc(remote_sender_sem, 1);

    // Spin until the sender has written the tile and set our semaphore to VALID.
    // Data and semaphore travel on the same NOC/VC, so VALID implies tile is in L1.
    noc_semaphore_wait(sem_receiver_ptr, VALID);

    // Tile is in L1 at get_write_ptr(cb_id_scaler).  Signal the compute kernel.
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
