// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

#define L1_NOC_ALIGNMENT 16 // XXXXX is the defined elsewhere?

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void downstream_cb_acquire_pages(uint32_t n) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    DEBUG_STATUS('A', 'P', 'W');

    // Ensure last sem_inc has landed
    noc_async_write_barrier(); // XXXX TODO(pgk) can we do better on wormhole?

    while (*sem_addr < n);
    DEBUG_STATUS('A', 'P', 'D');
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, (uint32_t)sem_addr), -n);
}

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void downstream_cb_release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore(sem_id)), n);
}

template<uint32_t noc_xy,
         uint32_t sem_id,
         uint32_t cb_log_page_size>
FORCE_INLINE
uint32_t upstream_cb_acquire_pages(uint32_t cb_fence,
                                   uint32_t block_next_start_addr[],
                                   uint32_t rd_block_idx) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    static uint32_t available = 0;

    if (available == 0) {
        // Ensure last sem_inc has landed
        noc_async_write_barrier(); // XXXX TODO(pgk) can we do better on wormhole?

        DEBUG_STATUS('A', 'P', 'W');
        while ((available = *sem_addr) == 0);
        DEBUG_STATUS('A', 'P', 'D');
    }

    // Set a fence to limit how much is processed at once
    uint32_t limit = (block_next_start_addr[rd_block_idx] - cb_fence) >> cb_log_page_size;
    uint32_t usable = (available > limit) ? limit : available;

    noc_semaphore_inc(get_noc_addr_helper(noc_xy, (uint32_t)sem_addr), -usable);
    available -= usable;

    return usable;
}

template<uint32_t noc_xy,
         uint32_t sem_id,
         uint32_t cb_blocks,
         uint32_t cb_pages_per_block>
FORCE_INLINE
void upstream_cb_block_release_pages(uint32_t block_noc_writes_to_clear[],
                                     uint32_t& wr_block_idx) {

    uint32_t sem_addr = get_semaphore(sem_id);

    uint32_t noc_progress = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    if (noc_progress >= block_noc_writes_to_clear[wr_block_idx]) { // XXXXX ugh, 32 bit wrap?
        noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block);
        wr_block_idx++;
        wr_block_idx &= (cb_blocks - 1);

        // if >cb_pages_per_block are in flight away from this core
        // then we can fall behind by a block and never catch up
        // checking twice ensures we "gain" on the front if possible
        if (noc_progress >= block_noc_writes_to_clear[wr_block_idx]) {
            noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block);
            wr_block_idx++;
            wr_block_idx &= (cb_blocks - 1);
        }
    }
}

template<uint32_t cb_base,
         uint32_t cb_blocks>
FORCE_INLINE
void upstream_move_rd_to_next_block(uint32_t& cmd_ptr,
                                    uint32_t& cb_fence,
                                    uint32_t block_noc_writes_to_clear[],
                                    uint32_t& rd_block_idx) {

    // This is subtle: in the free-running case, we don't want to clear the current block
    // if the noc catches up so we artificially inflate the clear value by 1 when we start
    // a block and adjust it down by 1 here as we complete a block
    uint32_t write_count = block_noc_writes_to_clear[rd_block_idx];
    block_noc_writes_to_clear[rd_block_idx] = write_count - 1;

    rd_block_idx++;
    if (rd_block_idx == cb_blocks) {
        rd_block_idx = 0;
        cb_fence = cb_base;
        cmd_ptr = cb_base;
    }

    block_noc_writes_to_clear[rd_block_idx] = write_count; // this is plus 1
}

template<uint32_t cb_base,
         uint32_t cb_blocks,
         uint32_t cb_log_page_size,
         uint32_t noc_xy,
         uint32_t cb_sem>
FORCE_INLINE
void upstream_get_cb_page(uint32_t& cmd_ptr,
                          uint32_t& cb_fence,
                          uint32_t block_noc_writes_to_clear[],
                          uint32_t block_next_start_addr[],
                          uint32_t& rd_block_idx) {

    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        upstream_move_rd_to_next_block<cb_base,
                                       cb_blocks>(cmd_ptr,
                                                  cb_fence,
                                                  block_noc_writes_to_clear,
                                                  rd_block_idx);
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages = upstream_cb_acquire_pages<noc_xy,
                                                 cb_sem,
                                                 cb_log_page_size>(cb_fence,
                                                                   block_next_start_addr,
                                                                   rd_block_idx);
    cb_fence += n_pages << cb_log_page_size;
}
