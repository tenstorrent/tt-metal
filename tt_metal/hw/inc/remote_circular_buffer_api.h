// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/circular_buffer.h"
#include "tt_metal/hw/inc/debug/assert.h"
#include "utils/utils.h"
#ifndef COMPILE_FOR_TRISC
#include "dataflow_api.h"
#endif

namespace experimental {

namespace detail {

#ifndef COMPILE_FOR_TRISC
FORCE_INLINE void update_pages_sent(
    const RemoteSenderCBInterface& sender_cb_interface, uint32_t aligned_page_adjustment, uint8_t noc) {
    uint32_t aligned_pages_sent_addr = sender_cb_interface.aligned_pages_sent_ptr;
    uint32_t remote_noc_xy_addr = sender_cb_interface.receiver_noc_xy_ptr;
    uint32_t num_receivers = sender_cb_interface.num_receivers;

    // increment the aligned pages sent because we skipped to next aligned page location
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_sent_addr);
    volatile tt_l1_ptr uint32_t* remote_noc_xy_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        uint32_t remote_noc_xy = uint32_t(
            NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, remote_noc_xy_ptr[0]), DYNAMIC_NOC_Y(noc, remote_noc_xy_ptr[1])));
        *pages_sent_ptr += aligned_page_adjustment;
        uint64_t remote_ack_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)pages_sent_ptr);
        noc_semaphore_inc(remote_ack_ptr_addr, aligned_page_adjustment, noc);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        remote_noc_xy_ptr += 2;
    }
}

FORCE_INLINE void update_pages_acked(
    const RemoteReceiverCBInterface& receiver_cb_interface, uint32_t aligned_page_adjustment, uint8_t noc) {
    uint32_t aligned_pages_acked_addr = receiver_cb_interface.aligned_pages_acked_ptr;
    uint32_t sender_noc_x = receiver_cb_interface.sender_noc_x;
    uint32_t sender_noc_y = receiver_cb_interface.sender_noc_y;

    // increment the aligned pages acked because we skipped to next aligned page location
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_acked_addr);
    *pages_acked_ptr += aligned_page_adjustment;
    uint64_t remote_ack_ptr_addr = get_noc_addr(sender_noc_x, sender_noc_y, (uint32_t)pages_acked_ptr, noc);
    noc_semaphore_inc(remote_ack_ptr_addr, aligned_page_adjustment, noc);
}
#else
FORCE_INLINE void update_pages_sent(
    const RemoteSenderCBInterface& sender_cb_interface, uint32_t aligned_page_adjustment, uint8_t noc) {}
FORCE_INLINE void update_pages_acked(
    const RemoteReceiverCBInterface& receiver_cb_interface, uint32_t aligned_page_adjustment, uint8_t noc) {}
#endif
}  // namespace detail

template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_sender_cb_interface(uint32_t cb_id, uint32_t page_size, uint8_t noc) {
    ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_cb_interface.config_ptr)[3];
    uint32_t fifo_start_addr = sender_cb_interface.fifo_start_addr;
    uint32_t fifo_wr_ptr = sender_cb_interface.fifo_wr_ptr;
    uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
    if constexpr (update_remote_over_noc) {
        uint32_t aligned_page_adjustment = 0;
        if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
            aligned_page_adjustment =
                (fifo_start_addr + fifo_size - fifo_wr_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            next_fifo_wr_ptr = fifo_start_addr;
        } else if (next_fifo_wr_ptr != fifo_wr_ptr) {
            aligned_page_adjustment = (next_fifo_wr_ptr - fifo_wr_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
        }
        if (aligned_page_adjustment != 0) {
            detail::update_pages_sent(sender_cb_interface, aligned_page_adjustment, noc);
        }
    } else if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        next_fifo_wr_ptr = fifo_start_addr;
    }
    sender_cb_interface.fifo_wr_ptr = next_fifo_wr_ptr;
    sender_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    sender_cb_interface.fifo_page_size = page_size;
}

template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_receiver_cb_interface(uint32_t cb_id, uint32_t page_size, uint8_t noc) {
    ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.config_ptr)[3];
    uint32_t fifo_start_addr = receiver_cb_interface.fifo_start_addr;
    uint32_t fifo_rd_ptr = receiver_cb_interface.fifo_rd_ptr;
    uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
    uint32_t prev_fifo_limit_page_aligned = receiver_cb_interface.fifo_limit_page_aligned;

    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    if constexpr (update_remote_over_noc) {
        uint32_t aligned_page_adjustment = 0;
        if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
            aligned_page_adjustment = (fifo_size - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            next_fifo_rd_ptr = fifo_start_addr;
        } else if (next_fifo_rd_ptr != fifo_rd_ptr) {
            aligned_page_adjustment = (next_fifo_rd_ptr - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
        }
        if (aligned_page_adjustment != 0) {
            detail::update_pages_acked(receiver_cb_interface, aligned_page_adjustment, noc);
        }
    } else if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }
    receiver_cb_interface.fifo_rd_ptr = next_fifo_rd_ptr;
    receiver_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    receiver_cb_interface.fifo_page_size = page_size;
}

#ifndef COMPILE_FOR_TRISC

FORCE_INLINE void remote_cb_wait_front(uint32_t cb_id, uint32_t num_pages) {
    RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t fifo_limit_page_aligned = remote_cb.fifo_limit_page_aligned;
    if (remote_cb.fifo_rd_ptr + len_bytes >= fifo_limit_page_aligned) {
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.config_ptr)[3];
        len_bytes += remote_cb.fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    }
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t num_pages_recv = 0;
    uint32_t pages_acked = 0;
    uint32_t pages_sent = 0;

    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_acked_ptr - L1_ALIGNMENT);
    do {
        pages_acked = *pages_acked_ptr;
        pages_sent = *pages_sent_ptr;
        num_pages_recv = pages_sent - pages_acked;
    } while (num_pages_recv < num_pages_wait);
}

FORCE_INLINE void remote_cb_pop_front(uint32_t cb_id, uint32_t num_pages, uint8_t noc = noc_index) {
    RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t fifo_limit_page_aligned = remote_cb.fifo_limit_page_aligned;
    uint32_t fifo_rd_ptr = remote_cb.fifo_rd_ptr;
    if (fifo_rd_ptr + len_bytes >= fifo_limit_page_aligned) {
        uint32_t fifo_start_addr = remote_cb.fifo_start_addr;
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.config_ptr)[3];
        remote_cb.fifo_rd_ptr = fifo_start_addr + (fifo_rd_ptr + len_bytes - fifo_limit_page_aligned);
        len_bytes += fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    } else {
        remote_cb.fifo_rd_ptr += len_bytes;
    }
    uint32_t num_aligned_pages = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    detail::update_pages_acked(remote_cb, num_aligned_pages, noc);
}

FORCE_INLINE void remote_cb_reserve_back(uint32_t cb_id, uint32_t num_pages) {
    RemoteSenderCBInterface& remote_cb = get_remote_sender_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;

    uint32_t fifo_limit_page_aligned = remote_cb.fifo_limit_page_aligned;
    uint32_t fifo_start_addr = remote_cb.fifo_start_addr;
    uint32_t fifo_wr_ptr = remote_cb.fifo_wr_ptr;
    if (fifo_wr_ptr + len_bytes >= fifo_limit_page_aligned) {
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.config_ptr)[3];
        len_bytes += fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    }
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages;

    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr + L1_ALIGNMENT);

    uint32_t num_receivers = remote_cb.num_receivers;
    uint32_t fifo_aligned_num_pages =
        (fifo_limit_page_aligned - fifo_start_addr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

    for (uint32_t i = 0; i < num_receivers; ++i) {
        do {
            uint32_t pages_acked = *pages_acked_ptr;
            uint32_t pages_sent = *pages_sent_ptr;
            uint32_t sent_minus_ack = pages_sent - pages_acked;
            free_pages = fifo_aligned_num_pages >= sent_minus_ack ? (fifo_aligned_num_pages - sent_minus_ack) : 0;
        } while (free_pages < num_pages_wait);
        pages_acked_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
    }
}

FORCE_INLINE void remote_cb_push_back_and_write_pages(
    uint32_t cb_id,
    uint32_t local_cb_addr,
    uint32_t num_pages,
    uint32_t num_rows,
    uint32_t coalesced_num_pages_per_row,
    uint32_t coalesced_page_size,
    uint8_t noc = noc_index) {
    RemoteSenderCBInterface& remote_cb = get_remote_sender_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t fifo_wr_ptr = remote_cb.fifo_wr_ptr;
    uint32_t fifo_start_addr = remote_cb.fifo_start_addr;
    uint32_t fifo_limit_page_aligned = remote_cb.fifo_limit_page_aligned;
    if (fifo_wr_ptr + len_bytes >= fifo_limit_page_aligned) {
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.config_ptr)[3];
        len_bytes += fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    }
    uint32_t pages_sent = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t num_receivers = remote_cb.num_receivers;

    uint32_t next_receiver_start_addr_stride = coalesced_num_pages_per_row * coalesced_page_size;
    uint32_t next_block_row_stride = next_receiver_start_addr_stride * num_receivers;

    uint32_t dest_addr;

    uint32_t next_receiver_start_addr_offset = 0;
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* remote_noc_xy_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.receiver_noc_xy_ptr);
    for (uint32_t i = 0; i < num_receivers; ++i) {
        uint32_t src_addr = local_cb_addr + next_receiver_start_addr_offset;
        dest_addr = fifo_wr_ptr;

        uint32_t remote_noc_xy = uint32_t(
            NOC_XY_ENCODING(DYNAMIC_NOC_X(noc, remote_noc_xy_ptr[0]), DYNAMIC_NOC_Y(noc, remote_noc_xy_ptr[1])));
        uint64_t dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

        noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

        for (uint32_t h = 0; h < num_rows; ++h) {
            uint32_t prev_src_addr = src_addr;
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                if ((dest_addr + coalesced_page_size) > fifo_limit_page_aligned) {
                    uint32_t first_len_bytes = fifo_limit_page_aligned - dest_addr;
                    uint32_t second_len_bytes = coalesced_page_size - first_len_bytes;

                    if (first_len_bytes != 0) {
                        noc_async_write_one_packet(src_addr, dest_noc_addr, first_len_bytes, noc);
                        src_addr += first_len_bytes;
                    }

                    dest_addr = fifo_start_addr;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet(src_addr, dest_noc_addr, second_len_bytes, noc);

                    src_addr += second_len_bytes;
                    dest_addr += second_len_bytes;
                    dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                    noc_async_write_one_packet_set_state(dest_noc_addr, coalesced_page_size, noc);

                } else {
                    noc_async_write_one_packet_with_state(src_addr, dest_noc_addr, noc);

                    src_addr += coalesced_page_size;
                    dest_addr += coalesced_page_size;
                }
            }
            src_addr = prev_src_addr + next_block_row_stride;
        }
        next_receiver_start_addr_offset += next_receiver_start_addr_stride;
        *pages_sent_ptr += pages_sent;

        uint64_t remote_sent_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)pages_sent_ptr);
        noc_semaphore_inc(remote_sent_ptr_addr, pages_sent, noc);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        remote_noc_xy_ptr += 2;
    }

    if (dest_addr == fifo_limit_page_aligned) {
        dest_addr = fifo_start_addr;
    }
    remote_cb.fifo_wr_ptr = dest_addr;
}

#endif

template <uint32_t num_local_cbs>
FORCE_INLINE void align_local_cbs_to_remote_cb(
    uint32_t remote_cb_index, const uint32_t (&local_cb_indices)[num_local_cbs]) {
    // We assert that the offset of sender and receiver common attributes are the same
    // so we can use either interface here
    const RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(remote_cb_index);
    uint32_t fifo_limit = remote_cb.fifo_limit_page_aligned >> cb_addr_shift;
    uint32_t fifo_size = fifo_limit - (remote_cb.fifo_start_addr >> cb_addr_shift);
    uint32_t fifo_ptr = remote_cb.fifo_rd_ptr >> cb_addr_shift;
    for (uint32_t i = 0; i < num_local_cbs; i++) {
        LocalCBInterface& local_cb = get_local_cb_interface(local_cb_indices[i]);
        ASSERT(fifo_size % local_cb.fifo_page_size == 0);
        uint32_t fifo_num_pages = fifo_size / local_cb.fifo_page_size;
        local_cb.fifo_limit = fifo_limit;
        local_cb.fifo_size = fifo_size;
        local_cb.fifo_num_pages = fifo_num_pages;
        local_cb.fifo_wr_ptr = fifo_ptr;
        local_cb.fifo_rd_ptr = fifo_ptr;
    }
}

FORCE_INLINE void update_remote_cb_config_in_l1(uint32_t remote_cb_index) {
    // We assert that the offset of sender fifo_wr_ptr and receiver fifo_rd_ptr are the same
    // so just update the fifo_ptr using either interface
    RemoteReceiverCBInterface& remote_cb_interface = get_remote_receiver_cb_interface(remote_cb_index);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        remote_cb_interface.config_ptr + offsetof(RemoteReceiverCBInterface, fifo_rd_ptr)) =
        remote_cb_interface.fifo_rd_ptr;
}

}  // namespace experimental
