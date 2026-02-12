// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/circular_buffer_interface.h"
#include "api/debug/assert.h"
#include "api/alignment.h"
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/lock.h"
#endif

namespace experimental {

namespace detail {

#ifndef COMPILE_FOR_TRISC
#if defined(KERNEL_BUILD)
static constexpr uint8_t default_noc_mode = noc_mode;
#else
static constexpr uint8_t default_noc_mode = 0;
#endif
static constexpr uint8_t default_cmd_buf = write_at_cmd_buf;
template <uint8_t nm = default_noc_mode>
FORCE_INLINE void update_pages_sent(
    const RemoteSenderCBInterface& sender_cb_interface,
    uint32_t aligned_page_adjustment,
    uint8_t noc,
    bool posted,
    uint8_t cmd_buf) {
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
        noc_fast_atomic_increment<nm>(
            noc,
            cmd_buf,
            remote_ack_ptr_addr,
            NOC_UNICAST_WRITE_VC,
            aligned_page_adjustment,
            31 /*wrap*/,
            false /*linked*/,
            posted /*posted*/,
            MEM_NOC_ATOMIC_RET_VAL_ADDR);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        remote_noc_xy_ptr += 2;
    }
}

template <uint8_t nm = default_noc_mode>
FORCE_INLINE void update_pages_acked(
    const RemoteReceiverCBInterface& receiver_cb_interface,
    uint32_t aligned_page_adjustment,
    uint8_t noc,
    bool posted,
    uint8_t cmd_buf) {
    uint32_t aligned_pages_acked_addr = receiver_cb_interface.aligned_pages_acked_ptr;
    uint32_t sender_noc_x = receiver_cb_interface.sender_noc_x;
    uint32_t sender_noc_y = receiver_cb_interface.sender_noc_y;

    // increment the aligned pages acked because we skipped to next aligned page location
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_acked_addr);
    *pages_acked_ptr += aligned_page_adjustment;
    uint64_t remote_ack_ptr_addr = get_noc_addr(sender_noc_x, sender_noc_y, (uint32_t)pages_acked_ptr, noc);
    noc_fast_atomic_increment<nm>(
        noc,
        cmd_buf,
        remote_ack_ptr_addr,
        NOC_UNICAST_WRITE_VC,
        aligned_page_adjustment,
        31 /*wrap*/,
        false /*linked*/,
        posted /*posted*/,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);
}
#else
static constexpr uint8_t default_noc_mode = 0;
static constexpr uint8_t default_cmd_buf = 0;
template <uint8_t nm = default_noc_mode>
FORCE_INLINE void update_pages_sent(
    const RemoteSenderCBInterface& sender_cb_interface,
    uint32_t aligned_page_adjustment,
    uint8_t noc,
    bool posted,
    uint8_t cmd_buf) {}
template <uint8_t nm = default_noc_mode>
FORCE_INLINE void update_pages_acked(
    const RemoteReceiverCBInterface& receiver_cb_interface,
    uint32_t aligned_page_adjustment,
    uint8_t noc,
    bool posted,
    uint8_t cmd_buf) {}
#endif
}  // namespace detail

template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_sender_cb_interface(
    uint32_t cb_id,
    uint32_t page_size,
    uint8_t noc,
    uint8_t nm = detail::default_noc_mode,
    bool posted = true,
    uint8_t cmd_buf = detail::default_cmd_buf) {
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
            if (nm == DM_DYNAMIC_NOC) {
                detail::update_pages_sent<DM_DYNAMIC_NOC>(
                    sender_cb_interface, aligned_page_adjustment, noc, posted, cmd_buf);
            } else {
                detail::update_pages_sent<DM_DEDICATED_NOC>(
                    sender_cb_interface, aligned_page_adjustment, noc, posted, cmd_buf);
            }
        }
    } else if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
        next_fifo_wr_ptr = fifo_start_addr;
    }
    sender_cb_interface.fifo_wr_ptr = next_fifo_wr_ptr;
    sender_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    sender_cb_interface.fifo_page_size = page_size;
}

template <bool update_remote_over_noc = false>
FORCE_INLINE void resize_remote_receiver_cb_interface(
    uint32_t cb_id,
    uint32_t page_size,
    uint8_t noc,
    uint8_t nm = detail::default_noc_mode,
    bool posted = true,
    uint8_t cmd_buf = detail::default_cmd_buf) {
    ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.config_ptr)[3];
    uint32_t fifo_start_addr = receiver_cb_interface.fifo_start_addr;
    uint32_t fifo_rd_ptr = receiver_cb_interface.fifo_rd_ptr;
    uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
    uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
    if constexpr (update_remote_over_noc) {
        uint32_t aligned_page_adjustment = 0;
        if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
            aligned_page_adjustment =
                (fifo_start_addr + fifo_size - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            next_fifo_rd_ptr = fifo_start_addr;
        } else if (next_fifo_rd_ptr != fifo_rd_ptr) {
            aligned_page_adjustment = (next_fifo_rd_ptr - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
        }
        if (aligned_page_adjustment != 0) {
            // wait for sender to send the credits first so ack ptr never go faster than sent ptr.
            uint32_t pages_acked = 0;
            uint32_t pages_sent = 0;
            uint32_t num_pages_recv = 0;
            volatile tt_l1_ptr uint32_t* pages_acked_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.aligned_pages_acked_ptr);
            volatile tt_l1_ptr uint32_t* pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                receiver_cb_interface.aligned_pages_acked_ptr - L1_ALIGNMENT);
            do {
                invalidate_l1_cache();
                pages_acked = *pages_acked_ptr;
                pages_sent = *pages_sent_ptr;
                num_pages_recv = pages_sent - pages_acked;
            } while (num_pages_recv < aligned_page_adjustment);

            if (nm == DM_DYNAMIC_NOC) {
                detail::update_pages_acked<DM_DYNAMIC_NOC>(
                    receiver_cb_interface, aligned_page_adjustment, noc, posted, cmd_buf);
            } else {
                detail::update_pages_acked<DM_DEDICATED_NOC>(
                    receiver_cb_interface, aligned_page_adjustment, noc, posted, cmd_buf);
            }
        }
    } else if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }
    receiver_cb_interface.fifo_rd_ptr = next_fifo_rd_ptr;
    receiver_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
    receiver_cb_interface.fifo_page_size = page_size;
}

#ifndef COMPILE_FOR_TRISC
#if defined(KERNEL_BUILD)
FORCE_INLINE void remote_cb_wait_front(uint32_t cb_id, uint32_t num_pages) {
    WAYPOINT("RCWF");
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
        invalidate_l1_cache();
        pages_acked = *pages_acked_ptr;
        pages_sent = *pages_sent_ptr;
        num_pages_recv = pages_sent - pages_acked;
    } while (num_pages_recv < num_pages_wait);
    WAYPOINT("RCWD");
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
    detail::update_pages_acked(remote_cb, num_aligned_pages, noc, false, write_at_cmd_buf);
}

FORCE_INLINE void remote_cb_reserve_back(uint32_t cb_id, uint32_t num_pages) {
    WAYPOINT("RCRB");
    RemoteSenderCBInterface& remote_cb = get_remote_sender_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;

    uint32_t fifo_limit_page_aligned = remote_cb.fifo_limit_page_aligned;
    uint32_t fifo_start_addr = remote_cb.fifo_start_addr;
    uint32_t fifo_wr_ptr = remote_cb.fifo_wr_ptr;
    uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.config_ptr)[3];
    if (fifo_wr_ptr + len_bytes >= fifo_limit_page_aligned) {
        len_bytes += fifo_start_addr + fifo_size - fifo_limit_page_aligned;
    }
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    uint32_t free_pages;

    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr + L1_ALIGNMENT);

    uint32_t num_receivers = remote_cb.num_receivers;
    uint32_t fifo_aligned_num_pages = fifo_size / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

    for (uint32_t i = 0; i < num_receivers; ++i) {
        do {
            invalidate_l1_cache();
            uint32_t pages_acked = *pages_acked_ptr;
            uint32_t pages_sent = *pages_sent_ptr;
            uint32_t sent_minus_ack = pages_sent - pages_acked;
            free_pages = fifo_aligned_num_pages - sent_minus_ack;
        } while (free_pages < num_pages_wait);
        pages_acked_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
    }

    WAYPOINT("RCRD");
}

FORCE_INLINE void remote_cb_sender_barrier(uint32_t cb_id) {
    WAYPOINT("RCBW");
    RemoteSenderCBInterface& remote_cb = get_remote_sender_cb_interface(cb_id);

    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.aligned_pages_sent_ptr + L1_ALIGNMENT);

    uint32_t num_receivers = remote_cb.num_receivers;

    for (uint32_t i = 0; i < num_receivers; ++i) {
        while (*pages_acked_ptr != *pages_sent_ptr) {
        }
        pages_acked_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
    }
    WAYPOINT("RCBD");
}

template <bool skip_ptr_update = true>
FORCE_INLINE void remote_cb_push_back_and_write_pages(
    uint32_t cb_id,
    uint32_t local_cb_addr,
    uint32_t num_pages,
    uint32_t num_rows,
    uint32_t coalesced_num_pages_per_row,
    uint32_t coalesced_page_size,
    uint8_t noc = noc_index) {
    constexpr bool non_posted = !skip_ptr_update;
    constexpr bool posted = skip_ptr_update;
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

        noc_async_write_one_packet_set_state<posted>(dest_noc_addr, coalesced_page_size, noc);

        for (uint32_t h = 0; h < num_rows; ++h) {
            uint32_t prev_src_addr = src_addr;
            for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                dest_noc_addr = get_noc_addr_helper(remote_noc_xy, dest_addr);

                noc_async_write_one_packet_with_state<posted>(src_addr, dest_noc_addr, noc);

                src_addr += coalesced_page_size;
                dest_addr += coalesced_page_size;
            }
            src_addr = prev_src_addr + next_block_row_stride;
        }
        next_receiver_start_addr_offset += next_receiver_start_addr_stride;
        *pages_sent_ptr += pages_sent;

        uint64_t remote_sent_ptr_addr = get_noc_addr_helper(remote_noc_xy, (uint32_t)pages_sent_ptr);
        noc_semaphore_inc<posted>(remote_sent_ptr_addr, pages_sent, noc);
        pages_sent_ptr += 2 * L1_ALIGNMENT / sizeof(uint32_t);
        remote_noc_xy_ptr += 2;
    }

    if (dest_addr == fifo_limit_page_aligned) {
        dest_addr = fifo_start_addr;
    }
    remote_cb.fifo_wr_ptr = dest_addr;
}
#endif
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

#if !defined(COMPILE_FOR_TRISC) && defined(KERNEL_BUILD)

class Noc;
class CircularBuffer;

/** @brief Remote circular buffer API
 * Provides an interface for the Producer and Consumer cores of a Circular Buffer to be on different cores on the same
 * chip.
 */
class RemoteCircularBuffer {
public:
    /** @brief Enum class for the type of remote pointer update
     *
     * SKIP: Skip updating the remote pointer
     * UPDATE_OVER_NOC: Update the remote pointer over the NoC. This will cause a NoC transaction.
     */
    enum class RemotePointerUpdate { SKIP, UPDATE_OVER_NOC };

    /** @brief Construct a RemoteCircularBuffer
     *
     * @param remote_cb_index The index of the remote circular buffer
     */
    explicit RemoteCircularBuffer(uint32_t remote_cb_index) : remote_cb_index_(remote_cb_index) {}

    /** @brief Reserves the specified number of pages on the remote circular buffer
     *
     * This will block until the specified number of pages are available on the remote circular buffer.
     *
     * This is intended to be called by the sender core.
     *
     * @param num_pages The number of pages to reserve
     */
    void reserve_back(uint32_t num_pages) { remote_cb_reserve_back(remote_cb_index_, num_pages); }

    /** @brief Pushes the specified number of pages to the remote circular buffer
     *
     * Must call reserve_back before calling this function.
     *
     * This is intended to be called by the sender core.
     *
     * @tparam update_remote_pointer The type of remote pointer update
     *
     * @param noc The NoC to use for the remote pointer update
     * @param src The source to push from. Must be local.
     * @param num_pages The number of pages to push
     * @param num_rows The number of rows to push
     * @param coalesced_num_pages_per_row The number of coalesced pages per row
     * @param coalesced_page_size The size of the coalesced page
     */
    template <typename Src, RemotePointerUpdate update_remote_pointer = RemotePointerUpdate::UPDATE_OVER_NOC>
    void push_back(
        experimental::Noc& noc,
        const Src& src,
        uint32_t num_pages,
        uint32_t num_rows,
        uint32_t coalesced_num_pages_per_row,
        uint32_t coalesced_page_size,
        const typename experimental::noc_traits_t<Src>::src_args_type& src_args =
            typename experimental::noc_traits_t<Src>::src_args_type{}) {
        auto src_addr = experimental::noc_traits_t<Src>::template src_addr<experimental::Noc::AddressType::LOCAL_L1>(
            src, noc, src_args);
        remote_cb_push_back_and_write_pages<update_remote_pointer == RemotePointerUpdate::UPDATE_OVER_NOC>(
            remote_cb_index_,
            src_addr,
            num_pages,
            num_rows,
            coalesced_num_pages_per_row,
            coalesced_page_size,
            noc.get_noc_id());
    }

    /** @brief Resizes the sender's circular buffer page size
     *
     * Resizes the sender circular buffer's page size. This may result in noc transactions for synchronizing with the
     * remote receiver core.
     *
     * This is intended to be called by the sender core.
     *
     * @tparam update_remote_pointer The type of remote pointer update
     *
     * @param noc The NoC to use for the remote pointer update
     * @param page_size The new page size
     * @param noc_mode The NoC mode to use for the remote pointer update
     * @param posted Whether to use posted semaphore inc
     * @param cmd_buf The command buffer to use for the remote pointer update
     */
    template <RemotePointerUpdate update_remote_pointer = RemotePointerUpdate::UPDATE_OVER_NOC>
    void set_sender_page_size(
        experimental::Noc& noc,
        uint32_t page_size,
        uint8_t noc_mode = detail::default_noc_mode,
        bool posted = true,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        resize_remote_receiver_cb_interface<update_remote_pointer == RemotePointerUpdate::UPDATE_OVER_NOC>(
            remote_cb_index_, page_size, noc.get_noc_id(), noc_mode, posted, cmd_buf);
    }

    /** @brief Waits for the specified number of pages to be available in the remote circular buffer
     *
     * This is intended to be called by the receiver core.
     *
     * @param num_pages The number of pages to wait for
     */
    void wait_front(uint32_t num_pages) { remote_cb_wait_front(remote_cb_index_, num_pages); }

    /** @brief Pops the specified number of pages from the remote circular buffer
     *
     * This function is used by a receiver core to signal it is done with the specified amount of data to its sender
     * core. It will trigger NoC transactions to notify the remote CB that the data has been consumed. `wait_front`
     * should be called before calling this function to ensure the data is available.
     *
     * This is intended to be called by the receiver core.
     *
     * @param noc The NoC to use for the remote pointer update
     * @param num_pages The number of pages to pop
     */
    void pop_front(experimental::Noc& noc, uint32_t num_pages) {
        remote_cb_pop_front(remote_cb_index_, num_pages, noc.get_noc_id());
    }

    /** @brief Resizes the receiver's circular buffer page size
     *
     * Resizes the receiver's circular buffer page size. This may result in noc transactions for synchronizing with the
     * remote sender core.
     *
     * This is intended to be called by the receiver core.
     *
     * @tparam update_remote_pointer The type of remote pointer update
     *
     * @param noc The NoC to use for the remote pointer update
     * @param page_size The new page size
     * @param noc_mode The NoC mode to use for the remote pointer update
     * @param posted Whether to use posted semaphore inc
     * @param cmd_buf The command buffer to use for the remote pointer update
     */
    template <RemotePointerUpdate update_remote_pointer = RemotePointerUpdate::UPDATE_OVER_NOC>
    void set_receiver_page_size(
        experimental::Noc& noc,
        uint32_t page_size,
        uint8_t noc_mode = detail::default_noc_mode,
        Noc::ResponseMode response_mode = Noc::ResponseMode::POSTED,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        resize_remote_sender_cb_interface<update_remote_pointer == RemotePointerUpdate::UPDATE_OVER_NOC>(
            remote_cb_index_,
            page_size,
            noc.get_noc_id(),
            noc_mode,
            response_mode == Noc::ResponseMode::POSTED,
            cmd_buf);
    }

    /** @brief Waits for all pages to be consumed by the receiver core
     *
     */
    void barrier() { remote_cb_sender_barrier(remote_cb_index_); }

    /** @brief Writes the read/write pointers to L1
     *
     * The read/write pointers of the remote circular buffers are stored in L1, so that subsequent programs can resume
     * where the previous pointers were. During execution, this pointer is cached in a struct for optimal perf to avoid
     * repeated L1 reads/writes. This requires the user to call this function at the end of their kernel execution in
     * order to write the final value back to L1. This should only be called by one RISC per core which has the final
     * updated value.
     *
     * This can be called by either the sender or receiver core.
     *
     */
    void commit() { update_remote_cb_config_in_l1(remote_cb_index_); }

    /** @brief Acquire a scoped lock on the RemoteCircularBuffer. In debug mode, reads and writes to this remote
     * circular buffer are tracked by the debugger while the lock is held.
     *
     * @return A scoped lock on the RemoteCircularBuffer
     */
    [[nodiscard]] auto scoped_lock() {
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    uint32_t remote_cb_index_;
};

#endif

}  // namespace experimental
