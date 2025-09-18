// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_config.h"
#include "risc_attribs.h"
#include "dataflow_api.h"
#include "cq_helpers.hpp"

#include "debug/sanitize_noc.h"

// The command queue read interface controls reads from the issue region, host owns the issue region write interface
// Commands and data to send to device are pushed into the issue region
struct CQReadInterface {
    uint32_t issue_fifo_size;
    uint32_t issue_fifo_limit;  // range is inclusive of the limit
    uint32_t issue_fifo_rd_ptr;
    uint32_t issue_fifo_rd_toggle;
};

// The command queue write interface controls writes to the completion region, host owns the completion region read
// interface Data requests from device and event states are written to the completion region
struct CQWriteInterface {
    uint32_t completion_fifo_size;
    uint32_t completion_fifo_limit;  // range is inclusive of the limit
    uint32_t completion_fifo_wr_ptr;
    uint32_t completion_fifo_wr_toggle;
};

constexpr ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

FORCE_INLINE
uint32_t round_up_pow2(uint32_t v, uint32_t pow2_size) { return (v + (pow2_size - 1)) & ~(pow2_size - 1); }

FORCE_INLINE
uint32_t div_up(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

FORCE_INLINE
uint32_t wrap_ge(uint32_t a, uint32_t b) {
    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return diff >= 0;
}

FORCE_INLINE
uint32_t wrap_gt(uint32_t a, uint32_t b) {
    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return diff > 0;
}

constexpr bool use_fabric(uint64_t fabric_router_xy) { return fabric_router_xy != 0; }

template <
    enum CQNocFlags flags,
    enum CQNocWait wait = CQ_NOC_WAIT,
    enum CQNocSend send = CQ_NOC_SEND,
    uint32_t cmd_buf = NCRISC_WR_CMD_BUF,
    bool update_counters = false>
FORCE_INLINE void cq_noc_async_write_with_state(
    uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1, uint8_t noc = noc_index) {
    if constexpr (wait) {
        WAYPOINT("CNSW");
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        WAYPOINT("CNSD");
    }

    noc_write_with_state<DM_DEDICATED_NOC, cmd_buf, flags, CQ_NOC_send, CQ_NOC_wait, false>(
        noc, src_addr, dst_addr, size, ndests);

    if constexpr (send) {
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc, cmd_buf);
        noc_write_with_state<DM_DEDICATED_NOC, cmd_buf, CQ_NOC_sndl, send, CQ_NOC_wait, update_counters>(
            noc, src_addr, dst_addr, size, ndests);
    }
}

// More generic version of cq_noc_async_write_with_state: Allows writing an abitrary amount of data, when the NOC config
// (dst_noc, VC..) have been specified.
template <
    bool write_last_packet = true,
    bool update_counters = false,
    enum CQNocWait wait_first = CQ_NOC_WAIT,
    uint32_t cmd_buf = NCRISC_WR_CMD_BUF>
inline uint32_t cq_noc_async_write_with_state_any_len(
    uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1, uint8_t noc = noc_index) {
    if (size > NOC_MAX_BURST_SIZE) {
        cq_noc_async_write_with_state<CQ_NOC_SnDL, wait_first, CQ_NOC_SEND, cmd_buf, update_counters>(
            src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests);
        src_addr += NOC_MAX_BURST_SIZE;
        dst_addr += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
        while (size > NOC_MAX_BURST_SIZE) {
            cq_noc_async_write_with_state<CQ_NOC_SnDl, CQ_NOC_WAIT, CQ_NOC_SEND, cmd_buf, update_counters>(
                src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests, noc);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
        }
    }
    if constexpr (write_last_packet) {
        cq_noc_async_write_with_state<CQ_NOC_SnDL, CQ_NOC_WAIT, CQ_NOC_SEND, cmd_buf, update_counters>(
            src_addr, dst_addr, size, ndests, noc);
        return 0;
    } else {
        return size;
    }
}

template <enum CQNocFlags flags, bool mcast = false, bool linked = false, uint32_t cmd_buf = NCRISC_WR_CMD_BUF>
FORCE_INLINE void cq_noc_async_write_init_state(
    uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint8_t noc = noc_index) {
    WAYPOINT("CNIW");
    uint32_t heartbeat = 0;
    while (!noc_cmd_buf_ready(noc, cmd_buf)) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("CNID");

    constexpr enum CQNocCmdFlags cmd_flags = static_cast<enum CQNocCmdFlags>(
        (mcast ? CQ_NOC_CMD_FLAG_MCAST : 0x0) | (linked ? CQ_NOC_CMD_FLAG_LINKED : 0x0));
    constexpr uint32_t vc = mcast ? NOC_DISPATCH_MULTICAST_WRITE_VC : NOC_UNICAST_WRITE_VC;

    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, mcast ? DEBUG_SANITIZE_NOC_MULTICAST : DEBUG_SANITIZE_NOC_UNICAST);

    noc_write_init_state<cmd_buf, cmd_flags>(noc, vc);
    cq_noc_async_write_with_state<flags, CQ_NOC_wait, CQ_NOC_send, cmd_buf>(src_addr, dst_addr, size);
}

template <enum CQNocInlineFlags flags, enum CQNocWait wait = CQ_NOC_WAIT, enum CQNocSend send = CQ_NOC_SEND>
FORCE_INLINE void cq_noc_inline_dw_write_with_state(
    uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF, uint8_t noc = noc_index) {
#if defined(ARCH_BLACKHOLE)
    noc_async_writes_flushed();  // ensure inline_l1_src_addr is not overwritten
    uint32_t inline_l1_src_addr = noc_get_interim_inline_value_addr(noc, dst_addr);
    volatile tt_l1_ptr uint32_t* inline_l1_src_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inline_l1_src_addr);
    *inline_l1_src_addr_ptr = val;
    cq_noc_async_write_with_state<CQ_NOC_SnDL, CQ_NOC_WAIT, CQ_NOC_SEND, NCRISC_WR_REG_CMD_BUF>(
        inline_l1_src_addr, dst_addr, 4);
#else
    if constexpr (wait) {
        WAYPOINT("NISW");
        while (!noc_cmd_buf_ready(noc, NCRISC_WR_REG_CMD_BUF));
        WAYPOINT("NISD");
    }

    noc_inline_dw_write_with_state<NCRISC_WR_REG_CMD_BUF, flags, CQ_NOC_wait, CQ_NOC_send>(noc, dst_addr, val, be);

    if constexpr (send) {
        DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc, NCRISC_WR_REG_CMD_BUF);
        noc_inline_dw_write_with_state<NCRISC_WR_REG_CMD_BUF, CQ_NOC_INLINE_ndvb, CQ_NOC_wait, send>(noc, dst_addr, val, be);
    }
#endif
}

// TODO: noc_inline_dw_write currently hardcodes most of these parameters, which we copied here
// If needed, add templates for setting these
template <enum CQNocInlineFlags flags>
FORCE_INLINE void cq_noc_inline_dw_write_init_state(
    uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF, uint8_t noc = noc_index) {
#if defined(ARCH_BLACKHOLE)
    // On Blackhole inline writes are disabled so use cq_noc_async_write_init_state with inline write cmd buf
    // See comment in `noc_inline_dw_write` for more details
    uint32_t inline_l1_src_addr = noc_get_interim_inline_value_addr(noc, dst_addr);
    volatile tt_l1_ptr uint32_t* inline_l1_src_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inline_l1_src_addr);
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, NCRISC_WR_REG_CMD_BUF>(0, dst_addr, 0);
#else
    WAYPOINT("NIIW");
    uint32_t heartbeat = 0;
    while (!noc_cmd_buf_ready(noc, NCRISC_WR_REG_CMD_BUF)) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("NIID");

    constexpr uint32_t static_vc = NOC_UNICAST_WRITE_VC;
    constexpr enum CQNocCmdFlags cmd_flags = CQ_NOC_mkp;
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(
        noc, (cmd_flags & CQ_NOC_CMD_FLAG_MCAST) ? DEBUG_SANITIZE_NOC_MULTICAST : DEBUG_SANITIZE_NOC_UNICAST);

    noc_inline_dw_write_init_state<NCRISC_WR_REG_CMD_BUF, cmd_flags>(noc, static_vc);
    cq_noc_inline_dw_write_with_state<flags, CQ_NOC_wait, CQ_NOC_send>(dst_addr, val, be);
#endif
}

template <uint32_t sem_id>
FORCE_INLINE void cb_wait_all_pages(uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    // Downstream component sets the MSB as a terminate bit
    // Mask that off to avoid a race between the sem count and terminate
    n &= 0x7fffffff;

    WAYPOINT("TAPW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr & 0x7fffffff) != n);  // mask off terminate bit
    WAYPOINT("TAPD");
}

template <uint32_t sem_id>
FORCE_INLINE void cb_wait_all_pages(uint32_t n, uint32_t& additional_count) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    // Downstream component sets the MSB as a terminate bit
    // Mask that off to avoid a race between the sem count and terminate
    n &= 0x7fffffff;

    WAYPOINT("TAPW");
    do {
        invalidate_l1_cache();
    } while (((additional_count + *sem_addr) & 0x7fffffff) != n);  // mask off terminate bit
    WAYPOINT("TAPD");
}

template <uint32_t noc_xy, uint32_t sem_id>
void cb_acquire_pages(uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    // Ensure last sem_inc has landed
    noc_async_atomic_barrier();

    WAYPOINT("DAPW");
    // Use a wrapping compare here to compare distance
    // Required for trace which steals downstream credits and may make the value negative
    uint32_t heartbeat = 0;
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    } while (wrap_gt(n, *sem_addr));
    WAYPOINT("DAPD");
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, (uint32_t)sem_addr), -n);
}

template <uint32_t noc_xy, uint32_t sem_id>
void cb_acquire_pages(uint32_t n, uint32_t& additional_count) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    // Ensure last sem_inc has landed
    noc_async_atomic_barrier();

    WAYPOINT("DAPW");
    // Use a wrapping compare here to compare distance
    // Required for trace which steals downstream credits and may make the value negative
    uint32_t heartbeat = 0;
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    } while (wrap_gt(n, additional_count + *sem_addr));
    WAYPOINT("DAPD");
    additional_count -= n;
}

template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE void cb_release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), n, noc_idx);
}

template <uint32_t sem_id, uint32_t cb_log_page_size>
FORCE_INLINE uint32_t
cb_acquire_pages(uint32_t cb_fence, uint32_t block_next_start_addr[], uint32_t rd_block_idx, uint32_t& local_count) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(sem_id));

    static uint32_t upstream_count = 0;

    if (local_count == upstream_count) {
        WAYPOINT("UAPW");
        uint32_t heartbeat = 0;
        do {
            invalidate_l1_cache();
            IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, 0);
        } while ((upstream_count = *sem_addr) == local_count);
        WAYPOINT("UAPD");
    }

    // Set a fence to limit how much is processed at once
    uint32_t limit = (block_next_start_addr[rd_block_idx] - cb_fence) >> cb_log_page_size;
    uint32_t available = upstream_count - local_count;
    uint32_t usable = (available > limit) ? limit : available;

    local_count += usable;

    return usable;
}

// Do not release pages on the first call to the cb block release pages functions below
// This is because the first call means we don't have a previous block to release
static bool cb_block_released_prev_block = false;

template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id, uint32_t cb_pages_per_block>
FORCE_INLINE void cb_block_release_pages(uint32_t& block_noc_writes_to_clear, uint8_t noc = noc_index) {
    if (cb_block_released_prev_block) {
        WAYPOINT("CBRW");
        uint32_t sem_addr = get_semaphore<fd_core_type>(sem_id);
        while (!wrap_ge(NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT), block_noc_writes_to_clear));
        noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block, noc_idx);
        WAYPOINT("CBRD");
    } else {
        cb_block_released_prev_block = true;
    }
    block_noc_writes_to_clear = noc_nonposted_writes_num_issued[noc];
}

template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id, uint32_t cb_pages_per_block, typename T>
FORCE_INLINE void cb_block_release_pages_remote(
    T& relay_client, uint32_t& block_noc_writes_to_clear, uint8_t noc = noc_index) {
    if (cb_block_released_prev_block) {
        WAYPOINT("CBRW");
        while (!wrap_ge(NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT), block_noc_writes_to_clear));
        relay_client.template release_pages<noc_idx, noc_xy, sem_id>(cb_pages_per_block);
        WAYPOINT("CBRD");
    } else {
        cb_block_released_prev_block = true;
    }
    block_noc_writes_to_clear = noc_nonposted_writes_num_issued[noc];
}

template <uint32_t cb_blocks>
FORCE_INLINE void move_rd_to_next_block(uint32_t& rd_block_idx) {
    static_assert((cb_blocks & (cb_blocks - 1)) == 0);
    rd_block_idx++;
    rd_block_idx &= cb_blocks - 1;
}

template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id, uint32_t cb_pages_per_block, uint32_t cb_blocks>
FORCE_INLINE void move_rd_to_next_block_and_release_pages(
    uint32_t& block_noc_writes_to_clear, uint32_t& rd_block_idx, uint8_t noc = noc_index) {
    cb_block_release_pages<noc_idx, noc_xy, sem_id, cb_pages_per_block>(block_noc_writes_to_clear, noc);
    move_rd_to_next_block<cb_blocks>(rd_block_idx);
}

template <
    uint8_t noc_idx,
    uint32_t noc_xy,
    uint32_t sem_id,
    uint32_t cb_pages_per_block,
    uint32_t cb_blocks,
    typename T>
FORCE_INLINE void move_rd_to_next_block_and_release_pages_remote(
    T& relay_client, uint32_t& block_noc_writes_to_clear, uint32_t& rd_block_idx, uint8_t noc = noc_index) {
    cb_block_release_pages_remote<noc_idx, noc_xy, sem_id, cb_pages_per_block>(
        relay_client, block_noc_writes_to_clear, noc);
    move_rd_to_next_block<cb_blocks>(rd_block_idx);
}

template <
    uint32_t cb_base,
    uint32_t cb_blocks,
    uint32_t cb_log_page_size,
    uint32_t local_cb_sem,
    uint8_t upstream_noc_idx,
    uint32_t upstream_noc_xy,
    uint32_t upstream_cb_sem,
    uint32_t cb_pages_per_block>
FORCE_INLINE uint32_t get_cb_page_and_release_pages(
    uint32_t& cmd_ptr,
    uint32_t& cb_fence,
    uint32_t& block_noc_writes_to_clear,
    uint32_t block_next_start_addr[],
    uint32_t& rd_block_idx,
    uint32_t& local_count,
    uint8_t noc = noc_index) {
    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        if (rd_block_idx == cb_blocks - 1) {
            cmd_ptr = cb_base;
            cb_fence = cb_base;
        }
        move_rd_to_next_block_and_release_pages<
            upstream_noc_idx,
            upstream_noc_xy,
            upstream_cb_sem,
            cb_pages_per_block,
            cb_blocks>(block_noc_writes_to_clear, rd_block_idx, noc);
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages =
        cb_acquire_pages<local_cb_sem, cb_log_page_size>(cb_fence, block_next_start_addr, rd_block_idx, local_count);
    cb_fence += n_pages << cb_log_page_size;

    return n_pages;
}

template <
    uint32_t cb_base,
    uint32_t cb_blocks,
    uint32_t cb_log_page_size,
    uint32_t local_cb_sem,
    uint8_t upstream_noc_idx,
    uint32_t upstream_noc_xy,
    uint32_t upstream_cb_sem,
    uint32_t cb_pages_per_block,
    typename T>
FORCE_INLINE uint32_t get_cb_page_and_release_pages_remote(
    T& relay_client,
    uint32_t& cmd_ptr,
    uint32_t& cb_fence,
    uint32_t& block_noc_writes_to_clear,
    uint32_t block_next_start_addr[],
    uint32_t& rd_block_idx,
    uint32_t& local_count,
    uint8_t noc = noc_index) {
    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        if (rd_block_idx == cb_blocks - 1) {
            cmd_ptr = cb_base;
            cb_fence = cb_base;
        }
        move_rd_to_next_block_and_release_pages_remote<
            upstream_noc_idx,
            upstream_noc_xy,
            upstream_cb_sem,
            cb_pages_per_block,
            cb_blocks>(relay_client, block_noc_writes_to_clear, rd_block_idx, noc);
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages =
        cb_acquire_pages<local_cb_sem, cb_log_page_size>(cb_fence, block_next_start_addr, rd_block_idx, local_count);
    cb_fence += n_pages << cb_log_page_size;

    return n_pages;
}

template <uint32_t cb_base, uint32_t cb_blocks, uint32_t cb_log_page_size, uint32_t cb_sem>
FORCE_INLINE uint32_t get_cb_page(
    uint32_t& cmd_ptr,
    uint32_t& cb_fence,
    uint32_t block_next_start_addr[],
    uint32_t& rd_block_idx,
    uint32_t& local_count) {
    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        if (rd_block_idx == cb_blocks - 1) {
            cmd_ptr = cb_base;
            cb_fence = cb_base;
        }
        move_rd_to_next_block<cb_blocks>(rd_block_idx);
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages =
        cb_acquire_pages<cb_sem, cb_log_page_size>(cb_fence, block_next_start_addr, rd_block_idx, local_count);
    cb_fence += n_pages << cb_log_page_size;

    return n_pages;
}

constexpr uint32_t l1_to_local_cache_copy_chunk = 6;

// NOTE: CAREFUL USING THIS FUNCTION
// It is call "careful_copy" because you need to be careful...
// It copies beyond count by up to 5 elements make sure src and dst addresses are safe
template <uint32_t l1_to_local_cache_copy_chunk, uint32_t l1_cache_elements_rounded>
FORCE_INLINE void careful_copy_from_l1_to_local_cache(
    volatile uint32_t tt_l1_ptr* l1_ptr, uint32_t count, uint32_t* l1_cache) {
    uint32_t n = 0;
    ASSERT(l1_to_local_cache_copy_chunk == 6);
    ASSERT(count <= l1_cache_elements_rounded);
    while (n < count) {
        uint32_t v0 = l1_ptr[n + 0];
        uint32_t v1 = l1_ptr[n + 1];
        uint32_t v2 = l1_ptr[n + 2];
        uint32_t v3 = l1_ptr[n + 3];
        uint32_t v4 = l1_ptr[n + 4];
        uint32_t v5 = l1_ptr[n + 5];
        l1_cache[n + 0] = v0;
        l1_cache[n + 1] = v1;
        l1_cache[n + 2] = v2;
        l1_cache[n + 3] = v3;
        l1_cache[n + 4] = v4;
        l1_cache[n + 5] = v5;
        n += 6;
    }
}
