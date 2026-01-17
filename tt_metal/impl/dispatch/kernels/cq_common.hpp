// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_config.h"
#include "internal/risc_attribs.h"
#include "api/dataflow/dataflow_api.h"
#include "cq_helpers.hpp"

#include "internal/debug/sanitize.h"
#include "api/debug/assert.h"
#include <limits>

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
// Similar to the above function but this one takes noc-xy coordinates as a separate argument to permit 64-bit
// addressing at NOC tile
template <
    enum CQNocFlags flags,
    enum CQNocWait wait = CQ_NOC_WAIT,
    enum CQNocSend send = CQ_NOC_SEND,
    uint32_t cmd_buf = NCRISC_WR_CMD_BUF,
    bool update_counters = false>
FORCE_INLINE void cq_noc_async_wwrite_with_state(
    uint32_t src_addr,
    uint32_t dst_noc_addr,
    uint64_t dst_addr,
    uint32_t size = 0,
    uint32_t ndests = 1,
    uint8_t noc = noc_index) {
    if constexpr (wait) {
        WAYPOINT("CNSW");
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        WAYPOINT("CNSD");
    }
    noc_wwrite_with_state<DM_DEDICATED_NOC, cmd_buf, flags, CQ_NOC_send, CQ_NOC_wait, false>(
        noc, src_addr, dst_noc_addr, dst_addr, size, ndests);
    if constexpr (send) {
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc, cmd_buf);
        noc_wwrite_with_state<DM_DEDICATED_NOC, cmd_buf, CQ_NOC_sndl, send, CQ_NOC_wait, update_counters>(
            noc, src_addr, dst_noc_addr, dst_addr, size, ndests);
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
// Similar to the above function but this one takes noc-xy coordinates as a separate argument to permit 64-bit
// addressing at NOC tile
template <enum CQNocFlags flags, bool mcast = false, bool linked = false, uint32_t cmd_buf = NCRISC_WR_CMD_BUF>
FORCE_INLINE void cq_noc_async_wwrite_init_state(
    uint32_t src_addr, uint32_t dst_noc_addr, uint64_t dst_addr, uint32_t size = 0, uint8_t noc = noc_index) {
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
    cq_noc_async_wwrite_with_state<flags, CQ_NOC_wait, CQ_NOC_send, cmd_buf>(
        src_addr, dst_noc_addr, dst_addr, size, noc);
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
        noc_inline_dw_write_with_state<NCRISC_WR_REG_CMD_BUF, CQ_NOC_INLINE_ndvb, CQ_NOC_wait, send>(
            noc, dst_addr, val, be);
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

template <
    uint32_t my_sem_id,
    uint8_t noc_idx,
    uint32_t downstream_noc_xy,
    uint32_t downstream_sem_id,
    uint32_t buffer_base = 0,
    uint32_t buffer_end = 0,
    uint32_t buffer_page_size = 0>
class CBWriter {
public:
    FORCE_INLINE void acquire_pages(uint32_t n) {
        volatile tt_l1_ptr uint32_t* sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_sem_id));

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

    // Wait for all n pages to be available. If the consumer is using blocks, it may never return all pages at once
    // unless it calls release_all_pages to return partially-consumed blocks.
    FORCE_INLINE void wait_all_pages(uint32_t n) {
        volatile tt_l1_ptr uint32_t* sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_sem_id));

        // Downstream component sets the MSB as a terminate bit
        // Mask that off to avoid a race between the sem count and terminate
        n &= 0x7fffffff;

        WAYPOINT("TAPW");
        do {
            invalidate_l1_cache();
        } while (((additional_count + *sem_addr) & 0x7fffffff) != n);  // mask off terminate bit
        WAYPOINT("TAPD");
    }

    // Inform the consumer that n pages are available.
    FORCE_INLINE void release_pages(uint32_t n, uint32_t writer_ptr = 0, bool round_to_page_size = false) {
#if ASSERT_ENABLED
        if constexpr (buffer_page_size != 0) {
            constexpr uint32_t buffer_size = buffer_end - buffer_base;
            if constexpr (buffer_size != 0) {
                if (n != 0) {
                    // In the middle of writing a command, the writer pointer may not be aligned to the page size, but
                    // must always be past the number of pages released.
                    uint32_t adjusted_writer_ptr =
                        round_to_page_size ? (writer_ptr - (writer_ptr - buffer_base) % buffer_page_size) : writer_ptr;
                    uint64_t bytes = n * buffer_page_size;
                    uint32_t expected = watch_released_ptr_ + bytes;
                    if (expected > buffer_end) {
                        expected -= buffer_size;
                    }
                    // It's possible the writer_ptr wrapped and the expected pointer is at the very end of the buffer so
                    // it hasn't wrapped yet.
                    ASSERT((adjusted_writer_ptr == expected) || ((expected == buffer_end) && (adjusted_writer_ptr == buffer_base)));
                    watch_released_ptr_ = expected;
                }
            }
        }
#endif
        noc_semaphore_inc(
            get_noc_addr_helper(downstream_noc_xy, get_semaphore<fd_core_type>(downstream_sem_id)), n, noc_idx);
    }

    uint32_t additional_count{0};

#if ASSERT_ENABLED
private:
    // Pointer to the end of the last released page. Used for watcher assertions.
    uint32_t watch_released_ptr_{buffer_base};
#endif
};

// CBReader
// Lightweight reader for a semaphore-backed circular buffer (command/data ring).
// Responsibilities:
//  - Tracks producer credits via an upstream semaphore and maintains a local consumer count.
//  - Advances a byte-address "fence" that limits consumption to the current block boundary.
//  - Uses per-block next-start addresses to bound processing and handle wrap-around of the ring.
//  - Provides non-blocking availability via acquire_pages() and a blocking drain via wait_all_pages().
// Notes:
//  - This class only accounts for pages locally; it does NOT release credits back to the producer.
//    Use CBReaderWithReleasePolicy or CBReaderWithManualRelease when credits must be returned.
//  - Credits are returned per-block, not per-page.
template <
    uint32_t my_sem_id,
    uint32_t cb_log_page_size,
    uint32_t cb_blocks,
    uint32_t cb_pages_per_block,
    uint32_t cb_base>
class CBReader {
public:
    FORCE_INLINE void wait_all_pages() {
        volatile tt_l1_ptr uint32_t* sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_sem_id));

        uint32_t to_wait_for = upstream_count_;

        // Downstream component sets the MSB as a terminate bit
        // Mask that off to avoid a race between the sem count and terminate
        to_wait_for &= 0x7fffffff;

        WAYPOINT("TAPW");
        do {
            invalidate_l1_cache();
        } while ((*sem_addr & 0x7fffffff) != to_wait_for);  // mask off terminate bit
        WAYPOINT("TAPD");
    }

    // Return available space (in bytes) after data_ptr. This data will always be contiguous in memory and will never
    // wrap around.
    uint32_t available_bytes(uint32_t data_ptr) const { return cb_fence_ - data_ptr; }

protected:
    FORCE_INLINE void init() {
        for (uint32_t i = 0; i < cb_blocks; i++) {
            uint32_t next_block = i + 1;
            uint32_t offset = next_block * cb_pages_per_block * (1 << cb_log_page_size);
            this->block_next_start_addr_[i] = cb_base + offset;
        }

        this->cb_fence_ = cb_base;
        this->rd_block_idx_ = 0;
    }

    // Advance the block to the next index, wrapping around if necessary.
    FORCE_INLINE void move_rd_to_next_block() {
        static_assert((cb_blocks & (cb_blocks - 1)) == 0);
        rd_block_idx_++;
        rd_block_idx_ &= cb_blocks - 1;
    }

    // Acquire pages from upstream. Updates the cb_fence and returns the number of pages acquired. May block waiting for
    // credits from upstream if we already acquired all the pages previously.
    FORCE_INLINE uint32_t acquire_pages() {
        volatile tt_l1_ptr uint32_t* sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_sem_id));

        if (local_count_ == upstream_count_) {
            WAYPOINT("UAPW");
            uint32_t heartbeat = 0;
            do {
                invalidate_l1_cache();
                IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, 0);
            } while ((upstream_count_ = *sem_addr) == local_count_);
            WAYPOINT("UAPD");
        }

        // Set a fence to limit how much is processed at once
        uint32_t limit = (block_next_start_addr_[rd_block_idx_] - cb_fence_) >> cb_log_page_size;
        uint32_t available = upstream_count_ - local_count_;
        uint32_t usable = (available > limit) ? limit : available;

        local_count_ += usable;
        cb_fence_ += usable << cb_log_page_size;

        return usable;
    }

    // Byte address fence delimiting the end of currently usable data (do not process beyond this address).
    uint32_t cb_fence_{0};
    // Byte addresses of the start of the next block for each block index; used to cap processing per block and wrap.
    uint32_t block_next_start_addr_[cb_blocks]{};
    // Current read block index within the circular buffer.
    uint32_t rd_block_idx_{0};

private:
    // Last value read from the upstream semaphore (producer credits). Cached snapshot for availability checks.
    uint32_t upstream_count_{0};
    // Number of pages this reader has already accounted for (consumed) into the cb_fence_ region.
    uint32_t local_count_{0};
};

// Reader that releases a block of pages at a time. At most one block of pages can be available at a time.
template <
    uint32_t my_sem_id,
    uint32_t cb_log_page_size,
    uint32_t cb_blocks,
    uint8_t noc_idx,
    uint32_t noc_xy,
    uint32_t sem_id,
    uint32_t cb_pages_per_block,
    uint32_t cb_base,
    typename ReleasePolicy>
class CBReaderWithReleasePolicy : public CBReader<my_sem_id, cb_log_page_size, cb_blocks, cb_pages_per_block, cb_base> {
public:
    FORCE_INLINE void init() {
        this->CBReader<my_sem_id, cb_log_page_size, cb_blocks, cb_pages_per_block, cb_base>::init();
        this->block_noc_writes_to_clear_ = noc_nonposted_writes_num_issued[noc_index];
    }

    // Returns how much data is available. Will block until data is available. May release old pages before cmd_ptr to
    // writer. Updates cmd_ptr on wrap-around.
    // noc_nonposted_writes_num_issued[noc_index] must be updated before calling this function.
    // If this function doesn't return sufficient data, there are two options:
    // 1. Process all the available data and then call this function again.
    // 2. Call get_cb_page_and_release_pages to attempt to get more data.
    FORCE_INLINE uint32_t wait_for_available_data_and_release_old_pages(uint32_t& cmd_ptr) {
        if (this->available_bytes(cmd_ptr) == 0) {
            if (this->cb_fence_ == this->block_next_start_addr_[this->rd_block_idx_]) {
                if (this->rd_block_idx_ == cb_blocks - 1) {
                    cmd_ptr = cb_base;
                    this->cb_fence_ = cb_base;
                }
                move_rd_to_next_block_and_release_pages();
            }
            this->acquire_pages();
        }
        return this->available_bytes(cmd_ptr);
    }

    // Get new CB pages. If getting new pages would require switching the the next block, this will call on_boundary to
    // handle the orphan data that would otherwise be lost and will then release old pages to writer.
    //
    // The argument to on_boundary is whether the next block is the first block in the circular buffer (in which case
    // cmd_ptr is set to the base address after on_boundary is called).  noc_nonposted_writes_num_issued[noc_index] must
    // be updated before on_boundary returns.
    template <typename OnBoundaryFn>
    FORCE_INLINE uint32_t get_cb_page_and_release_pages(uint32_t& cmd_ptr, OnBoundaryFn&& on_boundary) {
        if (this->cb_fence_ == this->block_next_start_addr_[this->rd_block_idx_]) {
            const bool will_wrap = (this->rd_block_idx_ == cb_blocks - 1);
            on_boundary(will_wrap);
            if (will_wrap) {
                cmd_ptr = cb_base;
                this->cb_fence_ = cb_base;
            }
            move_rd_to_next_block_and_release_pages();
        }
        return this->acquire_pages();
    }

    FORCE_INLINE void release_all_pages(uint32_t curr_ptr) {
        release_block_pages();
        uint32_t pages_to_release =
            cb_pages_per_block - ((this->block_next_start_addr_[this->rd_block_idx_] - curr_ptr) >> cb_log_page_size);
        if (pages_to_release != 0) {
            ReleasePolicy::template release<noc_idx, noc_xy, sem_id>(pages_to_release);
        }
    }

private:
    FORCE_INLINE void release_block_pages() {
        // When finishing a block, don't immediately return it, but just store the number of nonposted write requests
        // issued. We will wait for writes to be sent and will return the  block when the next block is finished; this
        // allows time for writes from that block to complete. Note: this is incorrect if writes can be sent out of
        // order. We should use transaction IDs instead in that case.
        if (released_prev_block_) {
            WAYPOINT("CBRW");
            while (!wrap_ge(
                NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT), this->block_noc_writes_to_clear_));
            ReleasePolicy::template release<noc_idx, noc_xy, sem_id>(cb_pages_per_block);
            WAYPOINT("CBRD");
        } else {
            released_prev_block_ = true;
        }
        this->block_noc_writes_to_clear_ = noc_nonposted_writes_num_issued[noc_index];
    }

    FORCE_INLINE void move_rd_to_next_block_and_release_pages() {
        release_block_pages();
        this->move_rd_to_next_block();
    }

    bool released_prev_block_{false};
    // Snapshot of nonposted write requests issued at block entry; used to wait/clear before releasing credits.
    uint32_t block_noc_writes_to_clear_{0};
};

template <
    uint32_t my_sem_id,
    uint32_t cb_log_page_size,
    uint32_t cb_blocks,
    uint32_t cb_pages_per_block,
    uint32_t cb_base,
    uint32_t cb_end>
class CBReaderWithManualRelease : public CBReader<my_sem_id, cb_log_page_size, cb_blocks, cb_pages_per_block, cb_base> {
public:
    FORCE_INLINE void init() {
        this->CBReader<my_sem_id, cb_log_page_size, cb_blocks, cb_pages_per_block, cb_base>::init();
    }

    // Get a new CB page. Will update cmd_ptr on wrap-around. Returns the number of pages acquired. Will not release
    // pages to writer.
    FORCE_INLINE uint32_t get_cb_page(uint32_t& cmd_ptr) {
        // Strided past the data that has arrived, get the next page
        if (this->cb_fence_ == this->block_next_start_addr_[this->rd_block_idx_]) {
            if (this->rd_block_idx_ == cb_blocks - 1) {
                cmd_ptr = cb_base;
                this->cb_fence_ = cb_base;
            }
            this->move_rd_to_next_block();
        }

        return this->acquire_pages();
    }

    // Returns how much data is available. Will block until data is available.
    FORCE_INLINE uint32_t wait_for_available_data(uint32_t& cmd_ptr) {
        if (this->available_bytes(cmd_ptr) == 0) {
            get_cb_page(cmd_ptr);
        }
        return this->available_bytes(cmd_ptr);
    }

    // Advance cmd_ptr by length. If we wrap around, wrap the fence (should only happen if we hit the end exactly).
    FORCE_INLINE void consumed_data(uint32_t& cmd_ptr, uint32_t length) {
        // This is ugly: get_cb_page code can wrap and this can wrap
        // They peacefully coexist because we won't wrap there and here at once
        if (cmd_ptr + length >= cb_end) {
            length -= cb_end - cmd_ptr;
            cmd_ptr = cb_base;
            if (this->cb_fence_ == cb_end) {
                // We hit the nail on the head, wrap the fence
                ASSERT(length == 0);
                this->cb_fence_ = cb_base;
                // TODO eliminate usage of block_next_start_addr_ in this CB reader. rd_block_idx_ will point to the
                // last block, not the first block, so the limit calculation in acquire_pages will be incorrect. We
                // don't really use blocks for anything, here, so we should get rid of them and simplify the code.
            }
        }
        cmd_ptr += length;
    }
};

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
