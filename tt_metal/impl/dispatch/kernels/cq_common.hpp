// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_config.h"
#include "risc_attribs.h"
#include "dataflow_api.h"
#include "cq_helpers.hpp"

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

// The fast CQ noc commands write a subset of the NOC registers for each transaction
// leveraging the fact that many transactions re-use certain values (eg, length)
// Since there are a variety of dispatch paradigms, which values get reused
// depend on the fn
// Making template fns w/ a long list of booleans makes understanding what
// is/not sent tedious
// This is an attempt to pack that data in a way thats ~easy to visually parse
// S/s: send, do not send src address
// N/n: send, do not send noc address
// D/d: send, do not send dst address
// L/l: send, do not send length
constexpr uint32_t CQ_NOC_FLAG_SRC = 0x01;
constexpr uint32_t CQ_NOC_FLAG_NOC = 0x02;
constexpr uint32_t CQ_NOC_FLAG_DST = 0x04;
constexpr uint32_t CQ_NOC_FLAG_LEN = 0x08;

constexpr uint32_t CQ_NOC_INLINE_FLAG_VAL = 0x10;
constexpr uint32_t CQ_NOC_INLINE_FLAG_BE = 0x20;

enum CQNocFlags {
    CQ_NOC_sndl = 0,
    CQ_NOC_sndL = CQ_NOC_FLAG_LEN,
    CQ_NOC_snDl = CQ_NOC_FLAG_DST,
    CQ_NOC_snDL = CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNdl = CQ_NOC_FLAG_NOC,
    CQ_NOC_sNdL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNDl = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_sNDL = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_Sndl = CQ_NOC_FLAG_SRC,
    CQ_NOC_SndL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SnDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST,
    CQ_NOC_SnDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNdl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC,
    CQ_NOC_SNdL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_SNDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
};

enum CQNocInlineFlags {
    CQ_NOC_INLINE_ndvb = 0,
    CQ_NOC_INLINE_ndvB = CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_ndVb = CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_ndVB = CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDvb = CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_nDvB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_nDVb = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_nDVB = CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_Ndvb = CQ_NOC_FLAG_NOC,
    CQ_NOC_INLINE_NdvB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NdVb = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NdVB = CQ_NOC_FLAG_NOC | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDvb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_INLINE_NDvB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_BE,
    CQ_NOC_INLINE_NDVb = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL,
    CQ_NOC_INLINE_NDVB = CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_INLINE_FLAG_VAL | CQ_NOC_INLINE_FLAG_BE,
};

enum CQNocWait {
    CQ_NOC_wait = 0,
    CQ_NOC_WAIT = 1,
};
enum CQNocSend {
    CQ_NOC_send = 0,
    CQ_NOC_SEND = 1,
};

constexpr bool use_fabric(uint64_t fabric_router_xy) { return fabric_router_xy != 0; }

template <
    enum CQNocFlags flags,
    enum CQNocWait wait = CQ_NOC_WAIT,
    enum CQNocSend send = CQ_NOC_SEND,
    uint32_t cmd_buf = NCRISC_WR_CMD_BUF>
FORCE_INLINE void cq_noc_async_write_with_state(
    uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1, uint8_t noc = noc_index) {
    if constexpr (wait) {
        WAYPOINT("CNSW");
        while (!noc_cmd_buf_ready(noc, cmd_buf));
        WAYPOINT("CNSD");
    }

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
#ifdef ARCH_BLACKHOLE
        // Handles writing to PCIe
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32) & 0x1000000F);
#endif
        NOC_CMD_BUF_WRITE_REG(
            noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        ASSERT(size <= NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, size);
    }
    if constexpr (send) {
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc, cmd_buf);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
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
        cq_noc_async_write_with_state<CQ_NOC_SnDL, wait_first, CQ_NOC_SEND, cmd_buf>(
            src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests);
        src_addr += NOC_MAX_BURST_SIZE;
        dst_addr += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
        if constexpr (update_counters) {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += ndests;
        }
        while (size > NOC_MAX_BURST_SIZE) {
            cq_noc_async_write_with_state<CQ_NOC_SnDl, CQ_NOC_WAIT, CQ_NOC_SEND, cmd_buf>(
                src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests, noc);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
            if constexpr (update_counters) {
                noc_nonposted_writes_num_issued[noc] += 1;
                noc_nonposted_writes_acked[noc] += ndests;
            }
        }
    }
    if constexpr (write_last_packet) {
        cq_noc_async_write_with_state<CQ_NOC_SnDL, CQ_NOC_WAIT, CQ_NOC_SEND, cmd_buf>(
            src_addr, dst_addr, size, ndests, noc);
        if constexpr (update_counters) {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += ndests;
        }
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

    constexpr bool multicast_path_reserve = true;
    constexpr bool posted = false;
    constexpr uint32_t vc = mcast ? NOC_DISPATCH_MULTICAST_WRITE_VC : NOC_UNICAST_WRITE_VC;

    constexpr uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);

    cq_noc_async_write_with_state<flags, CQ_NOC_wait, CQ_NOC_send, cmd_buf>(src_addr, dst_addr, size);
}

template <enum CQNocInlineFlags flags, enum CQNocWait wait = CQ_NOC_WAIT, enum CQNocSend send = CQ_NOC_SEND>
FORCE_INLINE void cq_noc_inline_dw_write_with_state(
    uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF, uint8_t noc = noc_index) {
    if constexpr (wait) {
        WAYPOINT("NISW");
        while (!noc_cmd_buf_ready(noc, NCRISC_WR_REG_CMD_BUF));
        WAYPOINT("NISD");
    }

    if constexpr (flags & CQ_NOC_INLINE_FLAG_VAL) {
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_AT_DATA, val);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_LO, (uint32_t)(dst_addr));
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
#ifdef ARCH_BLACKHOLE
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_TARG_ADDR_MID, (uint32_t)(dst_addr >> 32) & 0x1000000F);
#endif
        NOC_CMD_BUF_WRITE_REG(
            noc,
            NCRISC_WR_REG_CMD_BUF,
            NOC_TARG_ADDR_COORDINATE,
            (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (flags & CQ_NOC_INLINE_FLAG_BE) {
        uint32_t be32 = be;
        uint32_t be_shift = (dst_addr & (NOC_WORD_BYTES - 1));
        be32 = (be32 << be_shift);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_AT_LEN_BE, be32);
    }
    if constexpr (send) {
        DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc, NCRISC_WR_REG_CMD_BUF);
        NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}

// TODO: noc_inline_dw_write currently hardcodes most of these parameters, which we copied here
// If needed, add templates for setting these
// TODO: uplift for BH to not do inline write
template <enum CQNocInlineFlags flags>
FORCE_INLINE void cq_noc_inline_dw_write_init_state(
    uint64_t dst_addr, uint32_t val = 0, uint8_t be = 0xF, uint8_t noc = noc_index) {
    WAYPOINT("NIIW");
    uint32_t heartbeat = 0;
    while (!noc_cmd_buf_ready(noc, NCRISC_WR_REG_CMD_BUF)) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    WAYPOINT("NIID");

    constexpr bool static_vc_alloc = true;
    constexpr bool mcast = false;
    constexpr bool posted = false;
    constexpr uint32_t static_vc = NOC_UNICAST_WRITE_VC;

    constexpr uint32_t noc_cmd_field = (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) | NOC_CMD_STATIC_VC(static_vc) |
                                       NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_WR_INLINE |
                                       (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
                                       (posted ? 0x0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_WR_REG_CMD_BUF, NOC_CTRL, noc_cmd_field);

    cq_noc_inline_dw_write_with_state<flags, CQ_NOC_wait, CQ_NOC_send>(dst_addr, val, be);
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

template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id, uint32_t cb_pages_per_block>
FORCE_INLINE void cb_block_release_pages(uint32_t& block_noc_writes_to_clear, uint8_t noc = noc_index) {
    // Do not release pages on the first call to this function
    // This is because the first call means we don't have a previous block to release
    static bool prev_block = false;
    if (prev_block) {
        WAYPOINT("CBRW");
        uint32_t sem_addr = get_semaphore<fd_core_type>(sem_id);
        while (!wrap_ge(NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT), block_noc_writes_to_clear));
        noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block, noc_idx);
        WAYPOINT("CBRD");
    } else {
        prev_block = true;
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
