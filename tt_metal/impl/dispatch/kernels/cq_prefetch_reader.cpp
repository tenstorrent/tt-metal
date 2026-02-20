// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch reader kernel (split-prefetcher BRISC half)
//  - Handles the _hd and _d variants; _h variant is excluded (reader-only role)
//  - Reads commands from host / upstream and processes them; write-path work will move to
//    cq_prefetch_writer.cpp (NCRISC) in future iterations
//  - Command-implementing functions are currently stubbed; their bodies will migrate to the writer
//
// Write cmd buf allocation:
//  - BRISC_WR_CMD_BUF: writes to downstream_noc_xy
//  - BRISC_WR_REG_CMD_BUF: small writes to dispatch_s_noc_xy. not much traffic on this path.
//
//  Using the normal NoC APIs for writes and/or inline_dw_writes are not allowed on this kernel.
//

#include "api/dataflow/dataflow_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_relay.hpp"
#include "api/debug/dprint.h"
#include "noc/noc_parameters.h"  // PCIE_ALIGNMENT

constexpr uint32_t CQ_PREFETCH_CMD_BARE_MIN_SIZE = PCIE_ALIGNMENT;  // for NOC PCIe alignemnt
static_assert(sizeof(CQPrefetchCmd) <= CQ_PREFETCH_CMD_BARE_MIN_SIZE);
static_assert(sizeof(CQPrefetchCmdLarge) <= CQ_PREFETCH_CMD_BARE_MIN_SIZE);
struct CQPrefetchHToPrefetchDHeader_s {
    uint64_t length;
    uint8_t raw_copy;  // If true, copy the data directly to the downstream.
};
union CQPrefetchHToPrefetchDHeader {
    CQPrefetchHToPrefetchDHeader_s header;
    unsigned char padding[CQ_PREFETCH_CMD_BARE_MIN_SIZE];
};
static_assert((sizeof(CQPrefetchHToPrefetchDHeader) & (CQ_PREFETCH_CMD_BARE_MIN_SIZE - 1)) == 0);

using prefetch_q_entry_type = uint16_t;

// Use named defines instead of get_compile_time_arg_val indices
constexpr uint32_t downstream_cb_base = DOWNSTREAM_CB_BASE;
constexpr uint32_t downstream_cb_log_page_size = DOWNSTREAM_CB_LOG_PAGE_SIZE;
constexpr uint32_t downstream_cb_pages = DOWNSTREAM_CB_PAGES;
constexpr uint32_t my_downstream_cb_sem_id = MY_DOWNSTREAM_CB_SEM_ID;
constexpr uint32_t downstream_cb_sem_id = DOWNSTREAM_CB_SEM_ID;

// unused for prefetch_d
constexpr uint32_t pcie_base = PCIE_BASE;
constexpr uint32_t pcie_size = PCIE_SIZE;
constexpr uint32_t prefetch_q_base = PREFETCH_Q_BASE;
constexpr uint32_t prefetch_q_size = PREFETCH_Q_SIZE;
constexpr uint32_t prefetch_q_rd_ptr_addr = PREFETCH_Q_RD_PTR_ADDR;
constexpr uint32_t prefetch_q_pcie_rd_ptr_addr = PREFETCH_Q_PCIE_RD_PTR_ADDR;

constexpr uint32_t cmddat_q_base = CMDDAT_Q_BASE;
constexpr uint32_t cmddat_q_size = CMDDAT_Q_SIZE;

// unused for prefetch_h
constexpr uint32_t scratch_db_base = SCRATCH_DB_BASE;
constexpr uint32_t scratch_db_size = SCRATCH_DB_SIZE;
constexpr uint32_t my_downstream_sync_sem_id = DOWNSTREAM_SYNC_SEM_ID;

// prefetch_d specific
constexpr uint32_t cmddat_q_pages = CMDDAT_Q_PAGES;
constexpr uint32_t my_upstream_cb_sem_id = MY_UPSTREAM_CB_SEM_ID;
constexpr uint32_t upstream_cb_sem_id = UPSTREAM_CB_SEM_ID;
constexpr uint32_t cmddat_q_log_page_size = CMDDAT_Q_LOG_PAGE_SIZE;
constexpr uint32_t cmddat_q_blocks = CMDDAT_Q_BLOCKS;

// used for prefetch_d <--> dispatch_s data path
constexpr uint32_t dispatch_s_buffer_base = DISPATCH_S_BUFFER_BASE;
constexpr uint32_t my_dispatch_s_cb_sem_id = MY_DISPATCH_S_CB_SEM_ID;
constexpr uint32_t downstream_dispatch_s_cb_sem_id = DOWNSTREAM_DISPATCH_S_CB_SEM_ID;
constexpr uint32_t dispatch_s_buffer_size = DISPATCH_S_BUFFER_SIZE;
constexpr uint32_t dispatch_s_cb_log_page_size = DISPATCH_S_CB_LOG_PAGE_SIZE;

constexpr uint32_t ringbuffer_size = RINGBUFFER_SIZE;

// fabric mux connection
constexpr uint32_t fabric_header_rb_base = FABRIC_HEADER_RB_BASE;
constexpr uint32_t fabric_header_rb_entries = FABRIC_HEADER_RB_ENTRIES;
constexpr uint32_t my_fabric_sync_status_addr = MY_FABRIC_SYNC_STATUS_ADDR;

constexpr uint8_t fabric_mux_x = FABRIC_MUX_X;
constexpr uint8_t fabric_mux_y = FABRIC_MUX_Y;
constexpr uint8_t fabric_mux_num_buffers_per_channel = FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL;
constexpr size_t fabric_mux_channel_buffer_size_bytes = FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES;
constexpr size_t fabric_mux_channel_base_address = FABRIC_MUX_CHANNEL_BASE_ADDRESS;
constexpr size_t fabric_mux_connection_info_address = FABRIC_MUX_CONNECTION_INFO_ADDRESS;
constexpr size_t fabric_mux_connection_handshake_address = FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS;
constexpr size_t fabric_mux_flow_control_address = FABRIC_MUX_FLOW_CONTROL_ADDRESS;
constexpr size_t fabric_mux_buffer_index_address = FABRIC_MUX_BUFFER_INDEX_ADDRESS;
constexpr size_t fabric_mux_status_address = FABRIC_MUX_STATUS_ADDRESS;
constexpr size_t fabric_mux_termination_signal_address = FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS;
constexpr size_t worker_credits_stream_id = WORKER_CREDITS_STREAM_ID;

constexpr size_t fabric_worker_flow_control_sem = FABRIC_WORKER_FLOW_CONTROL_SEM;
constexpr size_t fabric_worker_teardown_sem = FABRIC_WORKER_TEARDOWN_SEM;
constexpr size_t fabric_worker_buffer_index_sem = FABRIC_WORKER_BUFFER_INDEX_SEM;

constexpr uint8_t num_hops = NUM_HOPS;

constexpr uint32_t ew_dim = EW_DIM;
constexpr uint32_t to_mesh_id = TO_MESH_ID;

constexpr bool is_2d_fabric = FABRIC_2D;

constexpr uint32_t is_d_variant = IS_D_VARIANT;
constexpr uint32_t is_h_variant = IS_H_VARIANT;

constexpr uint32_t prefetch_q_end = prefetch_q_base + prefetch_q_size;
constexpr uint32_t cmddat_q_end = cmddat_q_base + cmddat_q_size;
constexpr uint32_t scratch_db_end = scratch_db_base + scratch_db_size;
constexpr uint32_t ringbuffer_end = scratch_db_base + ringbuffer_size;

// hd and h: fetch_q, cmddat_q, scratch_db
static_assert(
    !(is_h_variant) || (prefetch_q_base >= cmddat_q_end || cmddat_q_base >= prefetch_q_end),
    "prefetch_q and cmddat_q overlap");

static_assert(
    !(is_h_variant) || (prefetch_q_base >= scratch_db_end || scratch_db_base >= prefetch_q_end),
    "prefetch_q and scratch_db overlap");

static_assert(
    !(is_h_variant) || (scratch_db_base >= cmddat_q_end || cmddat_q_base >= scratch_db_end),
    "cmddat_q and scratch_db overlap");

// d: cmddat_q, scratch_db
static_assert(
    !(is_d_variant && !is_h_variant) || (scratch_db_base >= cmddat_q_end || cmddat_q_base >= scratch_db_end),
    "cmddat_q and scratch_db overlap");

constexpr uint8_t my_noc_index = NOC_INDEX;
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t dispatch_s_noc_xy =
    uint32_t(NOC_XY_ENCODING(DOWNSTREAM_SUBORDINATE_NOC_X, DOWNSTREAM_SUBORDINATE_NOC_Y));
constexpr uint64_t pcie_noc_xy =
    uint64_t(NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(PCIE_NOC_X), NOC_Y_PHYS_COORD(PCIE_NOC_Y)));
constexpr uint32_t downstream_cb_page_size = 1 << downstream_cb_log_page_size;
constexpr uint32_t dispatch_s_cb_page_size = 1 << dispatch_s_cb_log_page_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + (1 << downstream_cb_log_page_size) * downstream_cb_pages;
constexpr uint32_t dispatch_s_buffer_end = dispatch_s_buffer_base + dispatch_s_buffer_size;
constexpr uint32_t cmddat_q_page_size = 1 << cmddat_q_log_page_size;

constexpr uint32_t scratch_db_half_size = scratch_db_size / 2;
constexpr uint32_t scratch_db_base0 = scratch_db_base;
constexpr uint32_t scratch_db_base1 = scratch_db_base + scratch_db_half_size;

constexpr uint32_t prefetch_q_log_minsize = 4;

const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

constexpr uint32_t cmddat_q_pages_per_block = cmddat_q_pages / cmddat_q_blocks;

// Currently capping the same as dispatch
constexpr uint32_t max_read_packed_cmd =
    CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS * sizeof(CQPrefetchRelayPagedPackedSubCmd) / sizeof(uint32_t);
constexpr uint32_t l1_cache_elements = max_read_packed_cmd + 1;  // +1 for sentinel value
constexpr uint32_t l1_cache_elements_rounded =
    ((l1_cache_elements + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
        l1_to_local_cache_copy_chunk +
    (l1_to_local_cache_copy_chunk - 1);

static_assert(
    CQ_PREFETCH_CMD_RELAY_RINGBUFFER_MAX_SUB_CMDS * sizeof(CQPrefetchRelayRingbufferSubCmd) / sizeof(uint32_t) <
        l1_cache_elements_rounded,
    "CQ_PREFETCH_CMD_RELAY_RINGBUFFER_MAX_SUB_CMDS is too large for l1_cache_elements_rounded");

// Define these constexpr structs for a cleaner interface for process_relay_inline_cmd and
// process_exec_buf_relay_inline_cmd while ensuring that state for dispatch_master and dispatch_subordinate is passed in
// during compile time.
struct DispatchRelayInlineState {
    static constexpr uint32_t my_downstream_cb_sem = my_downstream_cb_sem_id;
    static constexpr uint32_t downstream_cb_sem = downstream_cb_sem_id;
    static constexpr uint32_t downstream_noc_encoding = downstream_noc_xy;
    static constexpr uint32_t downstream_page_size = downstream_cb_page_size;
    static constexpr uint32_t downstream_log_page_size = downstream_cb_log_page_size;
    static constexpr uint32_t downstream_cb_base_addr = downstream_cb_base;
    static constexpr uint32_t downstream_cb_end_addr = downstream_cb_end;
    static constexpr uint32_t downstream_write_cmd_buf = BRISC_WR_CMD_BUF;
    static constexpr uint32_t downstream_noc_index = my_noc_index;
    static inline CBWriter<
        my_downstream_cb_sem,
        my_noc_index,
        downstream_noc_xy,
        downstream_cb_sem,
        downstream_cb_base,
        downstream_cb_end,
        downstream_cb_page_size>
        cb_writer{};
};

struct DispatchSRelayInlineState {
    static constexpr uint32_t my_downstream_cb_sem = my_dispatch_s_cb_sem_id;
    static constexpr uint32_t downstream_cb_sem = downstream_dispatch_s_cb_sem_id;
    static constexpr uint32_t downstream_noc_encoding = dispatch_s_noc_xy;
    static constexpr uint32_t downstream_page_size = dispatch_s_cb_page_size;
    static constexpr uint32_t downstream_log_page_size = dispatch_s_cb_log_page_size;
    static constexpr uint32_t downstream_cb_base_addr = dispatch_s_buffer_base;
    static constexpr uint32_t downstream_cb_end_addr = dispatch_s_buffer_end;
    static constexpr uint32_t downstream_write_cmd_buf = BRISC_WR_REG_CMD_BUF;
    static constexpr uint32_t downstream_noc_index = my_noc_index;
    static inline CBWriter<
        my_dispatch_s_cb_sem_id,
        my_noc_index,
        dispatch_s_noc_xy,
        downstream_dispatch_s_cb_sem_id,
        dispatch_s_buffer_base,
        dispatch_s_buffer_end,
        dispatch_s_cb_page_size>
        cb_writer{};
};

struct PrefetchExecBufState {
    uint32_t page_id;
    uint32_t base_addr;
    uint32_t log_page_size;
    uint32_t pages;
    uint32_t length;
    uint32_t read_ptr;
    uint32_t prefetch_length;
};

// Global Variables
static uint32_t pcie_read_ptr = pcie_base;
static uint32_t downstream_data_ptr = downstream_cb_base;
static uint32_t downstream_data_ptr_s = dispatch_s_buffer_base;
static uint32_t block_next_start_addr[cmddat_q_blocks];
static uint32_t rd_block_idx = 0;
static uint32_t upstream_total_acquired_page_count = 0;
static uint32_t ringbuffer_wp = scratch_db_base;
static uint32_t ringbuffer_offset = 0;

// Runtime args
static uint32_t my_dev_id;
static uint32_t to_dev_id;
static uint32_t router_direction;

CQRelayClient<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes, fabric_header_rb_base>
    relay_client;

// Feature to stall the prefetcher, mainly for ExecBuf impl which reuses CmdDataQ
static enum StallState { STALL_NEXT = 2, STALLED = 1, NOT_STALLED = 0 } stall_state = NOT_STALLED;

static_assert((downstream_cb_base & (downstream_cb_page_size - 1)) == 0);

template <bool cmddat_wrap_enable, bool exec_buf>
bool process_cmd(
    uint32_t& cmd_ptr,
    uint32_t& downstream_data_ptr,
    uint32_t& stride,
    uint32_t* l1_cache,
    PrefetchExecBufState& exec_buf_state);

template <uint32_t downstream_cb_base_addr, uint32_t downstream_cmd_buf>
FORCE_INLINE void write_downstream(
    uint32_t& data_ptr,
    uint32_t& local_downstream_data_ptr,
    uint32_t length,
    uint32_t downstream_end,
    uint32_t downstream_noc_encoding = downstream_noc_xy) {
    uint32_t remaining = downstream_end - local_downstream_data_ptr;
    if (length > remaining) {
        if (remaining > 0) {
#if defined(FABRIC_RELAY)
            noc_async_write(
                data_ptr, get_noc_addr_helper(downstream_noc_encoding, local_downstream_data_ptr), remaining);
#else
            cq_noc_async_write_with_state_any_len<true, true, CQNocWait::CQ_NOC_WAIT, downstream_cmd_buf>(
                data_ptr, get_noc_addr_helper(downstream_noc_encoding, local_downstream_data_ptr), remaining);
#endif
            data_ptr += remaining;
            length -= remaining;
        }
        local_downstream_data_ptr = downstream_cb_base_addr;
    }

#if defined(FABRIC_RELAY)
    noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_encoding, local_downstream_data_ptr), length);
#else
    cq_noc_async_write_with_state_any_len<true, true, CQNocWait::CQ_NOC_WAIT, downstream_cmd_buf>(
        data_ptr, get_noc_addr_helper(downstream_noc_encoding, local_downstream_data_ptr), length);
#endif
    local_downstream_data_ptr += length;
}

// If prefetcher must stall after this fetch, wait for data to come back, and move to stalled state.
FORCE_INLINE void barrier_and_stall(uint32_t& pending_read_size, uint32_t& fence, uint32_t& cmd_ptr) {
    noc_async_read_barrier();
    if (fence < cmd_ptr) {
        cmd_ptr = fence;
    }
    fence += pending_read_size;
    pending_read_size = 0;
    stall_state = STALLED;
}

template <uint32_t preamble_size>
FORCE_INLINE uint32_t read_from_pcie(
    volatile tt_l1_ptr prefetch_q_entry_type*& prefetch_q_rd_ptr,
    uint32_t& fence,
    uint32_t& pcie_read_ptr,
    uint32_t cmd_ptr,
    uint32_t size) {
    uint32_t pending_read_size = 0;
    // Wrap cmddat_q
    if (fence + size + preamble_size > cmddat_q_end) {
        // only wrap if there are no commands ready, otherwise we'll leave some on the floor
        // TODO: does this matter for perf?
        if (cmd_ptr != fence) {
            // No pending reads, since the location of fence cannot be moved due to unread commands
            // in the cmddat_q -> reads cannot be issued to fill the queue.
            return pending_read_size;
        }
        fence = cmddat_q_base;
    }

    // Wrap pcie/hugepage
    if (pcie_read_ptr + size > pcie_base + pcie_size) {
        pcie_read_ptr = pcie_base;
    }

    uint64_t host_src_addr = pcie_noc_xy | pcie_read_ptr;
    // DPRINT << "read_from_pcie: " << fence + preamble_size << " " << pcie_read_ptr << ENDL();
    noc_async_read(host_src_addr, fence + preamble_size, size);
    pending_read_size = size + preamble_size;
    pcie_read_ptr += size;

    *prefetch_q_rd_ptr = 0;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t*)prefetch_q_rd_ptr_addr = (uint32_t)prefetch_q_rd_ptr;
    *(volatile tt_l1_ptr uint32_t*)prefetch_q_pcie_rd_ptr_addr = (uint32_t)pcie_read_ptr;

    prefetch_q_rd_ptr++;

    // Wrap prefetch_q
    if ((uint32_t)prefetch_q_rd_ptr == prefetch_q_end) {
        prefetch_q_rd_ptr = (volatile tt_l1_ptr prefetch_q_entry_type*)prefetch_q_base;
    }
    return pending_read_size;
}

// This routine can be called in 8 states based on the boolean values cmd_ready, prefetch_q_ready, read_pending:
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier (and re-evaluate prefetch_q_ready)
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier
//  - !cmd_ready,  prefetch_q_ready,  read_pending: read barrier, issue read
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: exit (don't add latency to the in flight request)
//
// With WH tagging of reads:
// open question: should fetcher loop on prefetch_q_ready issuing reads until !prefetch_q_ready
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier on oldest tag
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier
//  - !cmd_ready,  prefetch_q_ready,  read_pending: issue read, read barrier on oldest tag
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue and tag read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: issue and tag read
template <uint32_t preamble_size>
void fetch_q_get_cmds(uint32_t& fence, uint32_t& cmd_ptr, uint32_t& pcie_read_ptr) {
    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr prefetch_q_entry_type* prefetch_q_rd_ptr =
        (volatile tt_l1_ptr prefetch_q_entry_type*)prefetch_q_base;
    constexpr uint32_t prefetch_q_msb_mask = 1u << (sizeof(prefetch_q_entry_type) * CHAR_BIT - 1);

    if (stall_state == STALLED) {
        ASSERT(pending_read_size == 0);  // Before stalling, fetch must have been completed.
        return;
    }

    // DPRINT << "fetch_q_get_cmds: " << cmd_ptr << " " << fence << ENDL();
    if (fence < cmd_ptr) {
        cmd_ptr = fence;
    }

    bool cmd_ready = (cmd_ptr != fence);

    uint32_t prefetch_q_rd_ptr_local = *prefetch_q_rd_ptr;
    uint32_t fetch_size = (prefetch_q_rd_ptr_local & ~prefetch_q_msb_mask) << prefetch_q_log_minsize;
    bool stall_flag = (prefetch_q_rd_ptr_local & prefetch_q_msb_mask) != 0;
    stall_state = static_cast<StallState>(stall_flag << 1);  // NOT_STALLED -> STALL_NEXT if stall_flag is set

    if (fetch_size != 0 && pending_read_size == 0) {
        pending_read_size = read_from_pcie<preamble_size>(prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
        if (stall_state == STALL_NEXT && pending_read_size != 0) {
            // No pending reads -> stall_state can be set to STALLED, since the read to the cmd
            // that initiated the stall has been issued.
            // exec_buf is the first command being fetched and should be offset
            // by preamble size. After ensuring that the exec_buf command has been read (barrier),
            // exit.
            barrier_and_stall(pending_read_size, fence, cmd_ptr);  // STALL_NEXT -> STALLED
            return;
        }
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            noc_async_read_barrier();
            // wrap the cmddat_q
            if (fence < cmd_ptr) {
                cmd_ptr = fence;
            }

            fence += pending_read_size;
            pending_read_size = 0;

            // After the stall, re-check the host
            prefetch_q_rd_ptr_local = *prefetch_q_rd_ptr;
            fetch_size = (prefetch_q_rd_ptr_local & ~prefetch_q_msb_mask) << prefetch_q_log_minsize;

            if (fetch_size != 0) {
                stall_flag = (prefetch_q_rd_ptr_local & prefetch_q_msb_mask) != 0;
                stall_state =
                    static_cast<StallState>(stall_flag << 1);  // NOT_STALLED -> STALL_NEXT if stall_flag is set

                if (stall_state == STALL_NEXT) {
                    // If the prefetcher state reached here, it is issuing a read to the same "slot", since for exec_buf
                    // commands we will insert a read barrier. Hence, the exec_buf command will be concatenated to a
                    // previous command, and should not be offset by preamble size.
                    pending_read_size = read_from_pcie<0>(prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
                    if (pending_read_size != 0) {
                        // if pending_read_size == 0 read_from_pcie early exited, due to a wrap, i.e. the exec_buf cmd
                        // is at a wrapped location, and a read to it could not be issued, since there are existing
                        // commands in the cmddat_q. Only move the stall_state to stalled if the read to the cmd that
                        // initiated the stall was issued
                        barrier_and_stall(pending_read_size, fence, cmd_ptr);  // STALL_NEXT -> STALLED
                    }
                } else {
                    pending_read_size =
                        read_from_pcie<preamble_size>(prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
                }
            }
        } else {
            // By here, prefetch_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            WAYPOINT("HQW");
            uint32_t heartbeat = 0;
            while ((fetch_size = *prefetch_q_rd_ptr) == 0) {
                invalidate_l1_cache();
                IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
            }
            fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);
            WAYPOINT("HQD");
        }
    }
}

uint32_t process_debug_cmd(uint32_t cmd_ptr) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->debug.stride;
}

template <bool cmddat_wrap_enable, typename RelayInlineState>
static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr, uint32_t& local_downstream_data_ptr) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
template <bool cmddat_wrap_enable>
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr, uint32_t& dispatch_data_ptr) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_inline.stride;
}

// The hard problem here is: when an xfer lands exactly at a page boundary, who is responsible for getting the next
// page? For inner loop, call N grabs page N+1.  No client should ever hit this as inline_noflush puts 16 bytes at the
// top of the first page At the end, do not grab page N+1
template <int32_t round, bool test_for_nonzero>
static uint32_t write_pages_to_dispatcher(
    uint32_t& downstream_data_ptr, uint32_t scratch_write_addr, uint32_t amt_to_write) {
    uint32_t page_residual_space = downstream_cb_page_size - (downstream_data_ptr & (downstream_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + downstream_cb_page_size - round) / downstream_cb_page_size;

    // Grabbing all pages at once is ok if scratch_size < 3 * downstream_cb_block_size
    // test_for_nonzero is an optimization: inner loops moving lots of pages don't bother
    if (!test_for_nonzero || npages != 0) {
        DispatchRelayInlineState::cb_writer.acquire_pages(npages);
    }

    uint64_t noc_addr;
    if (downstream_data_ptr == downstream_cb_end) {
        downstream_data_ptr = downstream_cb_base;
    } else if (downstream_data_ptr + amt_to_write > downstream_cb_end) {  // wrap
        uint32_t last_chunk_size = downstream_cb_end - downstream_data_ptr;
        noc_addr = get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr);
#if defined(FABRIC_RELAY)
        noc_async_write(scratch_write_addr, noc_addr, last_chunk_size);
#else
        cq_noc_async_write_with_state_any_len<true, true>(scratch_write_addr, noc_addr, last_chunk_size);
#endif
        downstream_data_ptr = downstream_cb_base;
        scratch_write_addr += last_chunk_size;
        amt_to_write -= last_chunk_size;
    }
    noc_addr = get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr);

#if defined(FABRIC_RELAY)
    noc_async_write(scratch_write_addr, noc_addr, amt_to_write);
#else
    cq_noc_async_write_with_state_any_len<true, true>(scratch_write_addr, noc_addr, amt_to_write);
#endif
    downstream_data_ptr += amt_to_write;

    return npages;
}

// This isn't the right way to handle large pages, but expedient for now
// In the future, break them down into smaller pages...
template <bool is_dram>
uint32_t process_relay_paged_cmd_large(
    uint32_t cmd_ptr,
    uint32_t& downstream__data_ptr,
    uint32_t page_id,
    uint32_t base_addr,
    uint32_t page_size,
    uint32_t pages,
    uint32_t length_adjust) {
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

// This fn prefetches data from DRAM memory and writes data to the dispatch core.
template <bool is_dram>
uint32_t process_relay_paged_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t page_id) {
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

// Similar to relay_paged, this iterates and aggregates reads from multiple embedded relay_paged cmds
void process_relay_paged_packed_sub_cmds(uint32_t total_length, uint32_t* l1_cache) {}

template <bool cmddat_wrap_enable>
uint32_t process_relay_paged_packed_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_paged_packed.stride;
}

template <bool set_src_noc_addr = false>
void noc_read_64bit_any_len(uint32_t src_noc_addr, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
    // noc_read_state_init is unnecessary.
    if constexpr (set_src_noc_addr) {
        noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_sNdL, CQ_NOC_send, CQ_NOC_WAIT>(
            noc_index, src_noc_addr, 0, 0, 0);
    } else {
        // wait on command buf to be ready before issuing new programming
        noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_sndl, CQ_NOC_send, CQ_NOC_WAIT>(
            noc_index, 0, 0, 0, 0);
    }
    if (size > NOC_MAX_BURST_SIZE) {
        // Set length to max burst size.
        noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_sndL, CQ_NOC_send, CQ_NOC_wait>(
            noc_index, 0, 0, 0, NOC_MAX_BURST_SIZE);
        while (size > NOC_MAX_BURST_SIZE) {
            noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_SnDl, CQ_NOC_SEND, CQ_NOC_wait>(
                noc_index, 0, src_addr, dst_addr, 0);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
            // Do a wait before either the next iteration or the final read.
            noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_sndl, CQ_NOC_send, CQ_NOC_WAIT>(
                noc_index, 0, 0, 0, 0);
        }
    }
    noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_SnDL, CQ_NOC_SEND, CQ_NOC_wait>(
        noc_index, 0, src_addr, dst_addr, size);
}

uint32_t process_relay_linear_cmd(uint32_t cmd_ptr, uint32_t& downstream_data_ptr) {
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_stall(uint32_t cmd_ptr) { return CQ_PREFETCH_CMD_BARE_MIN_SIZE; }

// This function reads data from the DRAM and populates the cmddat_q l1 buffer.
void paged_read_into_cmddat_q(uint32_t& cmd_ptr, PrefetchExecBufState& exec_buf_state) {}

// processes the relay_inline cmd from an exec_buf
// Separate implementation that fetches more data from exec buf when cmd has been split
template <typename RelayInlineState>
FORCE_INLINE static uint32_t process_exec_buf_relay_inline_cmd(
    uint32_t& cmd_ptr, uint32_t& local_downstream_data_ptr, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_inline_noflush_cmd(
    uint32_t& cmd_ptr, uint32_t& dispatch_data_ptr, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_inline.stride;
}

template <uint32_t cmd_header_size = sizeof(CQPrefetchCmd)>
void* copy_into_l1_cache(
    uint32_t& cmd_ptr,
    uint32_t sub_cmds_length,
    uint32_t* l1_cache,
    PrefetchExecBufState& exec_buf_state,
    uint32_t& stride) {
    return nullptr;
}

// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_paged_packed_cmd(
    uint32_t& cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_paged_packed.stride;
}

uint32_t process_exec_buf_cmd(
    uint32_t cmd_ptr_outer, uint32_t& downstream_data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_paged_to_ringbuffer_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr) {
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_set_ringbuffer_offset(uint32_t cmd_ptr) { return CQ_PREFETCH_CMD_BARE_MIN_SIZE; }

void process_relay_ringbuffer_sub_cmds(uint32_t count, uint32_t* l1_cache) {}

template <bool cmddat_wrap_enable>
uint32_t process_relay_ringbuffer_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_ringbuffer.stride;
}

// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_ringbuffer_cmd(
    uint32_t& cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_ringbuffer.stride;
}

void process_relay_linear_packed_sub_cmds(uint32_t noc_xy_addr, uint32_t total_length, uint32_t* l1_cache) {}

template <bool cmddat_wrap_enable>
uint32_t process_relay_linear_packed_cmd(uint32_t cmd_ptr, uint32_t& downstream_data_ptr, uint32_t* l1_cache) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_linear_packed.stride;
}

// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_linear_packed_cmd(
    uint32_t& cmd_ptr, uint32_t& downstream_data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    return cmd->relay_linear_packed.stride;
}

template <bool cmddat_wrap_enable, bool exec_buf>
bool process_cmd(
    uint32_t& cmd_ptr,
    uint32_t& downstream_data_ptr,
    uint32_t& stride,
    uint32_t* l1_cache,
    PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    bool done = false;

    switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_LINEAR:
            // DPRINT << "relay linear: " << cmd_ptr << ENDL();
            stride = process_relay_linear_cmd(cmd_ptr, downstream_data_ptr);
            break;

        case CQ_PREFETCH_CMD_RELAY_PAGED:
            // DPRINT << "relay paged: " << cmd_ptr << ENDL();
            {
                uint32_t is_dram_and_length_adjust = cmd->relay_paged.is_dram_and_length_adjust;
                uint32_t is_dram = is_dram_and_length_adjust & (1 << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT);
                uint32_t start_page = cmd->relay_paged.start_page;
                if (is_dram) {
                    stride = process_relay_paged_cmd<true>(cmd_ptr, downstream_data_ptr, start_page);
                } else {
                    stride = process_relay_paged_cmd<false>(cmd_ptr, downstream_data_ptr, start_page);
                }
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED:
            // DPRINT << "relay paged packed" << ENDL();
            if (exec_buf) {
                stride =
                    process_exec_buf_relay_paged_packed_cmd(cmd_ptr, downstream_data_ptr, l1_cache, exec_buf_state);
            } else {
                stride = process_relay_paged_packed_cmd<cmddat_wrap_enable>(cmd_ptr, downstream_data_ptr, l1_cache);
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE:
            // DPRINT << "relay inline" << ENDL();
            if constexpr (exec_buf) {
                if (cmd->relay_inline.dispatcher_type == DispatcherSelect::DISPATCH_MASTER) {
                    stride = process_exec_buf_relay_inline_cmd<DispatchRelayInlineState>(
                        cmd_ptr, downstream_data_ptr, exec_buf_state);
                } else {
                    stride = process_exec_buf_relay_inline_cmd<DispatchSRelayInlineState>(
                        cmd_ptr, downstream_data_ptr_s, exec_buf_state);
                }
            } else {
                if (cmd->relay_inline.dispatcher_type == DispatcherSelect::DISPATCH_MASTER) {
                    stride = process_relay_inline_cmd<cmddat_wrap_enable, DispatchRelayInlineState>(
                        cmd_ptr, downstream_data_ptr);
                } else {
                    stride = process_relay_inline_cmd<cmddat_wrap_enable, DispatchSRelayInlineState>(
                        cmd_ptr, downstream_data_ptr_s);
                }
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            // DPRINT << "inline no flush" << ENDL();
            if (exec_buf) {
                stride = process_exec_buf_relay_inline_noflush_cmd(cmd_ptr, downstream_data_ptr, exec_buf_state);
            } else {
                stride = process_relay_inline_noflush_cmd<cmddat_wrap_enable>(cmd_ptr, downstream_data_ptr);
            }
            break;

        case CQ_PREFETCH_CMD_EXEC_BUF:
            // DPRINT << "exec buf: " << cmd_ptr << ENDL();
            ASSERT(!exec_buf);
            if (is_h_variant) {
                ASSERT(stall_state == STALLED);  // ExecBuf must be preceded by a prefetcher stall
            }
            stride = process_exec_buf_cmd(cmd_ptr, downstream_data_ptr, l1_cache, exec_buf_state);
            stall_state = NOT_STALLED;  // Stall is no longer required after ExecBuf finished.
            break;

        case CQ_PREFETCH_CMD_EXEC_BUF_END:
            // DPRINT << "exec buf end: " << cmd_ptr << ENDL();
            ASSERT(exec_buf);
            stride = process_exec_buf_relay_inline_cmd<DispatchRelayInlineState>(
                cmd_ptr, downstream_data_ptr, exec_buf_state);
            done = true;
            break;

        case CQ_PREFETCH_CMD_STALL:
            // DPRINT << "stall" << ENDL();
            stride = process_stall(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_DEBUG:
            // DPRINT << "debug" << ENDL();
            //  Splitting debug cmds not implemented for exec_bufs (yet)
            if (exec_buf) {
                ASSERT(0);
            }
            stride = process_debug_cmd(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_TERMINATE:
            // DPRINT << "prefetch terminating_" << is_h_variant << is_d_variant << ENDL();
            ASSERT(!exec_buf);
            done = true;
            break;

        case CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER:
            // DPRINT << "paged to ringbuffer" << ENDL();
            stride = process_paged_to_ringbuffer_cmd(cmd_ptr, downstream_data_ptr);
            break;

        case CQ_PREFETCH_CMD_SET_RINGBUFFER_OFFSET:
            // DPRINT << "set ringbuffer offset" << ENDL();
            stride = process_set_ringbuffer_offset(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_RELAY_RINGBUFFER:
            // DPRINT << "relay ringbuffer" << ENDL();
            if (exec_buf) {
                stride = process_exec_buf_relay_ringbuffer_cmd(cmd_ptr, downstream_data_ptr, l1_cache, exec_buf_state);
            } else {
                stride = process_relay_ringbuffer_cmd<cmddat_wrap_enable>(cmd_ptr, downstream_data_ptr, l1_cache);
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_LINEAR_PACKED:
            // DPRINT << "relay linear packed" << ENDL();
            if (exec_buf) {
                stride =
                    process_exec_buf_relay_linear_packed_cmd(cmd_ptr, downstream_data_ptr, l1_cache, exec_buf_state);
            } else {
                stride = process_relay_linear_packed_cmd<cmddat_wrap_enable>(cmd_ptr, downstream_data_ptr, l1_cache);
            }
            break;

        default:
            //  DPRINT << "prefetch invalid command:" << (uint32_t)cmd->base.cmd_id << " " << cmd_ptr << " " <<
            //                           cmddat_q_base << ENDL();
            //  DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
            //  DPRINT << HEX() << *((uint32_t*)cmd_ptr+1) << ENDL();
            //  DPRINT << HEX() << *((uint32_t*)cmd_ptr+2) << ENDL();
            //  DPRINT << HEX() << *((uint32_t*)cmd_ptr+3) << ENDL();
            //  DPRINT << HEX() << *((uint32_t*)cmd_ptr+4) << ENDL();
            WAYPOINT("!CMD");
            ASSERT(0);
    }

    return done;
}

// We require that all data for a single fetch is available before processing commands. We can't use a normal
// CBReaderWithReleasePolicy because that always releases pages when advancing between blocks,
// which would cause problems if the data spans multiple blocks.
CBReaderWithManualRelease<
    my_upstream_cb_sem_id,
    cmddat_q_log_page_size,
    cmddat_q_blocks,
    cmddat_q_pages_per_block,
    cmddat_q_base,
    cmddat_q_end>
    h_cmddat_q_reader;

// Used in prefetch_d downstream of a CQ_PREFETCH_CMD_RELAY_LINEAR_H command.
inline void relay_raw_data_to_downstream(uint32_t& data_ptr, uint64_t wlength, uint32_t& local_downstream_data_ptr) {
    // In initial return, we return the header bytes as well
    uint32_t initial_data_to_return = sizeof(CQPrefetchHToPrefetchDHeader);
    data_ptr += sizeof(CQPrefetchHToPrefetchDHeader);
    wlength -= sizeof(CQPrefetchHToPrefetchDHeader);
    // Stream data to downstream as it arrives. Acquire upstream pages incrementally.
    while (wlength > 0) {
        // Ensure at least one upstream page is available
        uint32_t available_data = h_cmddat_q_reader.wait_for_available_data(data_ptr);

        uint32_t can_read_now = available_data;
        if (can_read_now > wlength) {
            can_read_now = wlength;
        }

        // Decide whether this is the final chunk
        bool is_final_chunk = (can_read_now == wlength);

        uint32_t npages;
        if (is_final_chunk) {
            npages = write_pages_to_dispatcher<1, true>(local_downstream_data_ptr, data_ptr, can_read_now);
        } else {
            npages = write_pages_to_dispatcher<0, false>(local_downstream_data_ptr, data_ptr, can_read_now);
        }

        // Release pages consumed by this chunk
        if (npages != 0) {
            DispatchRelayInlineState::cb_writer.release_pages(npages, local_downstream_data_ptr, true);
        }

        // Advance pointers and wlength
        h_cmddat_q_reader.consumed_data(data_ptr, can_read_now);
        wlength -= can_read_now;

        // Release upstream pages so prefetch_h can make more available. Ensure to flush the writes just made to prevent
        // data race.
        noc_async_writes_flushed();
        // wait_for_available_data always returns up to a page boundary, so the rounding only matters on the final chunk
        // and lets us return the final bytes in the page early.
        uint32_t pages_to_free =
            (can_read_now + initial_data_to_return + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
        initial_data_to_return = 0;
        uint32_t watcher_data_ptr = data_ptr;
#if ASSERT_ENABLED
        if (wlength == 0) {
            // This normally happens after the end of the loop, but do it early here so we don't hit assertions. Data
            // until the end of the page isn't used.
            watcher_data_ptr = round_up_pow2(watcher_data_ptr, cmddat_q_page_size);
        }
#endif
        // data_ptr may not be page-aligned mid-stream, so allow it to be up to one page ahead
        relay_client.release_pages<
            my_noc_index,
            upstream_noc_xy,
            upstream_cb_sem_id,
            cmddat_q_base,
            cmddat_q_end,
            cmddat_q_page_size>(pages_to_free, watcher_data_ptr);
    }
    local_downstream_data_ptr =
        round_up_pow2(local_downstream_data_ptr, DispatchRelayInlineState::downstream_page_size);
    // Release an extra page to account for the dispatch command (via CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH) that would
    // have headed the first relayed segment
    DispatchRelayInlineState::cb_writer.release_pages(1, local_downstream_data_ptr);
    // Round upstream pointer to next cmddat page boundary for next command
    data_ptr = round_up_pow2(data_ptr, cmddat_q_page_size);
}

// Gets cmds from upstream prefetch_h
// Note the prefetch_h uses the HostQ and grabs whole commands
// Shared command processor assumes whole commands are present, really
// just matters for the inline command which could be re-implemented
// This grabs whole (possibly sets of if multiple in a page) commands.
// In the case raw_copy is set in the header, that data will be copied to the downstream, and this function will loop
// until commands are received.
inline uint32_t relay_cb_get_cmds(uint32_t& data_ptr, uint32_t& downstream_data_ptr) {
    while (true) {
        // DPRINT << "get_commands: data_ptr:0x" << HEX() << data_ptr << ", fence:0x" << fence << ",
        // downstream_data_ptr:0x" << downstream_data_ptr << ENDL();
        h_cmddat_q_reader.wait_for_available_data(data_ptr);

        volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader* cmd_ptr =
            (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader*)data_ptr;

        if (cmd_ptr->header.raw_copy) {
            uint64_t wlength = cmd_ptr->header.length;
            relay_raw_data_to_downstream(data_ptr, wlength, downstream_data_ptr);
        } else {
            uint32_t length = cmd_ptr->header.length;
            // Ensure the entire command payload is present before returning
            uint32_t pages_ready = h_cmddat_q_reader.available_bytes(data_ptr) >> cmddat_q_log_page_size;
            uint32_t pages_needed = (length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
            int32_t pages_pending = pages_needed - pages_ready;
            int32_t npages = 0;

            uint32_t dummy_data_ptr = data_ptr;
            while (npages < pages_pending) {
                npages += h_cmddat_q_reader.get_cb_page(dummy_data_ptr);
                IDLE_ERISC_RETURN(length - sizeof(CQPrefetchHToPrefetchDHeader));
            }

            data_ptr += sizeof(CQPrefetchHToPrefetchDHeader);
            return length - sizeof(CQPrefetchHToPrefetchDHeader);
        }
    }
}

void kernel_main_d() {
    PrefetchExecBufState exec_buf_state;

    h_cmddat_q_reader.init();
    uint32_t cmd_ptr = cmddat_q_base;

    bool done = false;
    uint32_t heartbeat = 0;
    uint32_t l1_cache[l1_cache_elements_rounded];

    // Cmdbuf allocation is not defined yet for fabric so we can't use stateful APIs on Dispatch D
#if defined(FABRIC_RELAY)
    relay_client.init<
        my_noc_index,
        fabric_mux_x,
        fabric_mux_y,
        worker_credits_stream_id,
        fabric_mux_channel_base_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_connection_info_address,
        fabric_mux_buffer_index_address,
        fabric_worker_flow_control_sem,
        fabric_worker_teardown_sem,
        fabric_worker_buffer_index_sem,
        fabric_mux_status_address,
        my_fabric_sync_status_addr,
        to_mesh_id,
        ew_dim,
        fabric_header_rb_base,
        num_hops,
        NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0), my_dev_id, to_dev_id, router_direction);
#else
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), 0, my_noc_index);
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchSRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(dispatch_s_noc_xy, downstream_data_ptr_s), 0, my_noc_index);
#endif

    // Initialize cmd_ptr tracking for release_pages synchronization assertions
    relay_client.init_cmd_ptr_tracking<cmddat_q_base>();

    while (!done) {
        // cmds come in packed batches based on HostQ reads in prefetch_h
        // once a packed batch ends, we need to jump to the next page
        uint32_t length = relay_cb_get_cmds(cmd_ptr, downstream_data_ptr);

        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        uint32_t amt_processed = 0;
        while (length > amt_processed) {
            uint32_t stride;
            done = process_cmd<true, false>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);
            amt_processed += stride;

            h_cmddat_q_reader.consumed_data(cmd_ptr, stride);
        }

        // TODO: evaluate less costly free pattern (blocks?)
        uint32_t total_length = length + sizeof(CQPrefetchHToPrefetchDHeader);
        uint32_t pages_to_free = (total_length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
        // Ensure all writes that consumed this payload have completed before releasing upstream pages
        noc_async_writes_flushed();

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, cmddat_q_page_size);
        relay_client.release_pages<
            my_noc_index,
            upstream_noc_xy,
            upstream_cb_sem_id,
            cmddat_q_base,
            cmddat_q_end,
            cmddat_q_page_size>(pages_to_free, cmd_ptr);
    }

    // Set upstream semaphore MSB to signal completion and path teardown
    // in case prefetch_d is connected to a depacketizing stage.
    relay_client.teardown<my_noc_index, upstream_noc_xy, upstream_cb_sem_id>();
}

void kernel_main_hd() {
    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;
    bool done = false;
    uint32_t heartbeat = 0;
    uint32_t l1_cache[l1_cache_elements_rounded];
    PrefetchExecBufState exec_buf_state;

    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), 0);
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchSRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(dispatch_s_noc_xy, downstream_data_ptr_s), 0);

    while (!done) {
        DeviceZoneScopedN("CQ-PREFETCH");
        constexpr uint32_t preamble_size = 0;
        fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);

        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;

        uint32_t stride;
        done = process_cmd<false, false>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);
        cmd_ptr += stride;
    }
}

void kernel_main() {
    set_l1_data_cache<true>();
#if defined(FABRIC_RELAY)
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": start (fabric relay. 2d = " << (uint32_t)is_2d_fabric
           << ")" << ENDL();
#else
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": start" << ENDL();
#endif

    // Get runtime args
    my_dev_id = get_arg_val<uint32_t>(OFFSETOF_MY_DEV_ID);
    to_dev_id = get_arg_val<uint32_t>(OFFSETOF_TO_DEV_ID);
    router_direction = get_arg_val<uint32_t>(OFFSETOF_ROUTER_DIRECTION);

    if (is_h_variant and is_d_variant) {
        kernel_main_hd();
    } else if (is_d_variant) {
        kernel_main_d();
    } else {
        ASSERT(0);
    }
    IDLE_ERISC_RETURN();

    // The reader stub never writes pages to the downstream CB (all relay functions are
    // stubbed), so the CBWriter's page accounting is always at its initial state and
    // wait_all_pages would hang waiting for credits that never arrive.  Skip it here;
    // the writer (NCRISC) sends the terminate directly to the dispatcher instead.

    noc_async_full_barrier();

    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": out" << ENDL();
    set_l1_data_cache<false>();
}
