// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch kernel
//  - 3 flavors: _hd (host and dram), _h (host only), _d (DRAM only)
//  - fetches commands from host (if applicable), executes
//  - uses HostQ for host handshaking, ComDatQ for commands (from host),
//    double buffered ScratchBuf for out of band data (e.g., from DRAM)
//  - syncs w/ dispatcher via 2 semaphores, page_ready, page_done
//
// Write cmd buf allocation:
//  - BRISC_WR_CMD_BUF: writes to downstream_noc_xy
//  - BRISC_WR_REG_CMD_BUF: small writes to dispatch_s_noc_xy. not much traffic on this path.
//
//  Using the normal NoC APIs for writes and/or inline_dw_writes are not allowed on this kernel.
//

#include "dataflow_api.h"
#include "dataflow_api_addrgen.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_relay.hpp"
#include "debug/dprint.h"
#include "noc/noc_parameters.h"  // PCIE_ALIGNMENT

constexpr uint32_t CQ_PREFETCH_CMD_BARE_MIN_SIZE = PCIE_ALIGNMENT;  // for NOC PCIe alignemnt
struct CQPrefetchHToPrefetchDHeader_s {
    uint32_t length;
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

constexpr uint32_t my_dev_id = MY_DEV_ID;
constexpr uint32_t ew_dim = EW_DIM;
constexpr uint32_t to_mesh_id = TO_MESH_ID;
constexpr uint32_t to_dev_id = TO_DEV_ID;
constexpr uint32_t router_direction = ROUTER_DIRECTION;

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
constexpr uint32_t dispatch_s_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_SUBORDINATE_NOC_X, DOWNSTREAM_SUBORDINATE_NOC_Y));
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

uint32_t my_downstream_cb_sem_additional_count = 0;
uint32_t my_dispatch_s_cb_sem_additional_count = 0;

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
    static constexpr uint32_t& downstream_cb_additional_count = my_downstream_cb_sem_additional_count;
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
    static constexpr uint32_t& downstream_cb_additional_count = my_dispatch_s_cb_sem_additional_count;
};

struct PrefetchExecBufState {
    uint32_t page_id;
    uint32_t base_addr;
    uint32_t log_page_size;
    uint32_t pages;
    uint32_t length;
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
        pending_read_size = read_from_pcie<preamble_size>(
            prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
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
                    pending_read_size = read_from_pcie<0>(
                        prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
                    if (pending_read_size != 0) {
                        // if pending_read_size == 0 read_from_pcie early exited, due to a wrap, i.e. the exec_buf cmd
                        // is at a wrapped location, and a read to it could not be issued, since there are existing
                        // commands in the cmddat_q. Only move the stall_state to stalled if the read to the cmd that
                        // initiated the stall was issued
                        barrier_and_stall(
                            pending_read_size, fence, cmd_ptr);  // STALL_NEXT -> STALLED
                    }
                } else {
                    pending_read_size = read_from_pcie<preamble_size>(
                        prefetch_q_rd_ptr, fence, pcie_read_ptr, cmd_ptr, fetch_size);
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
    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    uint32_t npages =
        (length + RelayInlineState::downstream_page_size - 1) >> RelayInlineState::downstream_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, RelayInlineState::my_downstream_cb_sem>(
        npages, RelayInlineState::downstream_cb_additional_count);

    uint32_t remaining = cmddat_q_end - data_ptr;
    if (cmddat_wrap_enable && length > remaining) {
        // wrap cmddat
        write_downstream<RelayInlineState::downstream_cb_base_addr, RelayInlineState::downstream_write_cmd_buf>(
            data_ptr,
            local_downstream_data_ptr,
            remaining,
            RelayInlineState::downstream_cb_end_addr,
            RelayInlineState::downstream_noc_encoding);
        length -= remaining;
        data_ptr = cmddat_q_base;
    }

    write_downstream<RelayInlineState::downstream_cb_base_addr, RelayInlineState::downstream_write_cmd_buf>(
        data_ptr,
        local_downstream_data_ptr,
        length,
        RelayInlineState::downstream_cb_end_addr,
        RelayInlineState::downstream_noc_encoding);

    local_downstream_data_ptr = round_up_pow2(local_downstream_data_ptr, RelayInlineState::downstream_page_size);
    noc_async_writes_flushed();
    cb_release_pages<my_noc_index, RelayInlineState::downstream_noc_encoding, RelayInlineState::downstream_cb_sem>(
        npages);
    return cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
template <bool cmddat_wrap_enable>
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr, uint32_t& dispatch_data_ptr) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(1, DispatchRelayInlineState::downstream_cb_additional_count);
    if (dispatch_data_ptr == downstream_cb_end) {
        dispatch_data_ptr = downstream_cb_base;
    }
    uint32_t remaining = cmddat_q_end - data_ptr;
    if (cmddat_wrap_enable && length > remaining) {
        // wrap cmddat
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), remaining);
        dispatch_data_ptr += remaining;
        length -= remaining;
        data_ptr = cmddat_q_base;
    }
    noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += length;

    return cmd->relay_inline.stride;
}

// The hard problem here is: when an xfer lands exactly at a page boundary, who is responsible for getting the next
// page? For inner loop, call N grabs page N+1.  No client should ever hit this as inline_noflush puts 16 bytes at the
// top of the first page At the end, do not grab page N+1
template <int32_t round, bool test_for_nonzero>
static uint32_t write_pages_to_dispatcher(
    uint32_t& downstream_data_ptr, uint32_t& scratch_write_addr, uint32_t& amt_to_write) {
    uint32_t page_residual_space = downstream_cb_page_size - (downstream_data_ptr & (downstream_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + downstream_cb_page_size - round) / downstream_cb_page_size;

    // Grabbing all pages at once is ok if scratch_size < 3 * downstream_cb_block_size
    // test_for_nonzero is an optimization: inner loops moving lots of pages don't bother
    if (!test_for_nonzero || npages != 0) {
        cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages, my_downstream_cb_sem_additional_count);
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
#if ENABLE_PREFETCH_DPRINTS
    DPRINT << "relay_paged_cmd_large: " << page_size << " " << pages << " " << length_adjust << ENDL();
#endif

    InterleavedAddrGen<is_dram> addr_gen{.bank_base_address = base_addr, .page_size = page_size};

    // First step - read into DB0
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
    uint32_t write_length = pages * page_size - length_adjust;
    uint32_t read_length;
    uint32_t amt_read;
    if (scratch_db_half_size >= write_length) {
        amt_read = write_length;
        read_length = 0;
    } else {
        amt_read = scratch_db_half_size;
        read_length = write_length - amt_read;
    }
    noc_async_read(noc_addr, scratch_read_addr, amt_read);
    uint32_t page_length = page_size - amt_read;
    uint32_t page_offset = amt_read;

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;

    noc_async_read_barrier();
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_read;
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id, page_offset);
        if (page_length <= scratch_db_half_size) {
            noc_async_read(noc_addr, scratch_read_addr, page_length);
            page_id++;
            page_offset = 0;
            amt_read = page_length;
            page_length = page_size;

            if (amt_read < scratch_db_half_size && read_length > amt_read) {
                noc_addr = addr_gen.get_noc_addr(page_id, 0);
                uint32_t amt_to_read = scratch_db_half_size - amt_read;
                noc_async_read(noc_addr, scratch_read_addr + amt_read, amt_to_read);
                page_length -= amt_to_read;
                amt_read = scratch_db_half_size;
                page_offset = amt_to_read;
            }
        } else {
            noc_async_read(noc_addr, scratch_read_addr, scratch_db_half_size);
            page_length -= scratch_db_half_size;
            page_offset += scratch_db_half_size;
            amt_read = scratch_db_half_size;
        }

        // Third step - write from DB
        if (write_length < amt_to_write) {
            amt_to_write = write_length;
            read_length = 0;
        } else {
            read_length -= amt_read;
        }

        write_length -= amt_to_write;
        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, scratch_write_addr, amt_to_write);
        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages);

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    if (write_length > 0) {
        scratch_write_addr = scratch_db_top[db_toggle];
        uint32_t amt_to_write = write_length;
        uint32_t npages = write_pages_to_dispatcher<1, true>(downstream_data_ptr, scratch_write_addr, amt_to_write);

        // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages + 1);
    } else {
        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(1);
    }

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

// This fn prefetches data from DRAM memory and writes data to the dispatch core.
// Reading from DRAM has the following characteristics:
//  - latency is moderately high ~400 cycles on WH
//  - DRAM bw is ~maximized when page size reaches 2K
//  - for kernel dispatch, it is expected that page sizes will often be <2K
//  - for buffer writing, page sizes will vary
//  - writing to dispatcher works best with 4K pages (2K pages cover overhead, 4K gives perf cushion)
//  - writing a 4K page takes ~32*4=128 cycles
//  - writing 4 4K pages is 512 cycles, close to parity w/ the latency of DRAM
//  - to hide the latency (~12% overhead), assume we need to read ~32 pages=128K, double buffered
//  - in other words, we'll never achieve high efficiency and always be (somewhat) latency bound
// Algorithm does:
//  - read a batch from DRAM
//  - loop: read a batch from DRAM while sending to dispatcher
//  - send a batch to dispatcher
// The size of the first read should be based on latency.  With small page sizes
// bandwidth will be low and we'll be DRAM bound (send to dispatcher is ~free).
// With larger pages we'll get closer to a bandwidth match
// The dispatch buffer is a ring buffer.
template <bool is_dram>
uint32_t process_relay_paged_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t page_id) {
    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t base_addr = cmd->relay_paged.base_addr;
    uint32_t page_size = cmd->relay_paged.page_size;
    uint32_t pages = cmd->relay_paged.pages;
    uint16_t length_adjust = cmd->relay_paged.is_dram_and_length_adjust & CQ_PREFETCH_RELAY_PAGED_LENGTH_ADJUST_MASK;

    if (page_size > scratch_db_half_size) {
        return process_relay_paged_cmd_large<is_dram>(
            cmd_ptr, downstream_data_ptr, page_id, base_addr, page_size, pages, length_adjust);
    }

    InterleavedAddrGen<is_dram> addr_gen{.bank_base_address = base_addr, .page_size = page_size};

    // First step - read into DB0
    uint32_t read_length = pages * page_size;
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint32_t amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
    uint32_t amt_read = 0;
    while (amt_to_read >= page_size) {
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
        noc_async_read(noc_addr, scratch_read_addr, page_size);
        scratch_read_addr += page_size;
        page_id++;
        amt_to_read -= page_size;
        amt_read += page_size;
    }
    noc_async_read_barrier();

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    read_length -= amt_read;
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_read;
        amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
        amt_read = 0;
        while (amt_to_read >= page_size) {
            uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
            noc_async_read(noc_addr, scratch_read_addr, page_size);
            scratch_read_addr += page_size;
            page_id++;
            amt_to_read -= page_size;
            amt_read += page_size;
        }

        // Third step - write from DB
        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, scratch_write_addr, amt_to_write);
        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages);

        read_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    // Note that we may write less than full pages despite reading full pages based on length_adjust
    // Expectation is that the gain from reading less is small to 0, revisit as needed
    ASSERT(length_adjust < page_size);
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_read - length_adjust;
    uint32_t npages = write_pages_to_dispatcher<1, true>(downstream_data_ptr, scratch_write_addr, amt_to_write);

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
    cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages + 1);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

// Similar to relay_paged, this iterates and aggregates reads from multiple
// embedded relay_paged cmds
void process_relay_paged_packed_sub_cmds(uint32_t total_length, uint32_t* l1_cache) {
    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    // First step - read multiple sub_cmds worth into DB0
    CQPrefetchRelayPagedPackedSubCmd tt_l1_ptr* sub_cmd = (CQPrefetchRelayPagedPackedSubCmd tt_l1_ptr*)(l1_cache);
    uint32_t read_length = sub_cmd->length;
    ASSERT(read_length <= scratch_db_half_size);
    uint32_t amt_to_read = (scratch_db_half_size > total_length) ? total_length : scratch_db_half_size;
    uint32_t amt_read = 0;
    uint32_t scratch_read_addr = scratch_db_top[0];

    while (read_length <= amt_to_read) {
        uint32_t page_id = sub_cmd->start_page;
        uint32_t log_page_size = sub_cmd->log_page_size;
        uint32_t base_addr = sub_cmd->base_addr;
        sub_cmd++;

        uint32_t page_size = 1 << log_page_size;
        InterleavedPow2AddrGen<true> addr_gen{.bank_base_address = base_addr, .log_base_2_of_page_size = log_page_size};

        uint32_t amt_to_read2 =
            (scratch_db_half_size - amt_read > read_length) ? read_length : scratch_db_half_size - amt_read;
        uint32_t amt_read2 = 0;
        while (amt_read2 < amt_to_read2) {
            uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
            uint32_t read_size = (amt_to_read2 - amt_read2 >= page_size) ? page_size : amt_to_read2 - amt_read2;
            noc_async_read(noc_addr, scratch_read_addr, read_size);
            scratch_read_addr += read_size;
            page_id++;
            amt_read2 += read_size;
        }

        amt_read += amt_read2;
        amt_to_read -= amt_read2;

        // note: below can walk off the end of the sub_cmds
        // this is ok as we store a sentinel non-zero value
        read_length = sub_cmd->length;
        ASSERT(read_length <= scratch_db_half_size || total_length == amt_read);
    }
    noc_async_read_barrier();

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    total_length -= amt_read;
    while (total_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_read;
        amt_read = 0;
        amt_to_read = (scratch_db_half_size > total_length) ? total_length : scratch_db_half_size;
        while (read_length <= amt_to_read) {
            uint32_t page_id = sub_cmd->start_page;
            uint32_t log_page_size = sub_cmd->log_page_size;
            uint32_t base_addr = sub_cmd->base_addr;
            sub_cmd++;

            uint32_t page_size = 1 << log_page_size;
            InterleavedPow2AddrGen<true> addr_gen{
                .bank_base_address = base_addr, .log_base_2_of_page_size = log_page_size};

            uint32_t amt_to_read2 =
                (scratch_db_half_size - amt_read > read_length) ? read_length : scratch_db_half_size - amt_read;
            uint32_t amt_read2 = 0;
            while (amt_read2 < amt_to_read2) {
                uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
                uint32_t read_size = (amt_to_read2 - amt_read2 >= page_size) ? page_size : amt_to_read2 - amt_read2;
                noc_async_read(noc_addr, scratch_read_addr, read_size);
                scratch_read_addr += read_size;
                page_id++;
                amt_read2 += read_size;
            }

            amt_read += amt_read2;
            amt_to_read -= amt_read2;

            // note: below can walk off the end of the sub_cmds
            // this is ok as we store a sentinel non-zero value
            read_length = sub_cmd->length;
            ASSERT(read_length <= scratch_db_half_size || total_length == amt_read);
        }

        // Third step - write from DB
        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, scratch_write_addr, amt_to_write);
        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages);

        total_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_read;
    uint32_t npages = write_pages_to_dispatcher<1, true>(downstream_data_ptr, scratch_write_addr, amt_to_write);

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
    cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages + 1);
}

template <bool cmddat_wrap_enable>
uint32_t process_relay_paged_packed_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t total_length = cmd->relay_paged_packed.total_length;
    uint32_t sub_cmds_length = cmd->relay_paged_packed.count * sizeof(CQPrefetchRelayPagedPackedSubCmd);
    uint32_t stride = cmd->relay_paged_packed.stride;
    ASSERT(total_length > 0);
    // DPRINT << "paged_packed: " << total_length << " " << cmd->relay_paged_packed.stride << ENDL();

    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);
    uint32_t remaining = cmddat_q_end - data_ptr;
    uint32_t* l1_cache_pos = l1_cache;
    if (cmddat_wrap_enable && sub_cmds_length > remaining) {
        // wrap cmddat
        uint32_t amt = remaining / sizeof(uint32_t);
        careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
            (volatile uint32_t tt_l1_ptr*)(data_ptr), amt, l1_cache_pos);
        sub_cmds_length -= remaining;
        data_ptr = cmddat_q_base;
        l1_cache_pos += amt;
    }

    uint32_t amt = sub_cmds_length / sizeof(uint32_t);
    // Check that the final write does not overflow the L1 cache and corrupt the stack
    // End address of final write is: curr_offset_into_cache + write_size_rounded_up_to_copy_chunk
    ASSERT(
        (uint32_t)(l1_cache_pos +
                   ((amt + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
                       l1_to_local_cache_copy_chunk -
                   l1_cache) < l1_cache_elements_rounded);
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(data_ptr), amt, l1_cache_pos);
    // Store a sentinal non 0 value at the end to save a test/branch in read path
    ((CQPrefetchRelayPagedPackedSubCmd*)&l1_cache_pos[amt])->length = 1;

    process_relay_paged_packed_sub_cmds(total_length, l1_cache);
    return stride;
}

uint32_t process_relay_linear_cmd(uint32_t cmd_ptr, uint32_t& downstream_data_ptr) {
    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t noc_xy_addr = cmd->relay_linear.noc_xy_addr;
    uint32_t read_addr = cmd->relay_linear.addr;
    uint32_t length = cmd->relay_linear.length;
    uint32_t read_length = length;
    // DPRINT << "relay_linear: " << cmd_ptr << " " << length << " " << read_addr << " " << noc_xy_addr << ENDL();

    // First step - read into DB0
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint32_t amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
    uint64_t noc_addr = get_noc_addr_helper(noc_xy_addr, read_addr);
    noc_async_read(noc_addr, scratch_read_addr, amt_to_read);
    read_addr += amt_to_read;
    noc_async_read_barrier();

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    read_length -= amt_to_read;
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_to_read;
        amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
        noc_addr = get_noc_addr_helper(noc_xy_addr, read_addr);
        noc_async_read(noc_addr, scratch_read_addr, amt_to_read);
        read_addr += amt_to_read;

        // Third step - write from DB
        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, scratch_write_addr, amt_to_write);

        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages);

        read_length -= amt_to_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_to_read;
    uint32_t npages = write_pages_to_dispatcher<1, true>(downstream_data_ptr, scratch_write_addr, amt_to_write);

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages + 1);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_stall(uint32_t cmd_ptr) {
    static uint32_t count = 0;

    count++;

    WAYPOINT("PSW");
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<fd_core_type>(my_downstream_sync_sem_id));
    uint32_t heartbeat = 0;
    do {
        invalidate_l1_cache();
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    } while (*sem_addr != count);
    WAYPOINT("PSD");

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void paged_read_into_cmddat_q(uint32_t read_ptr, PrefetchExecBufState& exec_buf_state) {
    uint32_t page_id = exec_buf_state.page_id;
    uint32_t base_addr = exec_buf_state.base_addr;
    uint32_t log_page_size = exec_buf_state.log_page_size;
    uint32_t page_size = 1 << log_page_size;
    uint32_t pages = exec_buf_state.pages;

    // TODO: tune how much is read
    uint32_t max_trace_buffer_pages_in_cmd_dat_q = cmddat_q_size >> log_page_size;
    uint32_t pages_at_once =
        (max_trace_buffer_pages_in_cmd_dat_q > pages) ? pages : max_trace_buffer_pages_in_cmd_dat_q;
    uint32_t read_length = pages_at_once << log_page_size;

    InterleavedAddrGen<true> addr_gen{.bank_base_address = base_addr, .page_size = page_size};

    while (pages_at_once != 0) {
        invalidate_l1_cache();
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
        noc_async_read(noc_addr, read_ptr, page_size);
        read_ptr += page_size;
        page_id++;
        pages_at_once--;
    }

    exec_buf_state.page_id = page_id;
    exec_buf_state.pages -= pages_at_once;
    exec_buf_state.length += read_length;
}

// processes the relay_inline cmd from an exec_buf
// ie, reads the data from dram and relays it on
// Separate implementation that fetches more data from exec buf when cmd has been split
template <typename RelayInlineState>
FORCE_INLINE static uint32_t process_exec_buf_relay_inline_cmd(
    uint32_t& cmd_ptr, uint32_t& local_downstream_data_ptr, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint8_t dispatcher_type = cmd->relay_inline.dispatcher_type;
    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    // DPRINT << "relay_inline_exec_buf_cmd:" << length << ENDL();
    uint32_t npages =
        (length + RelayInlineState::downstream_page_size - 1) >> RelayInlineState::downstream_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, RelayInlineState::my_downstream_cb_sem>(
        npages, RelayInlineState::downstream_cb_additional_count);
    uint32_t stride = cmd->relay_inline.stride;
    uint32_t remaining_stride = exec_buf_state.length;
    uint32_t remaining = exec_buf_state.length - sizeof(CQPrefetchCmd);
    while (length > remaining) {
        // wrap cmddat
        write_downstream<RelayInlineState::downstream_cb_base_addr, RelayInlineState::downstream_write_cmd_buf>(
            data_ptr,
            local_downstream_data_ptr,
            remaining,
            RelayInlineState::downstream_cb_end_addr,
            RelayInlineState::downstream_noc_encoding);
        length -= remaining;
        stride -= remaining_stride;
        exec_buf_state.length = 0;
        data_ptr = cmddat_q_base;
        cmd_ptr = cmddat_q_base;

        // fetch more
        noc_async_writes_flushed(RelayInlineState::downstream_noc_index);
        paged_read_into_cmddat_q(cmd_ptr, exec_buf_state);
        remaining = exec_buf_state.length;
        remaining_stride = exec_buf_state.length;
        noc_async_read_barrier();
    }
    write_downstream<RelayInlineState::downstream_cb_base_addr, RelayInlineState::downstream_write_cmd_buf>(
        data_ptr,
        local_downstream_data_ptr,
        length,
        RelayInlineState::downstream_cb_end_addr,
        RelayInlineState::downstream_noc_encoding);
    local_downstream_data_ptr = round_up_pow2(local_downstream_data_ptr, RelayInlineState::downstream_page_size);
    noc_async_writes_flushed(RelayInlineState::downstream_noc_index);
    cb_release_pages<my_noc_index, RelayInlineState::downstream_noc_encoding, RelayInlineState::downstream_cb_sem>(
        npages);

    return stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_inline_noflush_cmd(
    uint32_t& cmd_ptr, uint32_t& dispatch_data_ptr, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    uint32_t stride = cmd->relay_inline.stride;

    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(1, my_downstream_cb_sem_additional_count);
    if (dispatch_data_ptr == downstream_cb_end) {
        dispatch_data_ptr = downstream_cb_base;
    }
    uint32_t remaining_stride = exec_buf_state.length;
    uint32_t remaining = exec_buf_state.length - sizeof(CQPrefetchCmd);
    while (length > remaining) {
        // wrap cmddat
#if defined(FABRIC_RELAY)
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), remaining);
#else
        cq_noc_async_write_with_state_any_len<true, true>(
            data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), remaining);
#endif
        dispatch_data_ptr += remaining;
        length -= remaining;
        stride -= remaining_stride;
        exec_buf_state.length = 0;
        data_ptr = cmddat_q_base;
        cmd_ptr = cmddat_q_base;

        // fetch more
        noc_async_writes_flushed();
        paged_read_into_cmddat_q(cmd_ptr, exec_buf_state);
        noc_async_read_barrier();
        remaining = exec_buf_state.length;
        remaining_stride = exec_buf_state.length;
    }

#if defined(FABRIC_RELAY)
    noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), length);
#else
    cq_noc_async_write_with_state_any_len<true, true>(
        data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), length);
#endif
    dispatch_data_ptr += length;

    return stride;
}

void* copy_into_l1_cache(
    uint32_t& cmd_ptr,
    uint32_t sub_cmds_length,
    uint32_t* l1_cache,
    PrefetchExecBufState& exec_buf_state,
    uint32_t& stride) {
    uint32_t remaining_stride = exec_buf_state.length;
    uint32_t remaining = (exec_buf_state.length - sizeof(CQPrefetchCmd));
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)(cmd_ptr + sizeof(CQPrefetchCmd));
    uint32_t* l1_cache_pos = l1_cache;
    while (sub_cmds_length > remaining) {
        uint32_t amt = remaining / sizeof(uint32_t);
        careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
            l1_ptr, amt, l1_cache_pos);

        l1_cache_pos += amt;
        sub_cmds_length -= remaining;
        stride -= remaining_stride;
        exec_buf_state.length = 0;
        cmd_ptr = cmddat_q_base;
        l1_ptr = (volatile uint32_t tt_l1_ptr*)(cmd_ptr);
        paged_read_into_cmddat_q(cmd_ptr, exec_buf_state);
        noc_async_read_barrier();
        remaining = exec_buf_state.length;
        remaining_stride = exec_buf_state.length;
    }
    uint32_t amt = sub_cmds_length / sizeof(uint32_t);
    // Check that the final write does not overflow the L1 cache and corrupt the stack.
    // End address of final write is: curr_offset_into_cache + write_size_rounded_up_to_copy_chunk
    ASSERT(
        (uint32_t)(l1_cache_pos +
                   ((amt + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
                       l1_to_local_cache_copy_chunk -
                   l1_cache) < l1_cache_elements_rounded);
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        l1_ptr, amt, l1_cache_pos);

    // Return a pointer to right after the last copy
    return &l1_cache_pos[amt];
}

// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_paged_packed_cmd(
    uint32_t& cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t total_length = cmd->relay_paged_packed.total_length;
    uint32_t sub_cmds_length = cmd->relay_paged_packed.count * sizeof(CQPrefetchRelayPagedPackedSubCmd);
    uint32_t stride = cmd->relay_paged_packed.stride;
    ASSERT(total_length > 0);
    // DPRINT << "paged_packed: " << total_length << " " << cmd->relay_paged_packed.stride << ENDL();

    void* end = copy_into_l1_cache(cmd_ptr, sub_cmds_length, l1_cache, exec_buf_state, stride);

    // Store a sentinal non 0 value at the end to save a test/branch in read path
    ((CQPrefetchRelayPagedPackedSubCmd*)end)->length = 1;

    process_relay_paged_packed_sub_cmds(total_length, l1_cache);
    return stride;
}

uint32_t process_exec_buf_cmd(
    uint32_t cmd_ptr_outer, uint32_t& downstream_data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    // dispatch on eth cores is memory constrained, so exec_buf re-uses the cmddat_q
    // prefetch_h stalls upon issuing an exec_buf to prevent conflicting use of the cmddat_q,
    // the exec_buf contains the release commands
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr_outer;

    exec_buf_state.page_id = 0;
    exec_buf_state.base_addr = cmd->exec_buf.base_addr;
    exec_buf_state.log_page_size = cmd->exec_buf.log_page_size;
    exec_buf_state.pages = cmd->exec_buf.pages;
    exec_buf_state.length = 0;

    bool done = false;
    while (!done) {
        uint32_t cmd_ptr = cmddat_q_base;

        paged_read_into_cmddat_q(cmd_ptr, exec_buf_state);
        noc_async_read_barrier();

        while (exec_buf_state.length > 0) {
            uint32_t stride;
            done = process_cmd<false, true>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);

            if (done) {
                break;
            }

            exec_buf_state.length -= stride;
            cmd_ptr += stride;
        }
    }

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_paged_to_ringbuffer_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr) {
    // This ensures that a previous cmd using the ringbuffer have completed.
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t start_page = cmd->paged_to_ringbuffer.start_page;
    uint32_t base_addr = cmd->paged_to_ringbuffer.base_addr;
    uint8_t log2_page_size = cmd->paged_to_ringbuffer.log2_page_size;
    uint32_t page_size = 1 << log2_page_size;
    uint32_t length = cmd->paged_to_ringbuffer.length;
    uint8_t flags = cmd->paged_to_ringbuffer.flags;
    uint32_t wp_update_offset = cmd->paged_to_ringbuffer.wp_offset_update;

    ASSERT(length <= wp_update_offset);

    if (flags & CQ_PREFETCH_PAGED_TO_RING_BUFFER_FLAG_RESET_TO_START) {
        ringbuffer_wp = scratch_db_base;
    }

    ASSERT(length % DRAM_ALIGNMENT == 0);
    ASSERT(wp_update_offset + ringbuffer_wp <= ringbuffer_end);

    ringbuffer_offset = ringbuffer_wp - scratch_db_base;

    const bool is_dram = true;
    InterleavedPow2AddrGen<is_dram> addr_gen{.bank_base_address = base_addr, .log_base_2_of_page_size = log2_page_size};

    uint32_t scratch_read_addr = ringbuffer_wp;
    uint32_t page_id = start_page;
    while (length >= page_size) {
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
        noc_async_read(noc_addr, scratch_read_addr, page_size);
        scratch_read_addr += page_size;
        page_id++;
        length -= page_size;
    }
    if (length > 0) {
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
        noc_async_read(noc_addr, scratch_read_addr, length);
        scratch_read_addr += length;
    }

    ringbuffer_wp += wp_update_offset;

    // The consumer will perform a read barrier.

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_set_ringbuffer_offset(uint32_t cmd_ptr) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t offset = cmd->set_ringbuffer_offset.offset;

    if (cmd->set_ringbuffer_offset.update_wp) {
        ringbuffer_wp = scratch_db_base + offset;
        ASSERT(ringbuffer_wp <= ringbuffer_end);
    } else {
        ringbuffer_offset = offset;
    }

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void process_relay_ringbuffer_sub_cmds(uint32_t count, uint32_t* l1_cache) {
    CQPrefetchRelayRingbufferSubCmd tt_l1_ptr* sub_cmd = (CQPrefetchRelayRingbufferSubCmd tt_l1_ptr*)(l1_cache);
    ASSERT(count > 0);

    noc_async_read_barrier();
    uint32_t ringbuffer_start = ringbuffer_offset + scratch_db_base;

    for (uint32_t i = 0; i < count - 1; i++) {
        uint32_t start = ringbuffer_start + sub_cmd->start;
        uint32_t length = sub_cmd->length;

        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, start, length);

        cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages);
        sub_cmd++;
    }
    uint32_t start = ringbuffer_start + sub_cmd->start;
    uint32_t length = sub_cmd->length;
    uint32_t npages = write_pages_to_dispatcher<1, false>(downstream_data_ptr, start, length);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
    cb_release_pages<my_noc_index, downstream_noc_xy, downstream_cb_sem_id>(npages + 1);
    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);
}

template <bool cmddat_wrap_enable>
uint32_t process_relay_ringbuffer_cmd(uint32_t cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t count = cmd->relay_ringbuffer.count;
    uint32_t sub_cmds_length = count * sizeof(CQPrefetchRelayRingbufferSubCmd);
    uint32_t stride = cmd->relay_ringbuffer.stride;
    // DPRINT << "relay_ringbuffer: " << count << " " << cmd->relay_ringbuffer.stride << ENDL();

    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);
    uint32_t remaining = cmddat_q_end - data_ptr;
    uint32_t* l1_cache_pos = l1_cache;
    if (cmddat_wrap_enable && sub_cmds_length > remaining) {
        // wrap cmddat
        uint32_t amt = remaining / sizeof(uint32_t);
        careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
            (volatile uint32_t tt_l1_ptr*)(data_ptr), amt, l1_cache_pos);
        sub_cmds_length -= remaining;
        data_ptr = cmddat_q_base;
        l1_cache_pos += amt;
    }

    uint32_t amt = sub_cmds_length / sizeof(uint32_t);
    // Check that the final write does not overflow the L1 cache and corrupt the stack
    // End address of final write is: curr_offset_into_cache + write_size_rounded_up_to_copy_chunk
    ASSERT(
        (uint32_t)(l1_cache_pos +
                   ((amt + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
                       l1_to_local_cache_copy_chunk -
                   l1_cache) < l1_cache_elements_rounded);
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(data_ptr), amt, l1_cache_pos);

    process_relay_ringbuffer_sub_cmds(count, l1_cache);
    return stride;
}

// Separate implementation that fetches more data from exec buf when cmd has been split
static uint32_t process_exec_buf_relay_ringbuffer_cmd(
    uint32_t& cmd_ptr, uint32_t& downstream__data_ptr, uint32_t* l1_cache, PrefetchExecBufState& exec_buf_state) {
    volatile CQPrefetchCmd tt_l1_ptr* cmd = (volatile CQPrefetchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t count = cmd->relay_ringbuffer.count;
    uint32_t sub_cmds_length = count * sizeof(CQPrefetchRelayRingbufferSubCmd);
    uint32_t stride = cmd->relay_ringbuffer.stride;

    copy_into_l1_cache(cmd_ptr, sub_cmds_length, l1_cache, exec_buf_state, stride);

    process_relay_ringbuffer_sub_cmds(count, l1_cache);
    return stride;
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
            // DPRINT << "relay dram page: " << cmd_ptr << ENDL();
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

// This function is only valid when called on the H variant
// It expects the NoC async write state to be initialized to point to the downstream
static uint32_t process_relay_inline_all(uint32_t data_ptr, uint32_t fence, bool is_exec_buf) {
    uint32_t length = fence - data_ptr;
    // Downstream doesn't have FetchQ to tell it how much data to process
    // This packet header just contains the length
    volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader* dptr = (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader*)data_ptr;
    dptr->header.length = length;

    uint32_t npages = (length + downstream_cb_page_size - 1) >> downstream_cb_log_page_size;

    // Assume the dispatch buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages, my_downstream_cb_sem_additional_count);
    if (is_exec_buf) {
        // swipe all the downstream page credits from ourselves...
        // prefetch_h stalls sending commands to prefetch_d until notified by dispatch_d that the exec_buf is done
        // exec_buf completing on dispatch_h will free the pages and allow sending again
        my_downstream_cb_sem_additional_count -= downstream_cb_pages;

        // OK to continue prefetching once the page credits are returned
        stall_state = NOT_STALLED;
    }

    // Write sizes below may exceed NOC_MAX_BURST_SIZE so we use the any_len version
    // Amount to write depends on how much free space
    uint32_t downstream_pages_left = (downstream_cb_end - downstream_data_ptr) >> downstream_cb_log_page_size;
    if (downstream_pages_left >= npages) {
        // WAIT is not needed here because previous writes have already been flushed. Prefetch H only uses this
        // function and this function always flushes before returning
        relay_client
            .write_atomic_inc_any_len<my_noc_index, downstream_noc_xy, downstream_cb_sem_id, false, NCRISC_WR_CMD_BUF>(
                data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length, npages);
        downstream_data_ptr += npages * downstream_cb_page_size;
    } else {
        uint32_t tail_pages = npages - downstream_pages_left;
        uint32_t available = downstream_pages_left * downstream_cb_page_size;
        if (available > 0) {
            relay_client.write_any_len<my_noc_index, false, NCRISC_WR_CMD_BUF, true>(
                data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), available);
            data_ptr += available;
            length -= available;
        }

        // Remainder
        // WAIT is needed here because previously "if (available > 0)" then it used the write buf which may still be
        // busy at this point
        relay_client
            .write_atomic_inc_any_len<my_noc_index, downstream_noc_xy, downstream_cb_sem_id, true, NCRISC_WR_CMD_BUF>(
                data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_base), length, npages);

        downstream_data_ptr = downstream_cb_base + tail_pages * downstream_cb_page_size;
    }

    noc_async_writes_flushed();

    return fence;
}

// Gets cmds from upstream prefetch_h
// Note the prefetch_h uses the HostQ and grabs whole commands
// Shared command processor assumes whole commands are present, really
// just matters for the inline command which could be re-implemented
// This grabs whole (possibly sets of if multiple in a page) commands
inline uint32_t relay_cb_get_cmds(uint32_t& fence, uint32_t& data_ptr) {
    // DPRINT << "get_commands: " << data_ptr << " " << fence << " " << cmddat_q_base << " " << cmddat_q_end << ENDL();
    if (data_ptr == fence) {
        get_cb_page<cmddat_q_base, cmddat_q_blocks, cmddat_q_log_page_size, my_upstream_cb_sem_id>(
            data_ptr, fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
    }

    volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader* cmd_ptr =
        (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader*)data_ptr;
    uint32_t length = cmd_ptr->header.length;

    uint32_t pages_ready = (fence - data_ptr) >> cmddat_q_log_page_size;
    uint32_t pages_needed = (length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
    int32_t pages_pending = pages_needed - pages_ready;
    int32_t npages = 0;

    // TODO
    // Ugly: get_cb_page was written to process 1 page at a time, we need multiple
    // If it wraps, it resets the data_ptr to the top of the buffer, hand it a dummy for now
    uint32_t dummy_data_ptr = data_ptr;
    while (npages < pages_pending) {
        npages += get_cb_page<cmddat_q_base, cmddat_q_blocks, cmddat_q_log_page_size, my_upstream_cb_sem_id>(
            dummy_data_ptr, fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
        IDLE_ERISC_RETURN(length - sizeof(CQPrefetchHToPrefetchDHeader));
    }

    data_ptr += sizeof(CQPrefetchHToPrefetchDHeader);

    return length - sizeof(CQPrefetchHToPrefetchDHeader);
}

void kernel_main_h() {
    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;
    bool done = false;
    uint32_t heartbeat = 0;

    // Fetch q uses read buf. Write buf for process_relay_inline_all can be setup once
    relay_client.init<
        my_noc_index,
        fabric_mux_x,
        fabric_mux_y,
        worker_credits_stream_id,
        fabric_mux_channel_base_address,
        fabric_mux_flow_control_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_connection_info_address,
        fabric_mux_buffer_index_address,
        fabric_worker_flow_control_sem,
        fabric_worker_teardown_sem,
        fabric_worker_buffer_index_sem,
        fabric_mux_status_address,
        my_fabric_sync_status_addr,
        my_dev_id,
        to_dev_id,
        to_mesh_id,
        ew_dim,
        router_direction,
        fabric_header_rb_base,
        num_hops,
        NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0));

    while (!done) {
        fetch_q_get_cmds<sizeof(CQPrefetchHToPrefetchDHeader)>(fence, cmd_ptr, pcie_read_ptr);

        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        volatile CQPrefetchCmd tt_l1_ptr* cmd =
            (volatile CQPrefetchCmd tt_l1_ptr*)(cmd_ptr + sizeof(CQPrefetchHToPrefetchDHeader));
        uint32_t cmd_id = cmd->base.cmd_id;
        // Infer that an exec_buf command is to be executed based on the stall state.
        bool is_exec_buf = (stall_state == STALLED);
        cmd_ptr = process_relay_inline_all(cmd_ptr, fence, is_exec_buf);

        // Note: one fetch_q entry can contain multiple commands
        // The code below assumes these commands arrive individually, packing them would require parsing all cmds
        if (cmd_id == CQ_PREFETCH_CMD_TERMINATE) {
            // DPRINT << "prefetch terminating_10" << ENDL();
            done = true;
        }
    }
}

void kernel_main_d() {
    PrefetchExecBufState exec_buf_state;

    for (uint32_t i = 0; i < cmddat_q_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * cmddat_q_pages_per_block * cmddat_q_page_size;
        block_next_start_addr[i] = cmddat_q_base + offset;
    }

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

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
        fabric_mux_flow_control_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_connection_info_address,
        fabric_mux_buffer_index_address,
        fabric_worker_flow_control_sem,
        fabric_worker_teardown_sem,
        fabric_worker_buffer_index_sem,
        fabric_mux_status_address,
        my_fabric_sync_status_addr,
        my_dev_id,
        to_dev_id,
        to_mesh_id,
        ew_dim,
        router_direction,
        fabric_header_rb_base,
        num_hops,
        NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0));
#else
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), 0, my_noc_index);
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, DispatchSRelayInlineState::downstream_write_cmd_buf>(
        0, get_noc_addr_helper(dispatch_s_noc_xy, downstream_data_ptr_s), 0, my_noc_index);
#endif

    while (!done) {
        // cmds come in packed batches based on HostQ reads in prefetch_h
        // once a packed batch ends, we need to jump to the next page
        uint32_t length = relay_cb_get_cmds(fence, cmd_ptr);

        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        uint32_t amt_processed = 0;
        while (length > amt_processed) {
            uint32_t stride;
            done = process_cmd<true, false>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);
            amt_processed += stride;

            // This is ugly: relay_inline_cmd code can wrap and this can wrap
            // They peacefully coexist because we won't wrap there and here at once
            if (cmd_ptr + stride >= cmddat_q_end) {
                stride -= cmddat_q_end - cmd_ptr;
                cmd_ptr = cmddat_q_base;
                if (fence == cmddat_q_end) {
                    // We hit the nail on the head, wrap the fence
                    ASSERT(stride == 0);
                    fence = cmddat_q_base;
                }
            }
            cmd_ptr += stride;
        }

        // TODO: evaluate less costly free pattern (blocks?)
        uint32_t total_length = length + sizeof(CQPrefetchHToPrefetchDHeader);
        uint32_t pages_to_free = (total_length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
        relay_client.release_pages<my_noc_index, upstream_noc_xy, upstream_cb_sem_id>(pages_to_free);

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, cmddat_q_page_size);
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
#if defined(FABRIC_RELAY)
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": start (fabric relay. 2d = " << (uint32_t)is_2d_fabric
           << ")" << ENDL();
#else
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": start" << ENDL();
#endif

    if (is_h_variant and is_d_variant) {
        kernel_main_hd();
    } else if (is_h_variant) {
        kernel_main_h();
    } else if (is_d_variant) {
        kernel_main_d();
    } else {
        ASSERT(0);
    }
    IDLE_ERISC_RETURN();

    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_downstream_cb_sem_id>(downstream_cb_pages, my_downstream_cb_sem_additional_count);

    noc_async_full_barrier();

    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": out" << ENDL();
}
