// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch kernel
//  - receives data in pages from prefetch kernel into the dispatch buffer ring buffer
//  - processes commands with embedded data from the dispatch buffer to write/sync/etc w/ destination
//  - sync w/ prefetcher is via 2 semaphores, page_ready, page_done
//  - page size must be a power of 2
//  - # blocks must evenly divide the dispatch buffer size
//  - dispatch buffer base must be page size aligned

#include "dataflow_api.h"
#include "dataflow_api_addrgen.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_relay.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

// The command queue write interface controls writes to the completion region, host owns the completion region read
// interface Data requests from device and event states are written to the completion region

CQWriteInterface cq_write_interface;

constexpr uint32_t dispatch_cb_base = DISPATCH_CB_BASE;
constexpr uint32_t dispatch_cb_log_page_size = DISPATCH_CB_LOG_PAGE_SIZE;
constexpr uint32_t dispatch_cb_pages = DISPATCH_CB_PAGES;
constexpr uint32_t my_dispatch_cb_sem_id = MY_DISPATCH_CB_SEM_ID;
constexpr uint32_t upstream_dispatch_cb_sem_id = UPSTREAM_DISPATCH_CB_SEM_ID;
constexpr uint32_t dispatch_cb_blocks = DISPATCH_CB_BLOCKS;
constexpr uint32_t upstream_sync_sem = UPSTREAM_SYNC_SEM;
constexpr uint32_t command_queue_base_addr = COMMAND_QUEUE_BASE_ADDR;
constexpr uint32_t completion_queue_base_addr = COMPLETION_QUEUE_BASE_ADDR;
constexpr uint32_t completion_queue_size = COMPLETION_QUEUE_SIZE;
constexpr uint32_t downstream_cb_base = DOWNSTREAM_CB_BASE;
constexpr uint32_t downstream_cb_size = DOWNSTREAM_CB_SIZE;
constexpr uint32_t my_downstream_cb_sem_id = MY_DOWNSTREAM_CB_SEM_ID;
constexpr uint32_t downstream_cb_sem_id = DOWNSTREAM_CB_SEM_ID;
constexpr uint32_t split_dispatch_page_preamble_size = SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE;
constexpr uint32_t split_prefetch = SPLIT_PREFETCH;
constexpr uint32_t prefetch_h_noc_xy = PREFETCH_H_NOC_XY;
constexpr uint32_t prefetch_h_local_downstream_sem_addr = PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR;
constexpr uint32_t prefetch_h_max_credits = PREFETCH_H_MAX_CREDITS;
constexpr uint32_t packed_write_max_unicast_sub_cmds =
    PACKED_WRITE_MAX_UNICAST_SUB_CMDS;  // Number of cores in compute grid
constexpr uint32_t dispatch_s_sync_sem_base_addr = DISPATCH_S_SYNC_SEM_BASE_ADDR;
constexpr uint32_t max_num_worker_sems = MAX_NUM_WORKER_SEMS;  // maximum number of worker semaphores
constexpr uint32_t max_num_go_signal_noc_data_entries =
    MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES;  // maximum number of go signal data words
constexpr uint32_t mcast_go_signal_addr = MCAST_GO_SIGNAL_ADDR;
constexpr uint32_t unicast_go_signal_addr = UNICAST_GO_SIGNAL_ADDR;
constexpr uint32_t distributed_dispatcher = DISTRIBUTED_DISPATCHER;
constexpr uint32_t host_completion_q_wr_ptr = HOST_COMPLETION_Q_WR_PTR;
constexpr uint32_t dev_completion_q_wr_ptr = DEV_COMPLETION_Q_WR_PTR;
constexpr uint32_t dev_completion_q_rd_ptr = DEV_COMPLETION_Q_RD_PTR;

constexpr uint32_t first_stream_used = FIRST_STREAM_USED;

constexpr uint32_t virtualize_unicast_cores = VIRTUALIZE_UNICAST_CORES;
constexpr uint32_t num_virtual_unicast_cores = NUM_VIRTUAL_UNICAST_CORES;
constexpr uint32_t num_physical_unicast_cores = NUM_PHYSICAL_UNICAST_CORES;

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

constexpr uint8_t num_hops = static_cast<uint8_t>(NUM_HOPS);

constexpr uint32_t my_dev_id = MY_DEV_ID;
constexpr uint32_t ew_dim = EW_DIM;
constexpr uint32_t to_mesh_id = TO_MESH_ID;
constexpr uint32_t to_dev_id = TO_DEV_ID;
constexpr uint32_t router_direction = ROUTER_DIRECTION;

constexpr bool is_2d_fabric = static_cast<bool>(FABRIC_2D);

constexpr uint32_t is_d_variant = IS_D_VARIANT;
constexpr uint32_t is_h_variant = IS_H_VARIANT;

constexpr uint8_t upstream_noc_index = UPSTREAM_NOC_INDEX;
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t dispatch_s_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_SUBORDINATE_NOC_X, DOWNSTREAM_SUBORDINATE_NOC_Y));
constexpr uint8_t my_noc_index = NOC_INDEX;
constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint64_t pcie_noc_xy =
    uint64_t(NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(PCIE_NOC_X), NOC_Y_PHYS_COORD(PCIE_NOC_Y)));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;

constexpr uint32_t completion_queue_end_addr = completion_queue_base_addr + completion_queue_size;
constexpr uint32_t completion_queue_page_size = dispatch_cb_page_size;
constexpr uint32_t completion_queue_log_page_size = dispatch_cb_log_page_size;
constexpr uint32_t completion_queue_size_16B = completion_queue_size >> 4;
constexpr uint32_t completion_queue_page_size_16B = completion_queue_page_size >> 4;
constexpr uint32_t completion_queue_end_addr_16B = completion_queue_end_addr >> 4;
constexpr uint32_t completion_queue_base_addr_16B = completion_queue_base_addr >> 4;
constexpr uint32_t dispatch_cb_size = dispatch_cb_page_size * dispatch_cb_pages;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + dispatch_cb_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + downstream_cb_size;

// Break buffer into blocks, 1/n of the total (dividing equally)
// Do bookkeeping (release, etc) based on blocks
// Note: due to the current method of release pages, up to 1 block of pages
// may be unavailable to the prefetcher at any time
constexpr uint32_t dispatch_cb_pages_per_block = dispatch_cb_pages / dispatch_cb_blocks;

static uint32_t rd_block_idx;

static uint32_t cb_fence;  // walks through cb page by page
static uint32_t cmd_ptr;   // walks through pages in cb cmd by cmd
static uint32_t downstream_cb_data_ptr = downstream_cb_base;
static uint32_t write_offset[CQ_DISPATCH_MAX_WRITE_OFFSETS];  // added to write address on non-host writes

static uint32_t upstream_total_acquired_page_count;

CQRelayClient<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes, fabric_header_rb_base>
    relay_client;

constexpr uint32_t packed_write_max_multicast_sub_cmds =
    get_packed_write_max_multicast_sub_cmds(packed_write_max_unicast_sub_cmds);
constexpr uint32_t max_write_packed_large_cmd =
    CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS * sizeof(CQDispatchWritePackedLargeSubCmd) / sizeof(uint32_t);
constexpr uint32_t max_write_packed_cmd =
    packed_write_max_unicast_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd) / sizeof(uint32_t);
constexpr uint32_t l1_cache_elements =
    (max_write_packed_cmd > max_write_packed_large_cmd) ? max_write_packed_cmd : max_write_packed_large_cmd;
constexpr uint32_t l1_cache_elements_rounded =
    ((l1_cache_elements + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
    l1_to_local_cache_copy_chunk;

// Used to send go signals asynchronously. Currently unused but this is a prototype for a GoSignalState
// ring buffer that can be used to store and then asynchronously send Go Signals.
struct GoSignalState {
    uint32_t go_signal;
    uint32_t wait_count;
};

static GoSignalState go_signal_state_ring_buf[4];
static uint8_t go_signal_state_wr_ptr = 0;
static uint8_t go_signal_state_rd_ptr = 0;

static uint32_t go_signal_noc_data[max_num_go_signal_noc_data_entries];

FORCE_INLINE volatile uint32_t* get_cq_completion_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(dev_completion_q_rd_ptr);
}

FORCE_INLINE volatile uint32_t* get_cq_completion_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(dev_completion_q_wr_ptr);
}

FORCE_INLINE
void completion_queue_reserve_back(uint32_t num_pages) {
    WAYPOINT("QRBW");
    // Transfer pages are aligned
    uint32_t data_size_16B = num_pages * completion_queue_page_size_16B;
    uint32_t completion_rd_ptr_and_toggle;
    uint32_t completion_rd_ptr;
    uint32_t completion_rd_toggle;
    uint32_t available_space;
    do {
        invalidate_l1_cache();
        completion_rd_ptr_and_toggle = *get_cq_completion_read_ptr();
        completion_rd_ptr = completion_rd_ptr_and_toggle & 0x7fffffff;
        completion_rd_toggle = completion_rd_ptr_and_toggle >> 31;
        // Toggles not equal means write ptr has wrapped but read ptr has not
        // so available space is distance from write ptr to read ptr
        // Toggles are equal means write ptr is ahead of read ptr
        // so available space is total space minus the distance from read to write ptr
        available_space =
            completion_rd_toggle != cq_write_interface.completion_fifo_wr_toggle
                ? completion_rd_ptr - cq_write_interface.completion_fifo_wr_ptr
                : (completion_queue_size_16B - (cq_write_interface.completion_fifo_wr_ptr - completion_rd_ptr));
    } while (data_size_16B > available_space);

    WAYPOINT("QRBD");
}

// This fn expects NOC coords to be preprogrammed
// Note that this fn does not increment any counters
FORCE_INLINE
void notify_host_of_completion_queue_write_pointer() {
    uint32_t completion_queue_write_ptr_addr = command_queue_base_addr + host_completion_q_wr_ptr;
    uint32_t completion_wr_ptr_and_toggle =
        cq_write_interface.completion_fifo_wr_ptr | (cq_write_interface.completion_fifo_wr_toggle << 31);
    volatile tt_l1_ptr uint32_t* completion_wr_ptr_addr = get_cq_completion_write_ptr();
    completion_wr_ptr_addr[0] = completion_wr_ptr_and_toggle;
#if defined(FABRIC_RELAY)
    noc_async_write(dev_completion_q_wr_ptr, pcie_noc_xy | completion_queue_write_ptr_addr, 4);
#else
    cq_noc_async_write_with_state<CQ_NOC_SnDL>(dev_completion_q_wr_ptr, completion_queue_write_ptr_addr, 4);
#endif
}

FORCE_INLINE
void completion_queue_push_back(uint32_t num_pages) {
    // Transfer pages are aligned
    uint32_t push_size_16B = num_pages * completion_queue_page_size_16B;
    cq_write_interface.completion_fifo_wr_ptr += push_size_16B;

    if (cq_write_interface.completion_fifo_wr_ptr >= completion_queue_end_addr_16B) {
        cq_write_interface.completion_fifo_wr_ptr =
            cq_write_interface.completion_fifo_wr_ptr - completion_queue_end_addr_16B + completion_queue_base_addr_16B;
        // Flip the toggle
        cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
    }

    // Notify host of updated completion wr ptr
    notify_host_of_completion_queue_write_pointer();
}

void process_write_host_h(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;

    uint32_t completion_write_ptr;
    // We will send the cmd back in the first X bytes, this makes the logic of reserving/pushing completion queue
    // pages much simpler since we are always sending writing full pages (except for last page)
    uint32_t length = cmd->write_linear_host.length;
    // DPRINT << "process_write_host_h: " << length << ENDL();
    uint32_t data_ptr = cmd_ptr;
#if !defined(FABRIC_RELAY)
    cq_noc_async_write_init_state<CQ_NOC_sNdl>(0, pcie_noc_xy, 0);
#endif
    while (length != 0) {
        // Get a page if needed
        if (cb_fence == data_ptr) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }
                if constexpr (is_h_variant && is_d_variant) {
                    move_rd_to_next_block_and_release_pages<
                        upstream_noc_index,
                        upstream_noc_xy,
                        upstream_dispatch_cb_sem_id,
                        dispatch_cb_pages_per_block,
                        dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
                } else {
                    move_rd_to_next_block_and_release_pages_remote<
                        upstream_noc_index,
                        upstream_noc_xy,
                        upstream_dispatch_cb_sem_id,
                        dispatch_cb_pages_per_block,
                        dispatch_cb_blocks>(relay_client, block_noc_writes_to_clear, rd_block_idx);
                }
            }
            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);

            cb_fence += n_pages * dispatch_cb_page_size;
        }
        uint32_t available_data = cb_fence - data_ptr;
        uint32_t xfer_size = (length > available_data) ? available_data : length;
        uint32_t npages = (xfer_size + completion_queue_page_size - 1) / completion_queue_page_size;
        completion_queue_reserve_back(npages);
        uint32_t completion_queue_write_addr = cq_write_interface.completion_fifo_wr_ptr << 4;
        // completion_queue_write_addr will never be equal to completion_queue_end_addr due to
        // completion_queue_push_back wrap logic so we don't need to handle this case explicitly to avoid 0 sized
        // transactions
        if (completion_queue_write_addr + xfer_size > completion_queue_end_addr) {
            uint32_t last_chunk_size = completion_queue_end_addr - completion_queue_write_addr;
#if defined(FABRIC_RELAY)
            noc_async_write(data_ptr, pcie_noc_xy | completion_queue_write_addr, last_chunk_size);
#else
            cq_noc_async_write_with_state_any_len(data_ptr, completion_queue_write_addr, last_chunk_size);
            uint32_t num_noc_packets_written = div_up(last_chunk_size, NOC_MAX_BURST_SIZE);
            noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
            noc_nonposted_writes_acked[noc_index] += num_noc_packets_written;
#endif
            completion_queue_write_addr = completion_queue_base_addr;
            data_ptr += last_chunk_size;
            length -= last_chunk_size;
            xfer_size -= last_chunk_size;
        }
#if defined(FABRIC_RELAY)
        noc_async_write(data_ptr, pcie_noc_xy | completion_queue_write_addr, xfer_size);
#else
        cq_noc_async_write_with_state_any_len(data_ptr, completion_queue_write_addr, xfer_size);
        // completion_queue_push_back below will do a write to host, so we add 1 to the number of data packets written
        uint32_t num_noc_packets_written = div_up(xfer_size, NOC_MAX_BURST_SIZE) + 1;
        noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
        noc_nonposted_writes_acked[noc_index] += num_noc_packets_written;
#endif

        // This will update the write ptr on device and host
        // We flush to ensure the ptr has been read out of l1 before we update it again
        completion_queue_push_back(npages);

        length -= xfer_size;
        data_ptr += xfer_size;
        noc_async_writes_flushed();
    }
    cmd_ptr = data_ptr;
}

void process_exec_buf_end_h() {
    if constexpr (split_prefetch) {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore<fd_core_type>(prefetch_h_local_downstream_sem_addr));

        noc_semaphore_inc(
            get_noc_addr_helper(prefetch_h_noc_xy, (uint32_t)sem_addr), prefetch_h_max_credits, noc_index);
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

// Relay, potentially through the mux/dmux/tunneller path
// Code below sends 1 page worth of data except at the end of a cmd
// This means the downstream buffers are always page aligned, simplifies wrap handling
template <uint32_t preamble_size>
void relay_to_next_cb(
    uint32_t data_ptr, uint32_t length, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    // TODO: Size for fabric
    static_assert(
        preamble_size == 0 || preamble_size == sizeof(tt::packet_queue::dispatch_packet_header_t),
        "Dispatcher preamble size must be 0 or sizeof(dispatch_packet_header_t)");

    // DPRINT << "relay_to_next_cb: " << data_ptr << " " << cb_fence << " " << length << ENDL();

    // First page should be valid since it has the command
    ASSERT(data_ptr <= dispatch_cb_end - dispatch_cb_page_size);
    ASSERT(data_ptr <= cb_fence - dispatch_cb_page_size);

    // regular write, inline writes, and atomic writes use different cmd bufs, so we can init state for each
    // TODO: Add support for stateful atomics. We can preserve state once cb_acquire_pages is changed to a free running
    // counter so we would only need to inc atomics downstream
    relay_client.init_write_state_only<my_noc_index, NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0));
    relay_client.init_inline_write_state_only<my_noc_index>(get_noc_addr_helper(downstream_noc_xy, 0));

    while (length > 0) {
        ASSERT(downstream_cb_end > downstream_cb_data_ptr);

        cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(1);

        uint32_t xfer_size;
        bool not_end_of_cmd;
        if (length > dispatch_cb_page_size - preamble_size) {
            xfer_size = dispatch_cb_page_size - preamble_size;
            not_end_of_cmd = true;
        } else {
            xfer_size = length;
            not_end_of_cmd = false;
        }

        if constexpr (preamble_size > 0) {
            uint32_t flag;
            relay_client.write_inline<my_noc_index>(
                get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr),
                xfer_size + preamble_size + not_end_of_cmd);
            downstream_cb_data_ptr += preamble_size;
            ASSERT(downstream_cb_data_ptr < downstream_cb_end);
        }
        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                uint32_t orphan_size = cb_fence - data_ptr;
                // No more writes from this block. Decrement the number of writes
                // since they were all accounted for.
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    ASSERT(cb_fence == dispatch_cb_end);
                    if (orphan_size != 0) {
                        relay_client.write<my_noc_index, true, NCRISC_WR_CMD_BUF>(
                            data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr), orphan_size);
                        length -= orphan_size;
                        xfer_size -= orphan_size;
                        downstream_cb_data_ptr += orphan_size;
                        if (downstream_cb_data_ptr == downstream_cb_end) {
                            downstream_cb_data_ptr = downstream_cb_base;
                        }
                        // All writes from this block have completed.
                        orphan_size = 0;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }

                move_rd_to_next_block_and_release_pages<
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block,
                    dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
            cb_fence += n_pages * dispatch_cb_page_size;
        }

        relay_client
            .write_atomic_inc_any_len<my_noc_index, downstream_noc_xy, downstream_cb_sem_id, true, NCRISC_WR_CMD_BUF>(
                data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr), xfer_size, 1);

        length -= xfer_size;
        data_ptr += xfer_size;
        downstream_cb_data_ptr += xfer_size;
        if (downstream_cb_data_ptr == downstream_cb_end) {
            downstream_cb_data_ptr = downstream_cb_base;
        }
    }

    // Move to next page
    downstream_cb_data_ptr = round_up_pow2(downstream_cb_data_ptr, dispatch_cb_page_size);
    if (downstream_cb_data_ptr == downstream_cb_end) {
        downstream_cb_data_ptr = downstream_cb_base;
    }

    cmd_ptr = data_ptr;
}

void process_write_host_d(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;
    // Remember: host transfer command includes the command in the payload, don't add it here
    uint32_t length = cmd->write_linear_host.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<split_dispatch_page_preamble_size>(
        data_ptr, length, block_noc_writes_to_clear, block_next_start_addr);
}

void relay_write_h(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;
    uint32_t length = sizeof(CQDispatchCmd) + cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<split_dispatch_page_preamble_size>(
        data_ptr, length, block_noc_writes_to_clear, block_next_start_addr);
}

void process_exec_buf_end_d(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    relay_to_next_cb<split_dispatch_page_preamble_size>(
        cmd_ptr, sizeof(CQDispatchCmd), block_noc_writes_to_clear, block_next_start_addr);
}

// Note that for non-paged writes, the number of writes per page is always 1
// This means each noc_write frees up a page
void process_write_linear(
    uint32_t num_mcast_dests, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;
    bool multicast = num_mcast_dests > 0;
    if (not multicast) {
        num_mcast_dests = 1;
    }

    uint32_t dst_noc = cmd->write_linear.noc_xy_addr;
    uint32_t write_offset_index = cmd->write_linear.write_offset_index;
    uint32_t dst_addr = cmd->write_linear.addr + write_offset[write_offset_index];
    uint32_t length = cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    if (multicast) {
        cq_noc_async_write_init_state<CQ_NOC_sNdl, true>(0, get_noc_addr_helper(dst_noc, dst_addr));
    } else {
        cq_noc_async_write_init_state<CQ_NOC_sNdl, false>(0, get_noc_addr_helper(dst_noc, dst_addr));
    }

    while (length != 0) {
        // More data needs to be written, but we've exhausted the CB. Acquire more pages.
        if (cb_fence == data_ptr) {
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }
                if constexpr (is_d_variant) {
                    // Dispatch HD/D upstream is local
                    move_rd_to_next_block_and_release_pages<
                        upstream_noc_index,
                        upstream_noc_xy,
                        upstream_dispatch_cb_sem_id,
                        dispatch_cb_pages_per_block,
                        dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
                } else {
                    // Dispatch H upstream is local
                    // Dispatch upstream is using NOC1 so init_noc_state again is not needed because this function
                    // uses non_dispatch_noc which is NOC0
                    move_rd_to_next_block_and_release_pages_remote<
                        upstream_noc_index,
                        upstream_noc_xy,
                        upstream_dispatch_cb_sem_id,
                        dispatch_cb_pages_per_block,
                        dispatch_cb_blocks>(relay_client, block_noc_writes_to_clear, rd_block_idx);
                }
            }
            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);

            cb_fence += n_pages * dispatch_cb_page_size;
        }
        // Transfer size is min(remaining_length, data_available_in_cb)
        uint32_t available_data = cb_fence - data_ptr;
        uint32_t xfer_size = length > available_data ? available_data : length;

        cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_mcast_dests);
        // Increment counters based on the number of packets that were written
        uint32_t num_noc_packets_written = div_up(xfer_size, NOC_MAX_BURST_SIZE);
        noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
        noc_nonposted_writes_acked[noc_index] += num_mcast_dests * num_noc_packets_written;
        length -= xfer_size;
        data_ptr += xfer_size;
        dst_addr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

void process_write(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;
    uint32_t num_mcast_dests = cmd->write_linear.num_mcast_dests;
    process_write_linear(num_mcast_dests, block_noc_writes_to_clear, block_next_start_addr);
}

template <bool is_dram>
void process_write_paged(uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;

    uint32_t page_id = cmd->write_paged.start_page;
    uint32_t base_addr = cmd->write_paged.base_addr;
    uint32_t page_size = cmd->write_paged.page_size;
    uint32_t pages = cmd->write_paged.pages;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    uint32_t write_length = pages * page_size;
    InterleavedAddrGen<is_dram> addr_gen{.bank_base_address = base_addr, .page_size = page_size};
    uint64_t dst_addr_offset = 0;  // Offset into page.

    // DPRINT << "process_write_paged - pages: " << pages << " page_size: " << page_size
    //        << " dispatch_cb_page_size: " << dispatch_cb_page_size << ENDL();

    while (write_length != 0) {
        // TODO #7360: Have more performant handling when page_size > dispatch_cb_page_size by not doing multiple writes
        // for one buffer page
        // More data needs to be written, but we've exhausted the CB. Acquire more pages.
        if (cb_fence == data_ptr) {
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }
                move_rd_to_next_block_and_release_pages<
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block,
                    dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }
            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);

            cb_fence += n_pages * dispatch_cb_page_size;
        }
        // Transfer size is min(remaining_length, data_available_in_cb)
        uint32_t available_data = cb_fence - data_ptr;
        uint32_t remaining_page_size = page_size - dst_addr_offset;
        uint32_t xfer_size = remaining_page_size > available_data ? available_data : remaining_page_size;
        // Cap the transfer size to the NOC packet size - use of One Packet NOC API (better performance
        // than writing a generic amount of data)
        xfer_size = xfer_size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : xfer_size;
        uint64_t dst = addr_gen.get_noc_addr(page_id, dst_addr_offset);

        noc_async_write<NOC_MAX_BURST_SIZE>(data_ptr, dst, xfer_size);
        // If paged write is not completed for a page (dispatch_cb_page_size < page_size) then add offset, otherwise
        // incr page_id.
        if (xfer_size < remaining_page_size) {
            // The above evaluates to: dst_addr_offset + xfer_size < page_size, but this saves a redundant calculation.
            dst_addr_offset += xfer_size;
        } else {
            page_id++;
            dst_addr_offset = 0;
        }

        write_length -= xfer_size;
        data_ptr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

// Packed write command
// Layout looks like:
//   - CQDispatchCmd struct
//   - count CQDispatchWritePackedSubCmd structs (max 1020)
//   - pad to L1 alignment
//   - count data packets of size size, each L1 aligned
//
// Note that there are multiple size restrictions on this cmd:
//  - all sub_cmds fit in one page
//  - size fits in one page
//
// Since all subcmds all appear in the first page and given the size restrictions
// this command can't be too many pages.  All pages are released at the end
template <bool mcast, typename WritePackedSubCmd>
void process_write_packed(
    uint32_t flags, uint32_t* l1_cache, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t count = cmd->write_packed.count;
    ASSERT(count <= (mcast ? packed_write_max_multicast_sub_cmds : packed_write_max_unicast_sub_cmds));
    constexpr uint32_t sub_cmd_size = sizeof(WritePackedSubCmd);
    // Copying in a burst is about a 30% net gain vs reading one value per loop below
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(cmd_ptr + sizeof(CQDispatchCmd)),
        count * sub_cmd_size / sizeof(uint32_t),
        l1_cache);

    uint32_t xfer_size = cmd->write_packed.size;
    uint32_t write_offset_index = cmd->write_packed.write_offset_index;
    uint32_t dst_addr = cmd->write_packed.addr + write_offset[write_offset_index];

    ASSERT(xfer_size <= dispatch_cb_page_size);

    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(WritePackedSubCmd);
    data_ptr = round_up_pow2(data_ptr, L1_ALIGNMENT);
    uint32_t stride =
        (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE) ? 0 : round_up_pow2(xfer_size, L1_ALIGNMENT);
    ASSERT(stride != 0 || data_ptr - cmd_ptr + xfer_size <= dispatch_cb_page_size);

    volatile uint32_t tt_l1_ptr* l1_addr = (uint32_t*)(cmd_ptr + sizeof(CQDispatchCmd));
    cq_noc_async_write_init_state<CQ_NOC_snDL, mcast>(0, dst_addr, xfer_size);

    // DPRINT << "dispatch_write_packed: " << xfer_size << " " << stride << " " << data_ptr << " " << count << " " <<
    // dst_addr << " " << ENDL();
    uint32_t writes = 0;
    uint32_t mcasts = 0;
    auto wait_for_barrier = [&]() {
        if (!mcast) {
            return;
        }
        noc_nonposted_writes_num_issued[noc_index] += writes;
        noc_nonposted_writes_acked[noc_index] += mcasts;
        writes = 0;
        mcasts = 0;
        // Workaround mcast path reservation hangs by always waiting for a write
        // barrier before doing an mcast that isn't linked to a previous mcast.
#ifdef TRACE_WRITE_BARRIERS
        DeviceZoneScopedN("noc_async_write_barrier");
#endif
        noc_async_write_barrier();
    };
    WritePackedSubCmd* sub_cmd_ptr = (WritePackedSubCmd*)l1_cache;
    while (count != 0) {
        uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
        uint32_t num_dests = mcast ? ((CQDispatchWritePackedMulticastSubCmd*)sub_cmd_ptr)->num_mcast_dests : 1;
        sub_cmd_ptr++;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);
        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion and issue orphan writes for this block
            // before proceeding to next block
            uint32_t orphan_size = 0;
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                orphan_size = cb_fence - data_ptr;
                if (orphan_size != 0) {
                    wait_for_barrier();
                    cq_noc_async_write_with_state<CQ_NOC_SNdL>(data_ptr, dst, orphan_size, num_dests);
                    writes++;
                    mcasts += num_dests;
                }
                // Handle wrapping on dispatch cb
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                } else {
                    data_ptr += orphan_size;
                }
                noc_nonposted_writes_num_issued[noc_index] += writes;
                noc_nonposted_writes_acked[noc_index] += mcasts;
                writes = 0;
                mcasts = 0;
                move_rd_to_next_block_and_release_pages<
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block,
                    dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
            cb_fence += n_pages * dispatch_cb_page_size;

            // This is done here so the common case doesn't have to restore the pointers
            if (orphan_size != 0) {
                uint32_t remainder_xfer_size = xfer_size - orphan_size;
                // Creating full NOC addr not needed as we are not programming the noc coords
                uint32_t remainder_dst_addr = dst_addr + orphan_size;
                wait_for_barrier();
                cq_noc_async_write_with_state<CQ_NOC_SnDL>(
                    data_ptr, remainder_dst_addr, remainder_xfer_size, num_dests);
                // Reset values expected below
                cq_noc_async_write_with_state<CQ_NOC_snDL, CQ_NOC_WAIT, CQ_NOC_send>(0, dst, xfer_size);
                writes++;
                mcasts += num_dests;

                count--;
                data_ptr += stride - orphan_size;

                continue;
            }
        }

        wait_for_barrier();
        cq_noc_async_write_with_state<CQ_NOC_SNdl>(data_ptr, dst, xfer_size, num_dests);
        writes++;
        mcasts += num_dests;

        count--;
        data_ptr += stride;
    }

    noc_nonposted_writes_num_issued[noc_index] += writes;
    noc_nonposted_writes_acked[noc_index] += mcasts;

    cmd_ptr = data_ptr;
}

// This routine below can be implemented to either prefetch sub_cmds into local memory or leave them in L1
// Prefetching into local memory limits the number of sub_cmds (used as kernel writes) in one cmd
// Leaving in L1 limits the number of bytes of data in one cmd (whole command must fit in CB)
//
// The code below prefetches sub_scmds into local cache because:
//  - it is likely faster (not measured yet, but base based on write_packed)
//  - allows pages to be released as they are processed (since prefetcher won't overwrite the sub-cmds)
//  - can presently handle 36 subcmds, or 7 5-processor kernels
// Without prefetching:
//  - cmd size is limited to CB size which is 128K and may go to 192K
//  - w/ 4K kernel binaries, 192K is 9 5-processor kernels, 128K is 6
//  - utilizing the full space creates a full prefetcher stall as all memory is tied up
//  - so a better practical full size is 3-4 full sets of 4K kernel binaries
// May eventually want a separate implementation for tensix vs eth dispatch
void process_write_packed_large(
    uint32_t* l1_cache, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t count = cmd->write_packed_large.count;
    uint32_t alignment = cmd->write_packed_large.alignment;
    uint32_t write_offset_index = cmd->write_packed_large.write_offset_index;
    uint32_t local_write_offset = write_offset[write_offset_index];
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(CQDispatchWritePackedLargeSubCmd);
    data_ptr = round_up_pow2(data_ptr, L1_ALIGNMENT);

    constexpr uint32_t sub_cmd_size = sizeof(CQDispatchWritePackedLargeSubCmd);
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(cmd_ptr + sizeof(CQDispatchCmd)),
        count * sub_cmd_size / sizeof(uint32_t),
        l1_cache);

    uint32_t writes = 0;
    uint32_t mcasts = noc_nonposted_writes_acked[noc_index];
    CQDispatchWritePackedLargeSubCmd* sub_cmd_ptr = (CQDispatchWritePackedLargeSubCmd*)l1_cache;

    bool init_state = true;
    bool must_barrier = true;
    while (count != 0) {
        uint32_t dst_addr = sub_cmd_ptr->addr + local_write_offset;
        uint32_t length = sub_cmd_ptr->length;
        uint32_t num_dests = sub_cmd_ptr->num_mcast_dests;
        uint32_t pad_size = align_power_of_2(length, alignment) - length;
        uint32_t unlink = sub_cmd_ptr->flags & CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
        auto wait_for_barrier = [&]() {
            if (!must_barrier) {
                return;
            }
            noc_nonposted_writes_num_issued[noc_index] += writes;

            mcasts += num_dests * writes;
            noc_nonposted_writes_acked[noc_index] = mcasts;
            writes = 0;
            // Workaround mcast path reservation hangs by always waiting for a write
            // barrier before doing an mcast that isn't linked to a previous mcast.
#ifdef TRACE_WRITE_BARRIERS
            DeviceZoneScopedN("noc_async_write_barrier");
#endif
            noc_async_write_barrier();
        };

        // Only re-init state after we have unlinked the last transaction
        // Otherwise we assume NOC coord hasn't changed
        // TODO: If we are able to send 0 length txn to unset link, we don't need a flag and can compare dst_noc to prev
        // to determine linking
        if (init_state) {
            uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
            cq_noc_async_write_init_state<CQ_NOC_sNdl, true, true>(0, get_noc_addr_helper(dst_noc, dst_addr));
            must_barrier = true;
        }

        sub_cmd_ptr++;

        while (length != 0) {
            // More data needs to be written, but we've exhausted the CB. Acquire more pages.
            if (data_ptr == cb_fence) {
                if (cb_fence == block_next_start_addr[rd_block_idx]) {
                    if (rd_block_idx == dispatch_cb_blocks - 1) {
                        cb_fence = dispatch_cb_base;
                        data_ptr = dispatch_cb_base;
                    }
                    // Block completion - account for all writes issued for this block before moving to next
                    noc_nonposted_writes_num_issued[noc_index] += writes;
                    mcasts += num_dests * writes;
                    writes = 0;
                    move_rd_to_next_block_and_release_pages<
                        upstream_noc_index,
                        upstream_noc_xy,
                        upstream_dispatch_cb_sem_id,
                        dispatch_cb_pages_per_block,
                        dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
                }
                uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                    cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
                cb_fence += n_pages * dispatch_cb_page_size;
            }
            // Transfer size is min(remaining_length, data_available_in_cb)
            uint32_t available_data = cb_fence - data_ptr;
            uint32_t xfer_size;
            if (length > available_data) {
                xfer_size = available_data;
                wait_for_barrier();
                cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_dests);
                must_barrier = false;
            } else {
                xfer_size = length;
                if (unlink) {
                    wait_for_barrier();
                    uint32_t rem_xfer_size =
                        cq_noc_async_write_with_state_any_len<false>(data_ptr, dst_addr, xfer_size, num_dests);
                    // Unset Link flag
                    cq_noc_async_write_init_state<CQ_NOC_sndl, true, false>(0, 0, 0);
                    uint32_t data_offset = xfer_size - rem_xfer_size;
                    cq_noc_async_write_with_state<CQ_NOC_SnDL, CQ_NOC_wait>(
                        data_ptr + data_offset, dst_addr + data_offset, rem_xfer_size, num_dests);
                    // Later writes must barrier, but the `must_barrier = true` in the `if (init_state)` block above
                    // will see to that.
                } else {
                    wait_for_barrier();
                    cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_dests);
                    must_barrier = false;
                }
            }
            writes += div_up(xfer_size, NOC_MAX_BURST_SIZE);
            length -= xfer_size;
            data_ptr += xfer_size;
            dst_addr += xfer_size;
        }

        init_state = unlink;

        noc_nonposted_writes_num_issued[noc_index] += writes;
        mcasts += num_dests * writes;
        writes = 0;

        // Handle padded size and potential wrap
        if (data_ptr + pad_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    ASSERT(cb_fence == dispatch_cb_end);
                    uint32_t orphan_size = cb_fence - data_ptr;
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    pad_size -= orphan_size;
                }
                move_rd_to_next_block_and_release_pages<
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block,
                    dispatch_cb_blocks>(block_noc_writes_to_clear, rd_block_idx);
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = cb_acquire_pages<my_dispatch_cb_sem_id, dispatch_cb_log_page_size>(
                cb_fence, block_next_start_addr, rd_block_idx, upstream_total_acquired_page_count);
            cb_fence += n_pages * dispatch_cb_page_size;
        }
        data_ptr += pad_size;

        count--;
    }
    noc_nonposted_writes_acked[noc_index] = mcasts;

    cmd_ptr = data_ptr;
}

static uint32_t process_debug_cmd(uint32_t cmd_ptr) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    return cmd_ptr + cmd->debug.stride;
}

FORCE_INLINE
uint32_t stream_wrap_ge(uint32_t a, uint32_t b) {
    constexpr uint32_t shift = 32 - MEM_WORD_ADDR_WIDTH;
    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return (diff << shift) >= 0;
}

static void process_wait() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    auto flags = cmd->wait.flags;

    uint32_t barrier = flags & CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER;
    uint32_t notify_prefetch = flags & CQ_DISPATCH_CMD_WAIT_FLAG_NOTIFY_PREFETCH;
    uint32_t clear_stream = flags & CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM;
    uint32_t wait_memory = flags & CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY;
    uint32_t wait_stream = flags & CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM;
    uint32_t count = cmd->wait.count;
    uint32_t stream = cmd->wait.stream;

    if (barrier) {
        // DPRINT << " DISPATCH BARRIER\n";
#ifdef TRACE_WRITE_BARRIERS
        DeviceZoneScopedN("noc_async_write_barrier");
#endif
        noc_async_write_barrier();
    }

    WAYPOINT("PWW");
    uint32_t heartbeat = 0;
    if (wait_memory) {
        uint32_t addr = cmd->wait.addr;
        volatile tt_l1_ptr uint32_t* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        // DPRINT << " DISPATCH WAIT " << HEX() << addr << DEC() << " count " << count << ENDL();
        do {
            invalidate_l1_cache();
            IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        } while (!wrap_ge(*sem_addr, count));
    }
    if (wait_stream) {
        volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        // DPRINT << " DISPATCH WAIT STREAM " << HEX() << stream << DEC() << " count " << count << ENDL();
        do {
            IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        } while (!stream_wrap_ge(*sem_addr, count));
    }
    WAYPOINT("PWD");

    if (clear_stream) {
        volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        uint32_t neg_sem_val = -(*sem_addr);
        NOC_STREAM_WRITE_REG(
            stream,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            neg_sem_val << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }
    if (notify_prefetch) {
        noc_semaphore_inc(
            get_noc_addr_helper(upstream_noc_xy, get_semaphore<fd_core_type>(upstream_sync_sem)),
            1,
            upstream_noc_index);
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

static void process_delay_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t count = cmd->delay.delay;
    for (volatile uint32_t i = 0; i < count; i++);
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t stream = cmd->mcast.wait_stream;
    // The location of the go signal embedded in the command does not meet NOC alignment requirements.
    // cmd_ptr is guaranteed to meet the alignment requirements, since it is written to by prefetcher over NOC.
    // Copy the go signal from an unaligned location to an aligned (cmd_ptr) location. This is safe as long as we
    // can guarantee that copying the go signal does not corrupt any other command fields, which is true (see
    // CQDispatchGoSignalMcastCmd).
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage = (volatile uint32_t tt_l1_ptr*)cmd_ptr;
    *aligned_go_signal_storage = cmd->mcast.go_signal;
    uint8_t go_signal_noc_data_idx = cmd->mcast.noc_data_start_index;
    if (cmd->mcast.num_mcast_txns > 0) {
        // Setup registers before waiting for workers so only the NOC_CMD_CTRL register needs to be touched after.
        uint64_t dst_noc_addr_multicast =
            get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], mcast_go_signal_addr);
        uint32_t num_dests = go_signal_noc_data[go_signal_noc_data_idx++];
        cq_noc_async_write_init_state<CQ_NOC_SNDL, true>(
            (uint32_t)aligned_go_signal_storage, dst_noc_addr_multicast, sizeof(uint32_t));
        noc_nonposted_writes_acked[noc_index] += num_dests;

        while (!stream_wrap_ge(
            NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), cmd->mcast.wait_count)) {
        }
        cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>(0, 0, 0);
        // Send GO signal to remaining destinations. Only the destination NOC needs to be modified.
        for (uint32_t i = 1, num_mcasts = cmd->mcast.num_mcast_txns; i < num_mcasts; ++i) {
            uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], mcast_go_signal_addr);
            uint32_t num_dests = go_signal_noc_data[go_signal_noc_data_idx++];
            cq_noc_async_write_with_state<CQ_NOC_sNdl>(0, dst, 0);
            noc_nonposted_writes_acked[noc_index] += num_dests;
        }
        noc_nonposted_writes_num_issued[noc_index] += cmd->mcast.num_mcast_txns;
    } else {
        while (!stream_wrap_ge(
            NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), cmd->mcast.wait_count)) {
        }
    }

    uint32_t num_unicasts = cmd->mcast.num_unicast_txns;
    if constexpr (virtualize_unicast_cores) {
        // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
        // This chip is virtualizing cores the go signal is unicasted to
        // In this case, the number of unicasts specified in the command can exceed
        // the number of actual cores on this chip.
        if (cmd->mcast.num_unicast_txns > num_physical_unicast_cores) {
            // If this is the case, cap the number of unicasts to avoid invalid NOC txns
            num_unicasts = num_physical_unicast_cores;
            // Fake updates from non-existent workers here. The dispatcher expects an ack from
            // the number of cores specified inside cmd->mcast.num_unicast_txns. If this is
            // greater than the number of cores actually on the chip, we must account for acks
            // from non-existent cores here.
            NOC_STREAM_WRITE_REG(
                stream,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
                (num_virtual_unicast_cores - num_physical_unicast_cores) << REMOTE_DEST_BUF_WORDS_FREE_INC);
        }
    }

    for (uint32_t i = 0; i < num_unicasts; ++i) {
        uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], unicast_go_signal_addr);
        noc_async_write_one_packet((uint32_t)(aligned_go_signal_storage), dst, sizeof(uint32_t));
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_notify_dispatch_s_go_signal_cmd() {
    // Update free running counter on dispatch_s, signalling that it's safe to send a go signal to workers
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t wait = cmd->notify_dispatch_s_go_signal.wait;
    // write barrier to wait before sending the go signal
    if (wait) {
        // DPRINT << " DISPATCH_S_NOTIFY BARRIER\n";
#ifdef TRACE_WRITE_BARRIERS
        DeviceZoneScopedN("noc_async_write_barrier");
#endif
        noc_async_write_barrier();
    }
    uint16_t index_bitmask = cmd->notify_dispatch_s_go_signal.index_bitmask;

    while (index_bitmask != 0) {
        uint32_t set_index = __builtin_ctz(index_bitmask);
        uint32_t dispatch_s_sync_sem_addr = dispatch_s_sync_sem_base_addr + set_index * L1_ALIGNMENT;
        if constexpr (distributed_dispatcher) {
            static uint32_t num_go_signals_safe_to_send[max_num_worker_sems] = {0};
            uint64_t dispatch_s_notify_addr = get_noc_addr_helper(dispatch_s_noc_xy, dispatch_s_sync_sem_addr);
            num_go_signals_safe_to_send[set_index]++;
            noc_inline_dw_write(dispatch_s_notify_addr, num_go_signals_safe_to_send[set_index]);
        } else {
            tt_l1_ptr uint32_t* notify_ptr = (uint32_t tt_l1_ptr*)(dispatch_s_sync_sem_addr);
            *notify_ptr = (*notify_ptr) + 1;
        }
        // Unset the bit
        index_bitmask &= index_bitmask - 1;
    }
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_go_signal_noc_data() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t num_words = cmd->set_go_signal_noc_data.num_words;
    ASSERT(num_words <= max_num_go_signal_noc_data_entries);
    volatile tt_l1_ptr uint32_t* data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cmd_ptr + sizeof(CQDispatchCmd));
    for (uint32_t i = 0; i < num_words; ++i) {
        go_signal_noc_data[i] = *(data_ptr++);
    }
    cmd_ptr = round_up_pow2((uint32_t)data_ptr, L1_ALIGNMENT);
}

static inline bool process_cmd_d(
    uint32_t& cmd_ptr, uint32_t* l1_cache, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    bool done = false;
re_run_command:
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    DeviceTimestampedData("process_cmd_d_dispatch", (uint32_t)cmd->base.cmd_id);
    switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR:
            WAYPOINT("DWB");
            // DPRINT << "cmd_write_linear\n";
            process_write(block_noc_writes_to_clear, block_next_start_addr);
            WAYPOINT("DWD");
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
            // DPRINT << "cmd_write_linear_h\n";
            if (is_h_variant) {
                process_write(block_noc_writes_to_clear, block_next_start_addr);
            } else {
                relay_write_h(block_noc_writes_to_clear, block_next_start_addr);
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            // DPRINT << "cmd_write_linear_h_host\n";
            if (is_h_variant) {
                process_write_host_h(block_noc_writes_to_clear, block_next_start_addr);
            } else {
                process_write_host_d(block_noc_writes_to_clear, block_next_start_addr);
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PAGED:
            // DPRINT << "cmd_write_paged is_dram: " << (uint32_t)cmd->write_paged.is_dram << ENDL();
            if (cmd->write_paged.is_dram) {
                process_write_paged<true>(block_noc_writes_to_clear, block_next_start_addr);
            } else {
                process_write_paged<false>(block_noc_writes_to_clear, block_next_start_addr);
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED: {
            // DPRINT << "cmd_write_packed" << ENDL();
            uint32_t flags = cmd->write_packed.flags;
            // Must match unpacking code in tt_metal/impl/profiler/profiler.cpp.
            uint32_t data = ((flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_MASK) >>
                             (CQ_DISPATCH_CMD_PACKED_WRITE_TYPE_SHIFT - 1)) |
                            bool(flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST);
            DeviceTimestampedData("packed_data_dispatch", data);
            if (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST) {
                process_write_packed<true, CQDispatchWritePackedMulticastSubCmd>(
                    flags, l1_cache, block_noc_writes_to_clear, block_next_start_addr);
            } else {
                process_write_packed<false, CQDispatchWritePackedUnicastSubCmd>(
                    flags, l1_cache, block_noc_writes_to_clear, block_next_start_addr);
            }
        } break;

        case CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL:
            // DPRINT << "cmd_notify_dispatch_s_go_signal" << ENDL();
            process_notify_dispatch_s_go_signal_cmd();
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:
            // DPRINT << "cmd_write_packed_large" << ENDL();
            // Must match unpacking code in tt_metal/impl/profiler/profiler.cpp.
            DeviceTimestampedData("packed_large_data_dispatch", cmd->write_packed_large.type);
            process_write_packed_large(l1_cache, block_noc_writes_to_clear, block_next_start_addr);
            break;

        case CQ_DISPATCH_CMD_WAIT:
            // DPRINT << "cmd_wait" << ENDL();
            process_wait();
            break;

        case CQ_DISPATCH_CMD_SINK: DPRINT << "cmd_sink" << ENDL(); break;

        case CQ_DISPATCH_CMD_DEBUG:
            DPRINT << "cmd_debug" << ENDL();
            cmd_ptr = process_debug_cmd(cmd_ptr);
            goto re_run_command;
            break;

        case CQ_DISPATCH_CMD_DELAY:
            DPRINT << "cmd_delay" << ENDL();
            process_delay_cmd();
            break;

        case CQ_DISPATCH_CMD_EXEC_BUF_END:
            // DPRINT << "cmd_exec_buf_end\n";
            if (is_h_variant) {
                process_exec_buf_end_h();
            } else {
                process_exec_buf_end_d(block_noc_writes_to_clear, block_next_start_addr);
            }
            break;

        case CQ_DISPATCH_CMD_SEND_GO_SIGNAL:
            // DPRINT << "cmd_go_send_go_signal" << ENDL();
            process_go_signal_mcast_cmd();
            break;

        case CQ_DISPATCH_SET_NUM_WORKER_SEMS:
            // DPRINT << "cmd_set_num_worker_sems" << ENDL();
            // This command is only used by dispatch_s
            ASSERT(0);
            cmd_ptr += sizeof(CQDispatchCmd);
            break;

        case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA: set_go_signal_noc_data(); break;

        case CQ_DISPATCH_CMD_SET_WRITE_OFFSET: {
            // DPRINT << "write offset: " << cmd->set_write_offset.offset0 << " " << cmd->set_write_offset.offset1 << "
            // "
            //        << cmd->set_write_offset.offset2 << " host id " << cmd->set_write_offset.program_host_id <<
            //        ENDL();
            DeviceTimestampedData("runtime_host_id_dispatch", cmd->set_write_offset.program_host_id);
            uint32_t offset_count = cmd->set_write_offset.offset_count;

            ASSERT(offset_count <= std::size(write_offset));
            uint32_t* cmd_write_offset = (uint32_t*)(cmd_ptr + sizeof(CQDispatchCmd));

            for (uint32_t i = 0; i < offset_count; i++) {
                write_offset[i] = cmd_write_offset[i];
            }
            cmd_ptr += sizeof(CQDispatchCmd) + sizeof(uint32_t) * offset_count;
            break;
        }

        case CQ_DISPATCH_CMD_TERMINATE:
            // DPRINT << "dispatch terminate\n";
            if (is_d_variant && !is_h_variant) {
                relay_to_next_cb<split_dispatch_page_preamble_size>(
                    cmd_ptr, sizeof(CQDispatchCmd), block_noc_writes_to_clear, block_next_start_addr);
            }
            cmd_ptr += sizeof(CQDispatchCmd);
            done = true;
            break;

        default:
            DPRINT << "dispatcher_d invalid command:" << cmd_ptr << " " << cb_fence << " " << dispatch_cb_base << " "
                   << dispatch_cb_end << " " << rd_block_idx << " "
                   << "xx" << ENDL();
            DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 1) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 2) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 3) << ENDL();
            WAYPOINT("!CMD");
            ASSERT(0);
    }

    return done;
}

static inline bool process_cmd_h(
    uint32_t& cmd_ptr, uint32_t& block_noc_writes_to_clear, uint32_t block_next_start_addr[]) {
    bool done = false;

    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;

    DeviceTimestampedData("process_cmd_h_dispatch", (uint32_t)cmd->base.cmd_id);
    switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
            // DPRINT << "dispatch_h write_linear_h\n";
            process_write(block_noc_writes_to_clear, block_next_start_addr);
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            // DPRINT << "dispatch_h linear_h_host\n";
            process_write_host_h(block_noc_writes_to_clear, block_next_start_addr);
            break;

        case CQ_DISPATCH_CMD_EXEC_BUF_END:
            // DPRINT << "dispatch_h exec_buf_end\n";
            process_exec_buf_end_h();
            break;
        case CQ_DISPATCH_CMD_TERMINATE:
            // DPRINT << "dispatch_h terminate\n";
            cmd_ptr += sizeof(CQDispatchCmd);
            done = true;
            break;

        default:
            DPRINT << "dispatcher_h invalid command:" << cmd_ptr << " " << cb_fence << " "
                   << " " << dispatch_cb_base << " " << dispatch_cb_end << " " << rd_block_idx << " "
                   << "xx" << ENDL();
            DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 1) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 2) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr + 3) << ENDL();
            WAYPOINT("!CMD");
            ASSERT(0);
    }

    return done;
}

void kernel_main() {
#if defined(FABRIC_RELAY)
    DPRINT << "dispatch_" << is_h_variant << is_d_variant << ": start (fabric relay. 2d = " << (uint32_t)is_2d_fabric
           << ")" << ENDL();
#else
    DPRINT << "dispatch_" << is_h_variant << is_d_variant << ": start" << ENDL();
#endif

    // Initialize local state of any additional nocs used instead of the default
    static_assert(my_noc_index != upstream_noc_index);
    if constexpr (my_noc_index != upstream_noc_index) {
        noc_local_state_init(upstream_noc_index);
    }

    for (size_t i = 0; i < max_num_worker_sems; i++) {
        uint32_t index = i + first_stream_used;

        NOC_STREAM_WRITE_REG(
            index,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            -NOC_STREAM_READ_REG(index, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)
                << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    static_assert(is_d_variant || split_dispatch_page_preamble_size == 0);

    upstream_total_acquired_page_count = 0;

    uint32_t block_next_start_addr[dispatch_cb_blocks];
    uint32_t l1_cache[l1_cache_elements_rounded];

    for (uint32_t i = 0; i < dispatch_cb_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * dispatch_cb_pages_per_block * dispatch_cb_page_size;
        block_next_start_addr[i] = dispatch_cb_base + offset;
    }

    cb_fence = dispatch_cb_base;
    rd_block_idx = 0;
    uint32_t block_noc_writes_to_clear = noc_nonposted_writes_num_issued[noc_index];
    cmd_ptr = dispatch_cb_base;
    write_offset[0] = 0;
    write_offset[1] = 0;
    write_offset[2] = 0;

    {
        uint32_t completion_queue_wr_ptr_and_toggle = *get_cq_completion_write_ptr();
        cq_write_interface.completion_fifo_wr_ptr = completion_queue_wr_ptr_and_toggle & 0x7fffffff;
        cq_write_interface.completion_fifo_wr_toggle = completion_queue_wr_ptr_and_toggle >> 31;
    }
    // Initialize the relay client for split dispatch
    if constexpr (!(is_h_variant && is_d_variant)) {
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
#endif
    }
    bool done = false;
    uint32_t heartbeat = 0;
    while (!done) {
        if (cmd_ptr == cb_fence) {
            if constexpr (is_h_variant && !is_d_variant) {
                get_cb_page_and_release_pages_remote<
                    dispatch_cb_base,
                    dispatch_cb_blocks,
                    dispatch_cb_log_page_size,
                    my_dispatch_cb_sem_id,
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block>(
                    relay_client,
                    cmd_ptr,
                    cb_fence,
                    block_noc_writes_to_clear,
                    block_next_start_addr,
                    rd_block_idx,
                    upstream_total_acquired_page_count);
            } else {
                get_cb_page_and_release_pages<
                    dispatch_cb_base,
                    dispatch_cb_blocks,
                    dispatch_cb_log_page_size,
                    my_dispatch_cb_sem_id,
                    upstream_noc_index,
                    upstream_noc_xy,
                    upstream_dispatch_cb_sem_id,
                    dispatch_cb_pages_per_block>(
                    cmd_ptr,
                    cb_fence,
                    block_noc_writes_to_clear,
                    block_next_start_addr,
                    rd_block_idx,
                    upstream_total_acquired_page_count);
            }
        }

        DeviceZoneScopedN("CQ-DISPATCH");
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        done = is_d_variant ? process_cmd_d(cmd_ptr, l1_cache, block_noc_writes_to_clear, block_next_start_addr)
                            : process_cmd_h(cmd_ptr, block_noc_writes_to_clear, block_next_start_addr);

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, dispatch_cb_page_size);
    }

    // Release any held pages from the previous block
    if (is_h_variant && !is_d_variant) {
        cb_block_release_pages_remote<
            upstream_noc_index,
            upstream_noc_xy,
            upstream_dispatch_cb_sem_id,
            dispatch_cb_pages_per_block>(relay_client, block_noc_writes_to_clear);
    } else {
        cb_block_release_pages<
            upstream_noc_index,
            upstream_noc_xy,
            upstream_dispatch_cb_sem_id,
            dispatch_cb_pages_per_block>(block_noc_writes_to_clear);
    }

    // Release any held pages from the current block
    uint32_t npages =
        dispatch_cb_pages_per_block - ((block_next_start_addr[rd_block_idx] - cmd_ptr) >> dispatch_cb_log_page_size);
    if (is_h_variant && !is_d_variant) {
        relay_client.release_pages<upstream_noc_index, upstream_noc_xy, upstream_dispatch_cb_sem_id>(npages);
    } else {
        cb_release_pages<upstream_noc_index, upstream_noc_xy, upstream_dispatch_cb_sem_id>(npages);
    }

    noc_async_write_barrier();

    // Confirm expected number of pages, spinning here is a leak
    cb_wait_all_pages<my_dispatch_cb_sem_id>(upstream_total_acquired_page_count);

    noc_async_full_barrier();

    if (is_h_variant && !is_d_variant) {
        relay_client.template teardown<upstream_noc_index, upstream_noc_xy, upstream_dispatch_cb_sem_id>();
    }
    // DPRINT << "dispatch_" << is_h_variant << is_d_variant << ": out" << ENDL();
}
