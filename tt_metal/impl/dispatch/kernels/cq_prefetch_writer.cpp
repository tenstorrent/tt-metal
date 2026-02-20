// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch writer stub kernel
//  - Intended to run on NCRISC of the same core as the prefetch reader (BRISC)
//  - Currently a stub; future work will move write operations here from the reader
//
// This kernel receives the same compile-time defines as cq_prefetch_reader.cpp so that
// both kernels share a single compilation unit configuration.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_relay.hpp"
#include "api/debug/dprint.h"

// Constants used to interact with the downstream dispatchers.
// fd_core_type is already defined in cq_common.hpp.
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t dispatch_s_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_SUBORDINATE_NOC_X, DOWNSTREAM_SUBORDINATE_NOC_Y));
constexpr uint32_t downstream_cb_base = DOWNSTREAM_CB_BASE;
constexpr uint32_t dispatch_s_buffer_base = DISPATCH_S_BUFFER_BASE;
constexpr uint32_t scratch_db_base = SCRATCH_DB_BASE;
constexpr uint32_t downstream_cb_sem_id = DOWNSTREAM_CB_SEM_ID;
constexpr uint32_t downstream_dispatch_s_cb_sem_id = DOWNSTREAM_DISPATCH_S_CB_SEM_ID;

// Reference remaining defines to keep the compilation set consistent with cq_prefetch_reader.cpp.
[[maybe_unused]] constexpr uint32_t pw_my_downstream_cb_sem_id = MY_DOWNSTREAM_CB_SEM_ID;
[[maybe_unused]] constexpr uint32_t pw_my_dispatch_s_cb_sem_id = MY_DISPATCH_S_CB_SEM_ID;
[[maybe_unused]] constexpr uint32_t pw_downstream_cb_log_page_size = DOWNSTREAM_CB_LOG_PAGE_SIZE;
[[maybe_unused]] constexpr uint32_t pw_downstream_cb_pages = DOWNSTREAM_CB_PAGES;
[[maybe_unused]] constexpr uint32_t pw_pcie_base = PCIE_BASE;
[[maybe_unused]] constexpr uint32_t pw_pcie_size = PCIE_SIZE;
[[maybe_unused]] constexpr uint32_t pw_prefetch_q_base = PREFETCH_Q_BASE;
[[maybe_unused]] constexpr uint32_t pw_prefetch_q_size = PREFETCH_Q_SIZE;
[[maybe_unused]] constexpr uint32_t pw_prefetch_q_rd_ptr_addr = PREFETCH_Q_RD_PTR_ADDR;
[[maybe_unused]] constexpr uint32_t pw_prefetch_q_pcie_rd_ptr_addr = PREFETCH_Q_PCIE_RD_PTR_ADDR;
[[maybe_unused]] constexpr uint32_t pw_cmddat_q_base = CMDDAT_Q_BASE;
[[maybe_unused]] constexpr uint32_t pw_cmddat_q_size = CMDDAT_Q_SIZE;
[[maybe_unused]] constexpr uint32_t pw_scratch_db_size = SCRATCH_DB_SIZE;
[[maybe_unused]] constexpr uint32_t pw_downstream_sync_sem_id = DOWNSTREAM_SYNC_SEM_ID;
[[maybe_unused]] constexpr uint32_t pw_cmddat_q_pages = CMDDAT_Q_PAGES;
[[maybe_unused]] constexpr uint32_t pw_my_upstream_cb_sem_id = MY_UPSTREAM_CB_SEM_ID;
[[maybe_unused]] constexpr uint32_t pw_upstream_cb_sem_id = UPSTREAM_CB_SEM_ID;
[[maybe_unused]] constexpr uint32_t pw_cmddat_q_log_page_size = CMDDAT_Q_LOG_PAGE_SIZE;
[[maybe_unused]] constexpr uint32_t pw_cmddat_q_blocks = CMDDAT_Q_BLOCKS;
[[maybe_unused]] constexpr uint32_t pw_dispatch_s_buffer_size = DISPATCH_S_BUFFER_SIZE;
[[maybe_unused]] constexpr uint32_t pw_dispatch_s_cb_log_page_size = DISPATCH_S_CB_LOG_PAGE_SIZE;
[[maybe_unused]] constexpr uint32_t pw_ringbuffer_size = RINGBUFFER_SIZE;
[[maybe_unused]] constexpr uint32_t pw_fabric_header_rb_base = FABRIC_HEADER_RB_BASE;
[[maybe_unused]] constexpr uint32_t pw_fabric_header_rb_entries = FABRIC_HEADER_RB_ENTRIES;
[[maybe_unused]] constexpr uint32_t pw_my_fabric_sync_status_addr = MY_FABRIC_SYNC_STATUS_ADDR;
[[maybe_unused]] constexpr uint8_t pw_fabric_mux_x = FABRIC_MUX_X;
[[maybe_unused]] constexpr uint8_t pw_fabric_mux_y = FABRIC_MUX_Y;
[[maybe_unused]] constexpr uint8_t pw_fabric_mux_num_buffers_per_channel = FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL;
[[maybe_unused]] constexpr size_t pw_fabric_mux_channel_buffer_size_bytes = FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES;
[[maybe_unused]] constexpr size_t pw_fabric_mux_channel_base_address = FABRIC_MUX_CHANNEL_BASE_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_connection_info_address = FABRIC_MUX_CONNECTION_INFO_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_connection_handshake_address = FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_flow_control_address = FABRIC_MUX_FLOW_CONTROL_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_buffer_index_address = FABRIC_MUX_BUFFER_INDEX_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_status_address = FABRIC_MUX_STATUS_ADDRESS;
[[maybe_unused]] constexpr size_t pw_fabric_mux_termination_signal_address = FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS;
[[maybe_unused]] constexpr size_t pw_worker_credits_stream_id = WORKER_CREDITS_STREAM_ID;
[[maybe_unused]] constexpr size_t pw_fabric_worker_flow_control_sem = FABRIC_WORKER_FLOW_CONTROL_SEM;
[[maybe_unused]] constexpr size_t pw_fabric_worker_teardown_sem = FABRIC_WORKER_TEARDOWN_SEM;
[[maybe_unused]] constexpr size_t pw_fabric_worker_buffer_index_sem = FABRIC_WORKER_BUFFER_INDEX_SEM;
[[maybe_unused]] constexpr uint8_t pw_num_hops = NUM_HOPS;
[[maybe_unused]] constexpr uint32_t pw_ew_dim = EW_DIM;
[[maybe_unused]] constexpr uint32_t pw_to_mesh_id = TO_MESH_ID;
[[maybe_unused]] constexpr bool pw_is_2d_fabric = FABRIC_2D;
[[maybe_unused]] constexpr uint32_t pw_is_d_variant = IS_D_VARIANT;
[[maybe_unused]] constexpr uint32_t pw_is_h_variant = IS_H_VARIANT;

void kernel_main() {
    // Build a TERMINATE command in local L1 (scratch), NOC-write it to each downstream
    // dispatcher's CB, then signal each dispatcher via its semaphore.
    // No synchronisation with the reader (BRISC) is needed at this stub stage.

    volatile tt_l1_ptr CQDispatchCmd* local_cmd =
        reinterpret_cast<volatile tt_l1_ptr CQDispatchCmd*>(scratch_db_base);
    local_cmd->base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;

    // --- Regular dispatcher ---
    noc_async_write(
        scratch_db_base,
        get_noc_addr_helper(downstream_noc_xy, downstream_cb_base),
        sizeof(CQDispatchCmd));
    noc_async_writes_flushed();
    noc_semaphore_inc(
        get_noc_addr_helper(downstream_noc_xy, get_semaphore<fd_core_type>(downstream_cb_sem_id)), 1);

    // --- Subordinate dispatcher (dispatch_s) ---
    noc_async_write(
        scratch_db_base,
        get_noc_addr_helper(dispatch_s_noc_xy, dispatch_s_buffer_base),
        sizeof(CQDispatchCmd));
    noc_async_writes_flushed();
    noc_semaphore_inc(
        get_noc_addr_helper(dispatch_s_noc_xy, get_semaphore<fd_core_type>(downstream_dispatch_s_cb_sem_id)), 1);
}
