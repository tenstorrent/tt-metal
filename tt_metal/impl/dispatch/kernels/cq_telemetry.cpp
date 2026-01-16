// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Telemetry kernel
// Pushes telemetry data from L1 buffer to host via D2H socket
// Uses Brisc NOC 0 command buffer 0 for PCIe writes

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/dev_msgs.h"
#include "tt_metal/impl/dispatch/kernels/perf_telemetry.hpp"
#include "api/debug/dprint.h"

// Perf telemetry page size - must match host-side kPerfTelemetryPageSize
constexpr uint32_t perf_telemetry_page_size = 64;

// Size of timestamp data to read from dispatch core (kernel_start + kernel_end)
constexpr uint32_t telemetry_timestamp_size = 2 * sizeof(telemetry_timestamp_t);  // 32 bytes

// Compile-time defines set by host:
// DISPATCH_CORE_NOC_X  - NOC X coordinate of dispatch_s core
// DISPATCH_CORE_NOC_Y  - NOC Y coordinate of dispatch_s core
// DISPATCH_DATA_ADDR_A - Address of kernel_start_a in dispatch_s's L1 mailbox
// DISPATCH_DATA_ADDR_B - Address of kernel_start_b in dispatch_s's L1 mailbox

// Pointer to perf telemetry config in mailbox (for reading config_buffer_addr)
volatile tt_l1_ptr perf_telemetry_msg_t* perf_telemetry_mailbox =
    reinterpret_cast<volatile tt_l1_ptr perf_telemetry_msg_t*>(GET_MAILBOX_ADDRESS_DEV(perf_telemetry));

// Telemetry socket state
static SocketSenderInterface perf_telemetry_socket;
static bool perf_telemetry_initialized = false;
static uint32_t perf_telemetry_pcie_xy_enc = 0;
static uint32_t perf_telemetry_data_addr_hi = 0;
static uint32_t perf_telemetry_host_write_ptr = 0;
static uint32_t perf_telemetry_host_fifo_start = 0;
static uint32_t perf_telemetry_fifo_page_aligned_size = 0;
static uint32_t perf_telemetry_l1_data_addr = 0;

// Initialize the perf telemetry socket interface
// Returns: true if initialized successfully, false if config not available
FORCE_INLINE
bool perf_telemetry_init() {
    if (perf_telemetry_initialized) {
        return true;
    }

    // Invalidate L1 cache to see host writes
    invalidate_l1_cache();

    // Read config buffer address from mailbox
    uint32_t socket_config_addr = perf_telemetry_mailbox->config_buffer_addr;
    if (socket_config_addr == 0) {
        // Config not set by host yet
        return false;
    }

    // Create socket interface from config buffer
    perf_telemetry_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(perf_telemetry_socket, perf_telemetry_page_size);

    // Read PCIe-specific config from the config buffer
    // Layout: 8 words MD + 4 words ack + downstream encoding area
    // [12] = pcie_xy_enc, [13] = data_addr_hi, [14] = bytes_sent_addr_hi
    // [15] = l1_data_buffer_address, [16] = l1_data_buffer_size
    // [2] = data_addr_lo (write_ptr)
    tt_l1_ptr uint32_t* socket_config_words = reinterpret_cast<tt_l1_ptr uint32_t*>(socket_config_addr);
    perf_telemetry_pcie_xy_enc = socket_config_words[12];
    perf_telemetry_data_addr_hi = socket_config_words[13];
    perf_telemetry_l1_data_addr = socket_config_words[15];
    uint32_t data_addr_lo = socket_config_words[2];  // Initial write_ptr = data buffer start

    // Calculate page-aligned FIFO size
    perf_telemetry_fifo_page_aligned_size =
        perf_telemetry_socket.downstream_fifo_total_size -
        (perf_telemetry_socket.downstream_fifo_total_size % perf_telemetry_page_size);

    // Track write pointer in host buffer
    perf_telemetry_host_write_ptr = data_addr_lo;
    perf_telemetry_host_fifo_start = data_addr_lo;

    perf_telemetry_initialized = true;
    return true;
}

// Push one page of telemetry data from L1 buffer to host via D2H socket
// Uses Brisc NOC 0 command buffer 0
// buffer_a: true to read from buffer A, false to read from buffer B
// Returns: true if data was sent, false otherwise
__attribute__((noinline)) bool perf_telemetry_push(bool buffer_a) {
    // Try to init if not already initialized
    if (!perf_telemetry_init()) {
        return false;
    }

    // Check if L1 data buffer is available
    if (perf_telemetry_l1_data_addr == 0) {
        return false;
    }

    if (buffer_a) {
        DeviceZoneScopedN("A");
    } else {
        DeviceZoneScopedN("B");
    }

    // Read 32 bytes from dispatch core's kernel_start_a or kernel_start_b into socket's L1 data buffer
    uint32_t dispatch_data_addr = buffer_a ? DISPATCH_DATA_ADDR_A : DISPATCH_DATA_ADDR_B;
    uint64_t dispatch_noc_addr = get_noc_addr(DISPATCH_CORE_NOC_X, DISPATCH_CORE_NOC_Y, dispatch_data_addr);
    noc_async_read(dispatch_noc_addr, perf_telemetry_l1_data_addr, telemetry_timestamp_size);
    noc_async_read_barrier();

    // Initialize NOC for PCIe writes (using command buffer 0)
    noc_write_init_state<0>(NOC_0, NOC_UNICAST_WRITE_VC);

    // Wait for space in the receiver's FIFO
    socket_reserve_pages(perf_telemetry_socket, 1);

    // Build 64-bit PCIe destination address
    uint64_t pcie_dest_addr = (static_cast<uint64_t>(perf_telemetry_data_addr_hi) << 32) |
                              static_cast<uint64_t>(perf_telemetry_host_write_ptr);

    // Write data from socket's L1 data buffer to PCIe-mapped host memory
    // Sends 64 bytes: kernel_start (16B) + kernel_end (16B) + 32B padding
    noc_wwrite_with_state<DM_DEDICATED_NOC, 0, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_0, perf_telemetry_l1_data_addr, perf_telemetry_pcie_xy_enc, pcie_dest_addr, perf_telemetry_page_size, 1);

    // Update host write pointer with wrap-around
    perf_telemetry_host_write_ptr += perf_telemetry_page_size;
    if (perf_telemetry_host_write_ptr >= perf_telemetry_host_fifo_start + perf_telemetry_fifo_page_aligned_size) {
        perf_telemetry_host_write_ptr = perf_telemetry_host_fifo_start;
    }

    // Update socket state
    socket_push_pages(perf_telemetry_socket, 1);

    // Notify host via PCIe
    pcie_socket_notify_receiver(perf_telemetry_socket);
    // Barrier to ensure PCIe write is visible to host
    noc_async_write_barrier();

    return true;
}

// Push sync response - writes directly to L1 buffer (no NOC read from dispatch)
__attribute__((noinline)) bool perf_telemetry_sync_push(uint32_t time_hi, uint32_t time_lo, uint32_t host_time) {
    if (!perf_telemetry_init()) {
        return false;
    }

    if (perf_telemetry_l1_data_addr == 0) {
        return false;
    }

    // Write sync data directly to L1 data buffer
    // Layout: time_hi, time_lo, id (echoed host_time), header (sync marker)
    tt_l1_ptr uint32_t* l1_data = reinterpret_cast<tt_l1_ptr uint32_t*>(perf_telemetry_l1_data_addr);
    l1_data[0] = time_hi;
    l1_data[1] = time_lo;
    l1_data[2] = host_time;
    l1_data[3] = TELEMETRY_SYNC_MARKER_ID;
    // Zero out the rest of the page (kernel_end fields)
    l1_data[4] = 0;
    l1_data[5] = 0;
    l1_data[6] = 0;
    l1_data[7] = 0;

    // Initialize NOC for PCIe writes
    noc_write_init_state<0>(NOC_0, NOC_UNICAST_WRITE_VC);

    // Wait for space in the receiver's FIFO
    socket_reserve_pages(perf_telemetry_socket, 1);

    // Build 64-bit PCIe destination address
    uint64_t pcie_dest_addr = (static_cast<uint64_t>(perf_telemetry_data_addr_hi) << 32) |
                              static_cast<uint64_t>(perf_telemetry_host_write_ptr);

    // Write data from L1 buffer to host via PCIe
    noc_wwrite_with_state<DM_DEDICATED_NOC, 0, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_0, perf_telemetry_l1_data_addr, perf_telemetry_pcie_xy_enc, pcie_dest_addr, perf_telemetry_page_size, 1);

    // Update host write pointer with wrap-around
    perf_telemetry_host_write_ptr += perf_telemetry_page_size;
    if (perf_telemetry_host_write_ptr >= perf_telemetry_host_fifo_start + perf_telemetry_fifo_page_aligned_size) {
        perf_telemetry_host_write_ptr = perf_telemetry_host_fifo_start;
    }

    // Update socket state
    socket_push_pages(perf_telemetry_socket, 1);

    // Notify host via PCIe
    pcie_socket_notify_receiver(perf_telemetry_socket);
    noc_async_write_barrier();

    return true;
}

// Handle sync requests from host
// Polls for host timestamp, captures device timestamp, and pushes response
// Stays in sync mode until sync_request is cleared by host
__attribute__((noinline)) void perf_telemetry_sync() {
    DPRINT << "TELEM: entering sync" << ENDL();

    // Try to init if not already initialized
    if (!perf_telemetry_init()) {
        DPRINT << "TELEM: sync init failed" << ENDL();
        return;
    }

    DPRINT << "TELEM: sync init ok" << ENDL();

    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    uint32_t sync_count = 0;
    // Stay in sync mode while sync_request is set
    while (perf_telemetry_mailbox->sync_request) {
        invalidate_l1_cache();

        uint32_t host_time = perf_telemetry_mailbox->sync_host_timestamp;
        if (host_time > 0) {
            DPRINT << "TELEM: sync got host_time=" << host_time << ENDL();

            // Capture device wall clock immediately
            uint32_t time_lo = p_reg[WALL_CLOCK_LOW_INDEX];
            uint32_t time_hi = p_reg[WALL_CLOCK_HIGH_INDEX];

            // Push sync response directly (no NOC read from dispatch)
            perf_telemetry_sync_push(time_hi, time_lo, host_time);

            // Clear host timestamp - signal we're ready for next
            perf_telemetry_mailbox->sync_host_timestamp = 0;
            sync_count++;
            DPRINT << "TELEM: sync pushed count=" << sync_count << ENDL();
        }
    }
    DPRINT << "TELEM: exiting sync, total=" << sync_count << ENDL();
}

void kernel_main() {
    DPRINT << "TELEM: kernel started, sync_req_addr=" << (uint32_t)&perf_telemetry_mailbox->sync_request << ENDL();

    // Initialize to idle state - wait for external trigger to push
    perf_telemetry_mailbox->telemetry_state = TELEMETRY_STATE_IDLE;

    uint32_t loop_count = 0;

    // Main telemetry loop - service different states
    while (true) {
        // Invalidate L1 cache to see host writes
        invalidate_l1_cache();

        // Print sync_request value periodically
        if ((loop_count++ & 0x7FFFF) == 0) {
            DPRINT << "TELEM: loop sync_req=" << perf_telemetry_mailbox->sync_request
                   << " sync_ts=" << perf_telemetry_mailbox->sync_host_timestamp << ENDL();
        }

        TelemetryState state = static_cast<TelemetryState>(perf_telemetry_mailbox->telemetry_state);

        switch (state) {
            case TELEMETRY_STATE_IDLE:
                // When idle, check for sync request from host
                if (perf_telemetry_mailbox->sync_request) {
                    DPRINT << "TELEM: sync_request detected!" << ENDL();
                    perf_telemetry_sync();
                }
                continue;

            case TELEMETRY_STATE_PUSH_A:
                // Push telemetry data from buffer A, then go back to idle
                perf_telemetry_push(true);
                perf_telemetry_mailbox->telemetry_state = TELEMETRY_STATE_IDLE;
                break;

            case TELEMETRY_STATE_PUSH_B:
                // Push telemetry data from buffer B, then go back to idle
                perf_telemetry_push(false);
                perf_telemetry_mailbox->telemetry_state = TELEMETRY_STATE_IDLE;
                break;

            case TELEMETRY_STATE_TERMINATE:
                // Exit the kernel
                return;
        }
    }
}
