// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Telemetry kernel
// Pushes telemetry data from L1 buffer to host via D2H socket
// Uses Brisc NOC 0 command buffer 0 for PCIe writes

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/dev_msgs.h"

// Perf telemetry page size - must match host-side kPerfTelemetryPageSize
constexpr uint32_t perf_telemetry_page_size = 64;

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
// Returns: true if data was sent, false otherwise
__attribute__((noinline)) bool perf_telemetry_push() {
    // Try to init if not already initialized
    if (!perf_telemetry_init()) {
        return false;
    }

    // Check if L1 data buffer is available
    if (perf_telemetry_l1_data_addr == 0) {
        return false;
    }

    // Initialize NOC for PCIe writes (using command buffer 0)
    noc_write_init_state<0>(NOC_0, NOC_UNICAST_WRITE_VC);

    // Wait for space in the receiver's FIFO
    socket_reserve_pages(perf_telemetry_socket, 1);

    // Build 64-bit PCIe destination address
    uint64_t pcie_dest_addr = (static_cast<uint64_t>(perf_telemetry_data_addr_hi) << 32) |
                              static_cast<uint64_t>(perf_telemetry_host_write_ptr);

    // Write data to PCIe-mapped host memory using PCIe write primitive
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

void kernel_main() {
    // Initialize to idle state - wait for external trigger to push
    perf_telemetry_mailbox->telemetry_state = TELEMETRY_STATE_IDLE;

    // Main telemetry loop - service different states
    while (true) {
        TelemetryState state = static_cast<TelemetryState>(perf_telemetry_mailbox->telemetry_state);

        switch (state) {
            case TELEMETRY_STATE_IDLE:
                // Wait for initialization - skip this iteration
                continue;

            case TELEMETRY_STATE_PUSH:
                // Push telemetry data, then go back to idle
                perf_telemetry_push();
                perf_telemetry_mailbox->telemetry_state = TELEMETRY_STATE_IDLE;
                break;

            case TELEMETRY_STATE_TERMINATE:
                // Exit the kernel
                return;
        }
    }
}
