// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// On-device loopback kernel for PipelineManager testing.
// Reads InjectDescriptor pages from H2D socket, produces ResultDescriptor pages,
// writes them to D2H socket. Mimics MockPipeline behavior:
//   PREFILL → non-sampled result (actual_token = -1)
//   DECODE  → sampled result (actual_token = token_id + 1)
// Exits when user_id == -1 (sentinel), echoing it back.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

// Wire format word indices — must match wire_format.hpp on the host.
// InjectPage (H2D): [0]=user_id [1]=token_id [2]=position [3]=mode [4]=spec_flag [5-7]=sampling params [8-15]=reserved
// ResultPage (D2H): [0]=user_id [1]=actual_token [2]=predicted_token [3]=mode [4]=position [5]=spec_flag [6]=sampled
// [7-15]=reserved

static constexpr uint32_t MODE_PREFILL = 0;
static constexpr uint32_t MODE_DECODE = 1;
static constexpr uint32_t SENTINEL_USER_ID = 0xFFFFFFFF;  // -1 as uint32
static constexpr uint32_t EMPTY_TOKEN = 0xFFFFFFFF;       // -1 as uint32
static constexpr uint32_t PAGE_SIZE_WORDS = 16;

void kernel_main() {
    constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* output_cb_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(output_cb_index));

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(recv_socket_config_addr);
    SocketSenderInterface sender_socket = create_sender_socket_interface(sender_socket_config_addr);

    set_receiver_socket_page_size(receiver_socket, page_size);
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    while (true) {
        // Read one inject page from H2D
        socket_wait_for_pages(receiver_socket, 1);

        // The page data is now at receiver_socket.read_ptr in L1.
        // Read the inject fields directly from L1.
        volatile tt_l1_ptr uint32_t* in_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr);

        uint32_t user_id = in_page[0];
        uint32_t token_id = in_page[1];
        uint32_t position = in_page[2];
        uint32_t mode = in_page[3];
        uint32_t spec_flag = in_page[4];

        // Reserve space in D2H socket for the result page
        socket_reserve_pages(sender_socket, 1);

        // Zero the output page
        for (uint32_t w = 0; w < PAGE_SIZE_WORDS; w++) {
            output_cb_addr[w] = 0;
        }

        output_cb_addr[0] = user_id;
        output_cb_addr[3] = mode;
        output_cb_addr[4] = position;
        output_cb_addr[5] = spec_flag;

        if (mode == MODE_DECODE) {
            output_cb_addr[1] = token_id + 1;  // actual_token = token_id + 1
            output_cb_addr[2] = EMPTY_TOKEN;   // predicted_token = -1
            output_cb_addr[6] = 1;             // sampled = true
        } else {
            output_cb_addr[1] = EMPTY_TOKEN;  // actual_token = -1
            output_cb_addr[2] = EMPTY_TOKEN;  // predicted_token = -1
            output_cb_addr[6] = 0;            // sampled = false
        }

        // Write result page to D2H socket over PCIe
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX,
            get_write_ptr(output_cb_index),
            pcie_xy_enc,
            ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                sender_socket.write_ptr,
            page_size,
            1);

        socket_push_pages(sender_socket, 1);
        socket_notify_receiver(sender_socket);
        socket_pop_pages(receiver_socket, 1);
        noc_async_writes_flushed();
        socket_notify_sender(receiver_socket);

        // Sentinel check: exit after echoing the sentinel back
        if (user_id == SENTINEL_USER_ID) {
            break;
        }
    }

    update_socket_config(receiver_socket);
    update_socket_config(sender_socket);
    socket_barrier(sender_socket);

    noc_async_write_barrier();
    noc_async_read_barrier();
}
