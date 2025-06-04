// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "debug/dprint.h"
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
// clang-format on

using namespace tt::tt_fabric;

// seed to re-generate the data and validate against incoming data
constexpr uint32_t prng_seed = get_compile_time_arg_val(0);

// total data/payload expected
constexpr uint32_t total_data_kb = get_compile_time_arg_val(1);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

// max packet size to generate mask
constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(2);
static_assert(max_packet_size_words > 3, "max_packet_size_words must be greater than 3");

// fabric command
constexpr uint32_t test_command = get_compile_time_arg_val(3);

// address to start reading from/poll on
constexpr uint32_t target_address = get_compile_time_arg_val(4);

// atomic increment for the ATOMIC_INC command
constexpr uint32_t atomic_increment = get_compile_time_arg_val(5);

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(6);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(7);
constexpr uint32_t gk_interface_addr_l = get_compile_time_arg_val(8);
constexpr uint32_t gk_interface_addr_h = get_compile_time_arg_val(9);
constexpr uint32_t client_interface_addr = get_compile_time_arg_val(10);
constexpr uint32_t client_pull_req_buf_addr = get_compile_time_arg_val(11);
constexpr uint32_t data_buffer_start_addr = get_compile_time_arg_val(12);
constexpr uint32_t data_buffer_size_words = get_compile_time_arg_val(13);

volatile tt_l1_ptr chan_req_buf* client_pull_req_buf =
    reinterpret_cast<tt_l1_ptr chan_req_buf*>(client_pull_req_buf_addr);
volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface =
    (volatile tt_l1_ptr fabric_pull_client_interface_t*)client_interface_addr;
uint64_t xy_local_addr;
socket_reader_state socket_reader;

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

#define PAYLOAD_MASK (0xFFFF0000)

void kernel_main() {
    uint64_t processed_packet_words = 0, num_packets = 0;
    volatile tt_l1_ptr uint32_t* poll_addr;
    uint32_t poll_val = 0;
    bool async_wr_check_failed = false;

    // parse runtime args
    uint32_t dest_device = get_arg_val<uint32_t>(0);

    tt_fabric_init();

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000000;
    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(data_buffer_start_addr), data_buffer_size_words * PACKET_WORD_SIZE_BYTES);
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000001;
    zero_l1_buf((uint32_t*)client_interface, sizeof(fabric_pull_client_interface_t));
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000002;
    zero_l1_buf((uint32_t*)client_pull_req_buf, sizeof(chan_req_buf));
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000003;

    client_interface->gk_interface_addr = get_noc_addr_helper(gk_interface_addr_h, gk_interface_addr_l);
    client_interface->gk_msg_buf_addr = client_interface->gk_interface_addr + offsetof(gatekeeper_info_t, gk_msg_buf);
    client_interface->pull_req_buf_addr = xy_local_addr | client_pull_req_buf_addr;
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000004;

    // make sure fabric node gatekeeper is available.
    tt_fabric_init();
    fabric_endpoint_init<RoutingType::ROUTING_TABLE>();

    socket_reader.init(data_buffer_start_addr, data_buffer_size_words);
    DPRINT << "Socket open on  " << dest_device << ENDL();
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000005;

    fabric_socket_open(
        client_interface,       // fabric client interface
        3,                      // the network plane to use for this socket
        2,                      // Temporal epoch for which the socket is being opened
        1,                      // Socket Id to open
        SOCKET_TYPE_DGRAM,      // Unicast, Multicast, SSocket, DSocket
        SOCKET_DIRECTION_RECV,  // Send or Receive
        dest_device >> 16,      // Remote mesh/device that is the socket data sender/receiver.
        dest_device & 0xFFFF,
        0  // fabric virtual channel.
    );
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000006;

    uint32_t loop_count = 0;
    uint32_t packet_count = 0;
    while (1) {
        if (!fvc_req_buf_is_empty(client_pull_req_buf) && fvc_req_valid(client_pull_req_buf)) {
            uint32_t req_index = client_pull_req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
            chan_request_entry_t* req = (chan_request_entry_t*)client_pull_req_buf->chan_req + req_index;
            pull_request_t* pull_req = &req->pull_request;
            if (socket_reader.packet_in_progress == 0) {
                DPRINT << "Socket Packet " << packet_count << ENDL();
            }
            if (pull_req->flags == FORWARD) {
                socket_reader.pull_socket_data(pull_req);
                test_results[TT_FABRIC_MISC_INDEX] = 0xDD000001;
                noc_async_read_barrier();
                update_pull_request_words_cleared(pull_req);
                socket_reader.pull_words_in_flight = 0;
                socket_reader.push_socket_data<false>();
            }

            if (socket_reader.packet_in_progress == 1 and socket_reader.packet_words_remaining == 0) {
                // wait for any pending sockat data writes to finish.
                test_results[TT_FABRIC_MISC_INDEX] = 0xDD000002;

                noc_async_write_barrier();

                test_results[TT_FABRIC_MISC_INDEX] = 0xDD000003;
                // clear the flags field to invalidate pull request slot.
                // flags will be set to non-zero by next requestor.
                req_buf_advance_rdptr((chan_req_buf*)client_pull_req_buf);
                socket_reader.packet_in_progress = 0;
                packet_count++;
                loop_count = 0;
            }
        }
        test_results[TT_FABRIC_MISC_INDEX] = 0xDD400000 | (loop_count & 0xfffff);

        loop_count++;
        if (packet_count > 0 and loop_count >= 0x10000) {
            DPRINT << "Socket Rx Finished" << packet_count << ENDL();
            break;
        }
    }

    // write out results
    set_64b_result(test_results, processed_packet_words, TT_FABRIC_WORD_CNT_INDEX);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);

    if (async_wr_check_failed) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
        test_results[TT_FABRIC_MISC_INDEX] = 0xff000005;
    }
}
