// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"
#include "tt_fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

// clang-format on

using namespace tt::tt_fabric;

uint32_t src_endpoint_id;
// constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);
static_assert(is_power_of_2(num_dest_endpoints), "num_dest_endpoints must be a power of 2");
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(2);

constexpr uint32_t data_buffer_start_addr = get_compile_time_arg_val(3);
constexpr uint32_t data_buffer_size_words = get_compile_time_arg_val(4);

constexpr uint32_t routing_table_start_addr = get_compile_time_arg_val(5);

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(6);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(7);

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t prng_seed = get_compile_time_arg_val(8);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(9);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(10);

static_assert(max_packet_size_words > 3, "max_packet_size_words must be greater than 3");

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(11);

constexpr bool skip_pkt_content_gen = get_compile_time_arg_val(12);
constexpr pkt_dest_size_choices_t pkt_dest_size_choice =
    static_cast<pkt_dest_size_choices_t>(get_compile_time_arg_val(13));

constexpr uint32_t data_sent_per_iter_low = get_compile_time_arg_val(14);
constexpr uint32_t data_sent_per_iter_high = get_compile_time_arg_val(15);
constexpr uint32_t test_command = get_compile_time_arg_val(16);

uint32_t base_target_address = get_compile_time_arg_val(17);

// atomic increment for the ATOMIC_INC command
constexpr uint32_t atomic_increment = get_compile_time_arg_val(18);
// constexpr uint32_t dest_device = get_compile_time_arg_val(21);
uint32_t dest_device;

constexpr uint32_t signal_address = get_compile_time_arg_val(19);
constexpr uint32_t client_interface_addr = get_compile_time_arg_val(20);

volatile local_pull_request_t* local_pull_request = (volatile local_pull_request_t*)(data_buffer_start_addr - 1024);
volatile tt_l1_ptr fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(routing_table_start_addr);
volatile fabric_client_interface_t* client_interface = (volatile fabric_client_interface_t*)client_interface_addr;

uint64_t xy_local_addr;
uint32_t target_address;
uint32_t noc_offset;
uint32_t gk_interface_addr_l;
uint32_t gk_interface_addr_h;

uint32_t time_seed;

void kernel_main() {
    tt_fabric_init();

    uint32_t rt_args_idx = 0;
    time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    src_endpoint_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t router_x = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t router_y = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    dest_device = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t rx_buf_size = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    gk_interface_addr_l = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    gk_interface_addr_h = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    target_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    // Read in the routing table
    uint64_t router_config_addr =
        NOC_XY_ADDR(NOC_X(router_x), NOC_Y(router_y), eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
    noc_async_read_one_packet(router_config_addr, routing_table_start_addr, sizeof(fabric_router_l1_config_t));
    noc_async_read_barrier();

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_STATUS_INDEX + 1] = (uint32_t)local_pull_request;

    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;
    test_results[PQ_TEST_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(data_buffer_start_addr), data_buffer_size_words * PACKET_WORD_SIZE_BYTES);
    zero_l1_buf((uint32_t*)local_pull_request, sizeof(local_pull_request_t));
    zero_l1_buf((uint32_t*)client_interface, sizeof(fabric_client_interface_t));
    client_interface->gk_interface_addr = ((uint64_t)gk_interface_addr_h << 32) | gk_interface_addr_l;
    client_interface->gk_msg_buf_addr =
        (((uint64_t)gk_interface_addr_h << 32) | gk_interface_addr_l) + offsetof(gatekeeper_info_t, gk_msg_buf);

    uint64_t data_words_sent = 0;
    uint32_t packet_count = 0;

    uint64_t dst_addr = ((uint64_t)noc_offset << 32 | target_address);

    fabric_async_write_add_header(
        data_buffer_start_addr,  // source address in sender’s memory
        dest_device >> 16,
        dest_device & 0xFFFF,
        dst_addr,                   // destination write address
        max_packet_size_words * 16  // number of bytes to write to remote destination
    );

    // make sure fabric node gatekeeper is available.
    fabric_endpoint_init();

    // wait till test sends start signal. This is set by test
    // once tt_fabric kernels have been launched on all the test devices.
    while (*(volatile tt_l1_ptr uint32_t*)signal_address == 0);

    uint64_t start_timestamp = get_timestamp();
    fabric_setup_pull_request(
        data_buffer_start_addr,     // source address in sender’s memory
        max_packet_size_words * 16  // number of bytes to write to remote destination
    );

    while (true) {
        client_interface->local_pull_request.pull_request.rd_ptr = 0;
        fabric_async_write<ASYNC_WR_SEND>(
            0,                       // the network plane to use for this transaction
            data_buffer_start_addr,  // source address in sender’s memory
            dest_device >> 16,
            dest_device & 0xFFFF,
            dst_addr,                   // destination write address
            max_packet_size_words * 16  // number of bytes to write to remote destination
        );
        data_words_sent += max_packet_size_words;
        packet_count++;
        uint32_t wr_ptr = client_interface->local_pull_request.pull_request.wr_ptr;
        while (client_interface->local_pull_request.pull_request.rd_ptr != wr_ptr) {
#pragma GCC unroll 4
            for (int i = 0; i < 4; i++) {
                asm("nop");
            }
        }
        if (data_words_sent >= total_data_words) {
            break;
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    uint64_t num_packets = packet_count;
    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, total_data_words, TX_TEST_IDX_TOT_DATA_WORDS);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);

    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_PASS;
    test_results[PQ_TEST_MISC_INDEX] = packet_count;
}
