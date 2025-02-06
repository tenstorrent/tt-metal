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
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(2);

constexpr uint32_t data_buffer_start_addr = get_compile_time_arg_val(3);
constexpr uint32_t data_buffer_size_words = get_compile_time_arg_val(4);

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

uint32_t dest_device;

constexpr uint32_t signal_address = get_compile_time_arg_val(19);
constexpr uint32_t client_interface_addr = get_compile_time_arg_val(20);

constexpr bool mcast_data = get_compile_time_arg_val(23);
constexpr uint32_t e_depth = get_compile_time_arg_val(24);
constexpr uint32_t w_depth = get_compile_time_arg_val(25);
constexpr uint32_t n_depth = get_compile_time_arg_val(26);
constexpr uint32_t s_depth = get_compile_time_arg_val(27);

volatile fabric_client_interface_t* client_interface;

fabric_client_push_interface_t client_push_interface;
uint64_t xy_local_addr;
uint32_t target_address;
uint32_t noc_offset;
uint32_t gk_interface_addr_l;
uint32_t gk_interface_addr_h;
uint32_t controller_noc_offset;
uint32_t time_seed;

inline void notify_traffic_controller() {
    // send semaphore increment to traffic controller kernel on this device.
    uint64_t dest_addr = get_noc_addr_helper(controller_noc_offset, signal_address);
    noc_fast_atomic_increment<DM_DYNAMIC_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        dest_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        31,
        false,
        false,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);
}

void kernel_main() {
    uint32_t rt_args_idx = 0;
    time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    src_endpoint_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    controller_noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t routing_plane = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    dest_device = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t rx_buf_size = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    gk_interface_addr_l = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    gk_interface_addr_h = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    if constexpr (ASYNC_WR & test_command) {
        base_target_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    }

    target_address = base_target_address;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000000;
    test_results[TT_FABRIC_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(data_buffer_start_addr), data_buffer_size_words * PACKET_WORD_SIZE_BYTES);

    uint64_t data_words_sent = 0;
    uint32_t packet_count = 0;

    uint64_t dst_addr = ((uint64_t)noc_offset << 32 | target_address);
    if constexpr (mcast_data) {
        fabric_async_write_multicast_add_header(
            data_buffer_start_addr,  // source address in sender’s memory
            dest_device >> 16,
            dest_device & 0xFFFF,
            dst_addr,                    // destination write address
            max_packet_size_words * 16,  // number of bytes to write to remote destination
            e_depth,
            w_depth,
            n_depth,
            s_depth);
    } else {
        fabric_async_write_add_header(
            data_buffer_start_addr,  // source address in sender’s memory
            dest_device >> 16,
            dest_device & 0xFFFF,
            dst_addr,                   // destination write address
            max_packet_size_words * 16  // number of bytes to write to remote destination
        );
    }

    // initalize client
    fabric_endpoint_init(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    // notify the controller kernel that this worker is ready to proceed
    notify_traffic_controller();

    // wait till test sends start signal. This is set by test
    // once tt_fabric kernels have been launched on all the test devices and
    // all tx workers are ready to send data
    while (*(volatile tt_l1_ptr uint32_t*)signal_address == 0);

#ifdef FVC_MODE_PULL
    uint64_t start_timestamp = get_timestamp();
    fabric_setup_pull_request(
        data_buffer_start_addr,     // source address in sender’s memory
        max_packet_size_words * 16  // number of bytes to write to remote destination
    );

    uint64_t start_timestamp = get_timestamp();

        if constexpr (mcast_data) {
            fabric_async_write_multicast<ASYNC_WR_SEND>(
                data_buffer_start_addr,  // source address in sender’s memory
                dest_device >> 16,
                dest_device & 0xFFFF,
                dst_addr,                    // destination write address
                max_packet_size_words * 16,  // number of bytes to write to remote destination
                e_depth,
                w_depth,
                n_depth,
                s_depth);
        } else {
            fabric_async_write<ASYNC_WR_SEND>(
                routing_plane,           // the network plane to use for this transaction
                data_buffer_start_addr,  // source address in sender’s memory
                dest_device >> 16,
                dest_device & 0xFFFF,
                dst_addr,                   // destination write address
                max_packet_size_words * 16  // number of bytes to write to remote destination
            );
        }

        data_words_sent += max_packet_size_words;
        packet_count++;
        uint32_t words_written = client_interface->local_pull_request.pull_request.words_written;
        while (client_interface->local_pull_request.pull_request.words_read != words_written) {
#pragma GCC unroll 4
            for (int i = 0; i < 4; i++) {
                asm("nop");
            }
        }
        if (data_words_sent >= total_data_words) {
            break;
        }
    }
#else
    fabric_client_router_reserve(0, dest_device >> 16, dest_device & 0xFFFF);
    uint64_t start_timestamp = get_timestamp();

    while (true) {
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
        // noc_async_writes_flushed();

        if (data_words_sent >= total_data_words) {
            break;
        }
    }
#endif
    noc_async_writes_flushed();

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    uint64_t num_packets = packet_count;
    set_64b_result(test_results, data_words_sent, TT_FABRIC_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, TT_FABRIC_CYCLES_INDEX);
    set_64b_result(test_results, total_data_words, TX_TEST_IDX_TOT_DATA_WORDS);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_MISC_INDEX] = packet_count;
}
