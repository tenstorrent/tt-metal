// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
// clang-format on

constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);
static_assert(is_power_of_2(num_dest_endpoints), "num_dest_endpoints must be a power of 2");
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(2);

constexpr uint32_t data_buffer_start_addr = get_compile_time_arg_val(3);
constexpr uint32_t data_buffer_size_words = get_compile_time_arg_val(4);

constexpr uint32_t routing_table_start_addr = get_compile_time_arg_val(5);
constexpr uint32_t router_x = get_compile_time_arg_val(6);
constexpr uint32_t router_y = get_compile_time_arg_val(7);

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(8);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(9);

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t prng_seed = get_compile_time_arg_val(10);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(11);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(12);

static_assert(max_packet_size_words > 3, "max_packet_size_words must be greater than 3");

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(13);

constexpr bool skip_pkt_content_gen = get_compile_time_arg_val(14);
constexpr pkt_dest_size_choices_t pkt_dest_size_choice =
    static_cast<pkt_dest_size_choices_t>(get_compile_time_arg_val(15));

constexpr uint32_t data_sent_per_iter_low = get_compile_time_arg_val(16);
constexpr uint32_t data_sent_per_iter_high = get_compile_time_arg_val(17);
constexpr uint32_t test_command = get_compile_time_arg_val(18);

constexpr uint32_t base_target_address = get_compile_time_arg_val(19);
uint32_t target_address = base_target_address;

// atomic increment for the ATOMIC_INC command
constexpr uint32_t atomic_increment = get_compile_time_arg_val(20);
constexpr uint32_t dest_device = get_compile_time_arg_val(21);

constexpr uint32_t base_target_address = get_compile_time_arg_val(19);
uint32_t target_address = base_target_address;

// atomic increment for the ATOMIC_INC command
constexpr uint32_t atomic_increment = get_compile_time_arg_val(20);

uint32_t max_packet_size_mask;

auto input_queue_state = select_input_queue<pkt_dest_size_choice>();
volatile local_pull_request_t *local_pull_request = (volatile local_pull_request_t *)(data_buffer_start_addr - 1024);
tt_l1_ptr volatile tt::tt_fabric::fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr tt::tt_fabric::fabric_router_l1_config_t*>(routing_table_start_addr);

fvc_producer_state_t test_producer __attribute__((aligned(16)));

uint64_t xy_local_addr;

packet_header_t packet_header __attribute__((aligned(16)));

// generates packets with random size and payload on the input side
inline bool test_buffer_handler_async_wr() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_words = test_producer.get_num_words_free();
    if (free_words < PACKET_HEADER_SIZE_WORDS) {
        return false;
    }

    // Each call to test_buffer_handler initializes only up to the end
    // of the producer buffer. Since the header is 3 words, we need to handle
    // split header cases, where the buffer has not enough space for the full header.
    // In this case, we write as many words as space available, and remaining header words
    // are written on next call.
    uint32_t byte_wr_addr = test_producer.get_local_buffer_write_addr();
    uint32_t words_to_init = std::min(free_words, test_producer.words_before_local_buffer_wrap());
    uint32_t words_initialized = 0;
    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) { // start of a new packet
            input_queue_state.next_packet(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);

            tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);

            packet_header.routing.flags = FORWARD;
            packet_header.routing.packet_size_bytes = input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
            packet_header.routing.dst_mesh_id = dest_device >> 16;
            packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
            packet_header.session.command = ASYNC_WR;
            packet_header.session.target_offset_l = target_address;
            packet_header.session.target_offset_h = 0x410;
            target_address += packet_header.routing.packet_size_bytes - PACKET_HEADER_SIZE_BYTES;
            packet_header.packet_parameters.misc_parameters.words[1] = input_queue_state.packet_rnd_seed;
            tt_fabric_add_header_checksum(&packet_header);
            uint32_t words_left = words_to_init - words_initialized;
            bool split_header = words_left < PACKET_HEADER_SIZE_WORDS;
            uint32_t header_words_to_init = PACKET_HEADER_SIZE_WORDS;
            if (split_header) {
                header_words_to_init = words_left;
            }

            for (uint32_t i = 0; i < (header_words_to_init * PACKET_WORD_SIZE_BYTES / 4); i++) {
                header_ptr[i] = ((uint32_t *)&packet_header)[i];
            }

            words_initialized += header_words_to_init;
            input_queue_state.curr_packet_words_remaining -= header_words_to_init;
            byte_wr_addr += header_words_to_init * PACKET_WORD_SIZE_BYTES;
        } else {
            uint32_t words_remaining = words_to_init - words_initialized;
            uint32_t packet_words_initialized = input_queue_state.curr_packet_size_words - input_queue_state.curr_packet_words_remaining;
            if (packet_words_initialized < PACKET_HEADER_SIZE_WORDS) {
                tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
                uint32_t header_words_initialized = packet_words_initialized;
                uint32_t header_words_to_init = PACKET_HEADER_SIZE_WORDS - header_words_initialized;
                uint32_t header_dword_index = header_words_initialized * PACKET_WORD_SIZE_BYTES / 4;
                header_words_to_init = std::min(words_remaining, header_words_to_init);
                for (uint32_t i = 0; i < (header_words_to_init * PACKET_WORD_SIZE_BYTES / 4); i++) {
                    header_ptr[i] = ((uint32_t *)&packet_header)[i + header_dword_index];
                }
                words_initialized += header_words_to_init;
                words_remaining = words_to_init - words_initialized;
                input_queue_state.curr_packet_words_remaining -= header_words_to_init;
                byte_wr_addr += header_words_to_init * PACKET_WORD_SIZE_BYTES;
                if (words_remaining == 0) {
                    // no space left for packet data.
                    break;
                }
            }

            uint32_t num_words = std::min(words_remaining, input_queue_state.curr_packet_words_remaining);
            if constexpr (!skip_pkt_content_gen) {
                uint32_t start_val =
                (input_queue_state.packet_rnd_seed & 0xFFFF0000) +
                (input_queue_state.curr_packet_size_words - input_queue_state.curr_packet_words_remaining - PACKET_HEADER_SIZE_WORDS);
                fill_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr), num_words, start_val);
            }
            words_initialized += num_words;
            input_queue_state.curr_packet_words_remaining -= num_words;
            byte_wr_addr += num_words * PACKET_WORD_SIZE_BYTES;
        }
    }
    test_producer.advance_local_wrptr(words_initialized);
    return false;
}

// generates packets with random size and payload on the input side
inline bool test_buffer_handler_atomic_inc() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_words = test_producer.get_num_words_free();
    if (free_words < PACKET_HEADER_SIZE_WORDS) {
        return false;
    }

    uint32_t byte_wr_addr = test_producer.get_local_buffer_write_addr();
    uint32_t words_to_init = std::min(free_words, test_producer.words_before_local_buffer_wrap());
    uint32_t words_initialized = 0;
    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) {  // start of a new packet
            input_queue_state.next_inline_packet(total_data_words);

            tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);

            packet_header.routing.flags = INLINE_FORWARD;
            packet_header.routing.dst_mesh_id = dest_device >> 16;
            packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
            packet_header.routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
            packet_header.session.command = ATOMIC_INC;
            packet_header.session.target_offset_l = target_address;
            packet_header.session.target_offset_h = 0x410;
            packet_header.packet_parameters.atomic_parameters.wrap_boundary = 31;
            packet_header.packet_parameters.atomic_parameters.increment = atomic_increment;
            tt_fabric_add_header_checksum(&packet_header);
            uint32_t words_left = words_to_init - words_initialized;
            bool split_header = words_left < PACKET_HEADER_SIZE_WORDS;
            uint32_t header_words_to_init = PACKET_HEADER_SIZE_WORDS;
            if (split_header) {
                header_words_to_init = words_left;
            }
            for (uint32_t i = 0; i < (header_words_to_init * PACKET_WORD_SIZE_BYTES / 4); i++) {
                header_ptr[i] = ((uint32_t*)&packet_header)[i];
            }

            words_initialized += header_words_to_init;
            input_queue_state.curr_packet_words_remaining -= header_words_to_init;
            byte_wr_addr += header_words_to_init * PACKET_WORD_SIZE_BYTES;
        } else {
            tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
            uint32_t header_words_initialized =
                input_queue_state.curr_packet_size_words - input_queue_state.curr_packet_words_remaining;
            uint32_t header_words_to_init = PACKET_HEADER_SIZE_WORDS - header_words_initialized;
            uint32_t header_dword_index = header_words_initialized * PACKET_WORD_SIZE_BYTES / 4;
            uint32_t words_left = words_to_init - words_initialized;
            header_words_to_init = std::min(words_left, header_words_to_init);

            for (uint32_t i = 0; i < (header_words_to_init * PACKET_WORD_SIZE_BYTES / 4); i++) {
                header_ptr[i] = ((uint32_t*)&packet_header)[i + header_dword_index];
            }
            words_initialized += header_words_to_init;
            input_queue_state.curr_packet_words_remaining -= header_words_to_init;
            byte_wr_addr += header_words_to_init * PACKET_WORD_SIZE_BYTES;
        }
    }
    test_producer.advance_local_wrptr(words_initialized);
    return false;
}

bool test_buffer_handler() {
    if constexpr (test_command == ASYNC_WR) {
        return test_buffer_handler_async_wr();
    } else if constexpr (test_command == ATOMIC_INC) {
        return test_buffer_handler_atomic_inc();
    }
}

void kernel_main() {
    tt_fabric_init();

    uint64_t router_config_addr = NOC_XY_ADDR(router_x, router_y, eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
    noc_async_read_one_packet(
        router_config_addr, routing_table_start_addr, sizeof(tt::tt_fabric::fabric_router_l1_config_t));
    noc_async_read_barrier();

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_STATUS_INDEX+1] = (uint32_t) local_pull_request;

    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;
    test_results[PQ_TEST_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(data_buffer_start_addr), data_buffer_size_words * PACKET_WORD_SIZE_BYTES);
    zero_l1_buf((uint32_t*)local_pull_request, sizeof(local_pull_request_t));
    zero_l1_buf((uint32_t*)&packet_header, sizeof(packet_header_t));

    if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
        input_queue_state.init(src_endpoint_id, prng_seed);
    } else if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
        input_queue_state.init(max_packet_size_words, 0);
    } else {
        input_queue_state.init(src_endpoint_id, prng_seed);
    }

    test_producer.init(data_buffer_start_addr, data_buffer_size_words, 0x0);

    uint32_t temp = max_packet_size_words;
    max_packet_size_mask = 0;
    temp >>= 1;
    while (temp) {
        max_packet_size_mask = (max_packet_size_mask << 1) + 1;
        temp >>= 1;
    }
    if ((max_packet_size_mask + 1) != max_packet_size_words) {
        // max_packet_size_words is not a power of 2
        // snap to next power of 2 mask
        max_packet_size_mask = (max_packet_size_mask << 1) + 1;
    }

    // wait till test sends start signal. This is set by test
    // once tt_fabric kernels have been launched on all the test devices.
    while (*(volatile uint32_t*)data_buffer_start_addr == 0);

    test_results[PQ_TEST_MISC_INDEX] = 0xff000001;

    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t zero_data_sent_iter = 0;
    uint64_t few_data_sent_iter = 0;
    uint64_t many_data_sent_iter = 0;
    uint64_t words_flushed = 0;
    bool timeout = false;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;

    uint32_t curr_packet_size = 0;
    uint32_t curr_packet_words_sent = 0;
    uint32_t packet_count = 0;

    while (true) {
        iter++;
#ifdef CHECK_TIMEOUT
        if (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }
#endif
        bool all_packets_initialized = test_buffer_handler();

        if (test_producer.get_curr_packet_valid()) {
            curr_packet_size = (test_producer.current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            uint32_t curr_data_words_sent = test_producer.pull_data_from_fvc_buffer<FVC_MODE_ENDPOINT>();
            curr_packet_words_sent += curr_data_words_sent;
            data_words_sent += curr_data_words_sent;
            if constexpr (!(data_sent_per_iter_low == 0 && data_sent_per_iter_high == 0)) {
                zero_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= 0);
                few_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= data_sent_per_iter_low);
                many_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent >= data_sent_per_iter_high);
            }
#ifdef CHECK_TIMEOUT
            progress_timestamp = (curr_data_words_sent > 0) ? get_timestamp_32b() : progress_timestamp;
#endif
            if (curr_packet_words_sent == curr_packet_size) {
                curr_packet_words_sent = 0;
                packet_count++;
            }
        } else if (test_producer.packet_corrupted) {
            DPRINT << "Packet Header Corrupted: packet " << packet_count
                   << " Addr: " << test_producer.get_local_buffer_read_addr() << ENDL();
            break;
        } else if (all_packets_initialized) {
            DPRINT << "all packets done" << ENDL();
            break;
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    uint64_t num_packets = input_queue_state.get_num_packets();
    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);
    set_64b_result(test_results, total_data_words, TX_TEST_IDX_TOT_DATA_WORDS);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);
    set_64b_result(test_results, zero_data_sent_iter, TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, few_data_sent_iter, TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, many_data_sent_iter, TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER);

    if (test_producer.packet_corrupted) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_BAD_HEADER;
        test_results[PQ_TEST_MISC_INDEX] = packet_count;
    } else if (!timeout) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_PASS;
        test_results[PQ_TEST_MISC_INDEX] = packet_count;
    } else {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        set_64b_result(test_results, words_flushed, TX_TEST_IDX_WORDS_FLUSHED);
    }
}
