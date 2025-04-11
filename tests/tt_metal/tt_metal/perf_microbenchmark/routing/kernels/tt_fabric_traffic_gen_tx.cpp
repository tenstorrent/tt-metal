// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
// clang-format on

using namespace tt::tt_fabric;

uint32_t src_endpoint_id;
// constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);
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

constexpr bool fixed_async_wr_notif_addr = get_compile_time_arg_val(22);

constexpr bool mcast_data = get_compile_time_arg_val(23);
constexpr uint32_t e_depth = get_compile_time_arg_val(24);
constexpr uint32_t w_depth = get_compile_time_arg_val(25);
constexpr uint32_t n_depth = get_compile_time_arg_val(26);
constexpr uint32_t s_depth = get_compile_time_arg_val(27);
constexpr uint32_t router_mode = get_compile_time_arg_val(28);

uint32_t max_packet_size_mask;

auto input_queue_state = select_input_queue<pkt_dest_size_choice>();
volatile local_pull_request_t *local_pull_request = (volatile local_pull_request_t *)(data_buffer_start_addr - 1024);
volatile tt_l1_ptr fabric_router_l1_config_t* routing_table;

#ifdef FVC_MODE_PULL
fvc_inbound_pull_state_t test_producer __attribute__((aligned(16)));
volatile fabric_pull_client_interface_t* client_interface =
    (volatile fabric_pull_client_interface_t*)client_interface_addr;
#else
fvc_inbound_push_state_t test_producer __attribute__((aligned(16)));
volatile fabric_push_client_interface_t* client_interface =
    (volatile fabric_push_client_interface_t*)client_interface_addr;
#endif
fvcc_inbound_state_t fvcc_test_producer __attribute__((aligned(16)));

uint64_t xy_local_addr;

packet_header_t packet_header __attribute__((aligned(16)));
low_latency_packet_header_t low_latency_packet_header __attribute__((aligned(16)));

uint32_t target_address;
uint32_t noc_offset;
uint32_t rx_addr_hi;
uint32_t controller_noc_offset;

// flag to check if need to zero out notification addr
bool reset_notif_addr = true;

uint32_t time_seed;

inline void notify_traffic_controller() {
    // send semaphore increment to traffic controller kernel on this device.
    uint64_t dest_addr = get_noc_addr_helper(controller_noc_offset, signal_address);
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
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

#ifdef FVC_MODE_PULL
// generates packets with random size and payload on the input sideß
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
    uint32_t curr_payload_bytes = 0;
    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) { // start of a new packet
            input_queue_state.next_packet(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);

            tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
            curr_payload_bytes =
                (input_queue_state.curr_packet_size_words - PACKET_HEADER_SIZE_WORDS) * PACKET_WORD_SIZE_BYTES;

            // check for wrap
            // if the size of fvc buffer is greater than the rx buffer size and/or rx is slow
            // data validation on rx could fail as tx could overwrite data
            // explicit sync is needed to ensure data validation in all scenarios
            if (target_address + curr_payload_bytes > rx_addr_hi) {
                target_address = base_target_address;
            }

            packet_header.routing.flags = FORWARD | (mcast_data ? MCAST_DATA : 0);
            packet_header.routing.packet_size_bytes = input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
            packet_header.routing.dst_mesh_id = dest_device >> 16;
            packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
            packet_header.session.command = ASYNC_WR;
            if constexpr (test_command & ATOMIC_INC) {
                packet_header.session.command |= ATOMIC_INC;
                packet_header.packet_parameters.async_wr_atomic_parameters.noc_xy = noc_offset;
                packet_header.packet_parameters.async_wr_atomic_parameters.increment = atomic_increment;
                if constexpr (fixed_async_wr_notif_addr) {
                    packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset = base_target_address;
                } else {
                    packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset = target_address;
                    reset_notif_addr = true;
                }
            }
            packet_header.session.target_offset_l = target_address;
            packet_header.session.target_offset_h = noc_offset;
            target_address += packet_header.routing.packet_size_bytes - PACKET_HEADER_SIZE_BYTES;
            if constexpr (mcast_data) {
                packet_header.packet_parameters.mcast_parameters.east = e_depth;
                packet_header.packet_parameters.mcast_parameters.west = w_depth;
                packet_header.packet_parameters.mcast_parameters.north = n_depth;
                packet_header.packet_parameters.mcast_parameters.south = s_depth;
            }
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
            if constexpr (test_command & ATOMIC_INC) {
                if (reset_notif_addr) {
                    tt_l1_ptr uint32_t* addr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
                    *addr = time_seed + input_queue_state.get_num_packets();
                    reset_notif_addr = false;
                }
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
            packet_header.session.target_offset_h = noc_offset;
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
#else
// generates packets with random size and payload on the input sideß
inline bool test_buffer_handler_async_wr() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_slots = test_producer.get_num_slots_free();
    if (free_slots < 1) {
        // no buffer slot available for new packet.
        return false;
    }

    // Each call to test_buffer_handler initializes only up to the end
    // of the producer buffer. Since the header is 3 words, we need to handle
    // split header cases, where the buffer has not enough space for the full header.
    // In this case, we write as many words as space available, and remaining header words
    // are written on next call.
    uint32_t slots_initialized = 0;
    uint32_t curr_payload_bytes = 0;
    while (slots_initialized < free_slots) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        uint32_t byte_wr_addr = test_producer.get_local_buffer_write_addr();

        input_queue_state.next_packet(
            num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);

        tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
        curr_payload_bytes =
            (input_queue_state.curr_packet_size_words - PACKET_HEADER_SIZE_WORDS) * PACKET_WORD_SIZE_BYTES;

        // check for wrap
        // if the size of fvc buffer is greater than the rx buffer size and/or rx is slow
        // data validation on rx could fail as tx could overwrite data
        // explicit sync is needed to ensure data validation in all scenarios
        if (target_address + curr_payload_bytes > rx_addr_hi) {
            target_address = base_target_address;
        }

#ifdef LOW_LATENCY_ROUTING
        low_latency_packet_header.routing.packet_size_bytes =
            input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
        low_latency_packet_header.routing.target_offset_l = target_address;
        low_latency_packet_header.routing.target_offset_h = noc_offset;
        low_latency_packet_header.routing.command = ASYNC_WR;

        if constexpr (test_command & ATOMIC_INC) {
            low_latency_packet_header.routing.command |= ATOMIC_INC;
            low_latency_packet_header.routing.atomic_offset_h = noc_offset;
            low_latency_packet_header.routing.atomic_increment = atomic_increment;
            low_latency_packet_header.routing.atomic_wrap = 31;
            if constexpr (fixed_async_wr_notif_addr) {
                low_latency_packet_header.routing.atomic_offset_l = base_target_address;
            } else {
                low_latency_packet_header.routing.atomic_offset_l = target_address;
                reset_notif_addr = true;
            }
        }

        target_address += low_latency_packet_header.routing.packet_size_bytes - PACKET_HEADER_SIZE_BYTES;
        // low latency routing fields are alread in the cached header.
        for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_BYTES / 4); i++) {
            header_ptr[i] = ((uint32_t*)&low_latency_packet_header)[i];
        }
#else
        packet_header.routing.flags = FORWARD | (mcast_data ? MCAST_DATA : 0);
        packet_header.routing.packet_size_bytes = input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
        packet_header.routing.dst_mesh_id = dest_device >> 16;
        packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
        packet_header.session.command = ASYNC_WR;
        if constexpr (test_command & ATOMIC_INC) {
            packet_header.session.command |= ATOMIC_INC;
            packet_header.packet_parameters.async_wr_atomic_parameters.noc_xy = noc_offset;
            packet_header.packet_parameters.async_wr_atomic_parameters.increment = atomic_increment;
            if constexpr (fixed_async_wr_notif_addr) {
                packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset = base_target_address;
            } else {
                packet_header.packet_parameters.async_wr_atomic_parameters.l1_offset = target_address;
                reset_notif_addr = true;
            }
        }
        packet_header.session.target_offset_l = target_address;
        packet_header.session.target_offset_h = noc_offset;
        target_address += packet_header.routing.packet_size_bytes - PACKET_HEADER_SIZE_BYTES;
        if constexpr (mcast_data) {
            packet_header.packet_parameters.mcast_parameters.east = e_depth;
            packet_header.packet_parameters.mcast_parameters.west = w_depth;
            packet_header.packet_parameters.mcast_parameters.north = n_depth;
            packet_header.packet_parameters.mcast_parameters.south = s_depth;
        }
        tt_fabric_add_header_checksum(&packet_header);
        for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_BYTES / 4); i++) {
            header_ptr[i] = ((uint32_t*)&packet_header)[i];
        }
#endif
        input_queue_state.curr_packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
        byte_wr_addr += PACKET_HEADER_SIZE_BYTES;

        if constexpr (!skip_pkt_content_gen) {
            uint32_t start_val = (input_queue_state.packet_rnd_seed & 0xFFFF0000) +
                                 (input_queue_state.curr_packet_size_words -
                                  input_queue_state.curr_packet_words_remaining - PACKET_HEADER_SIZE_WORDS);
            fill_packet_data(
                reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr),
                input_queue_state.curr_packet_words_remaining,
                start_val);
        }
        if constexpr (test_command & ATOMIC_INC) {
            if (reset_notif_addr) {
                tt_l1_ptr uint32_t* addr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
                *addr = time_seed + input_queue_state.get_num_packets();
                reset_notif_addr = false;
            }
        }
        slots_initialized += 1;
        input_queue_state.curr_packet_words_remaining = 0;
        test_producer.advance_local_wrptr(1);
    }
    return false;
}

// generates packets with random size and payload on the input side
inline bool test_buffer_handler_atomic_inc() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_slots = test_producer.get_num_slots_free();
    if (free_slots < 1) {
        return false;
    }

    uint32_t slots_initialized = 0;
    while (slots_initialized < free_slots) {
        if (input_queue_state.all_packets_done()) {
            break;
        }
        uint32_t byte_wr_addr = test_producer.get_local_buffer_write_addr();
        input_queue_state.next_inline_packet(total_data_words);
        tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);
#ifdef LOW_LATENCY_ROUTING
        low_latency_packet_header.routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
        low_latency_packet_header.routing.target_offset_h = noc_offset;
        low_latency_packet_header.routing.target_offset_l = target_address;
        low_latency_packet_header.routing.atomic_offset_h = noc_offset;
        low_latency_packet_header.routing.atomic_offset_l = target_address;
        low_latency_packet_header.routing.atomic_increment = atomic_increment;
        low_latency_packet_header.routing.atomic_wrap = 31;
        low_latency_packet_header.routing.command = ATOMIC_INC;
        // low latency routing fields are alread in the cached header.
        for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_BYTES / 4); i++) {
            header_ptr[i] = ((uint32_t*)&low_latency_packet_header)[i];
        }
#else
        packet_header.routing.flags = INLINE_FORWARD;
        packet_header.routing.dst_mesh_id = dest_device >> 16;
        packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
        packet_header.routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
        packet_header.session.command = ATOMIC_INC;
        packet_header.session.target_offset_l = target_address;
        packet_header.session.target_offset_h = noc_offset;
        packet_header.packet_parameters.atomic_parameters.wrap_boundary = 31;
        packet_header.packet_parameters.atomic_parameters.increment = atomic_increment;
        tt_fabric_add_header_checksum(&packet_header);
        for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_BYTES / 4); i++) {
            header_ptr[i] = ((uint32_t*)&packet_header)[i];
        }
#endif
        slots_initialized += 1;
        input_queue_state.curr_packet_words_remaining = 0;
        test_producer.advance_local_wrptr(1);
    }
    return false;
}
#endif

inline bool test_buffer_handler_fvcc() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_words = fvcc_test_producer.get_num_msgs_free() * PACKET_HEADER_SIZE_WORDS;
    if (free_words < PACKET_HEADER_SIZE_WORDS) {
        return false;
    }

    uint32_t byte_wr_addr = fvcc_test_producer.get_local_buffer_write_addr();
    uint32_t words_to_init = std::min(free_words, fvcc_test_producer.words_before_local_buffer_wrap());
    uint32_t words_initialized = 0;
    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) {  // start of a new packet
            input_queue_state.next_inline_packet(total_data_words);

            tt_l1_ptr uint32_t* header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr);

            packet_header.routing.flags = SYNC;
            packet_header.routing.dst_mesh_id = dest_device >> 16;
            packet_header.routing.dst_dev_id = dest_device & 0xFFFF;
            packet_header.routing.src_dev_id = routing_table->my_device_id;
            packet_header.routing.src_mesh_id = routing_table->my_mesh_id;
            packet_header.routing.packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
            packet_header.session.command = ASYNC_WR_RESP;
            packet_header.session.target_offset_l = target_address;
            packet_header.session.target_offset_h = noc_offset;
            packet_header.packet_parameters.misc_parameters.words[1] = 0;
            packet_header.packet_parameters.misc_parameters.words[2] = 0;
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
    fvcc_test_producer.advance_local_wrptr(words_initialized / PACKET_HEADER_SIZE_WORDS);
    return false;
}

bool test_buffer_handler() {
    if constexpr (test_command & ASYNC_WR) {
        return test_buffer_handler_async_wr();
    } else if constexpr (test_command == ATOMIC_INC) {
        return test_buffer_handler_atomic_inc();
    } else if constexpr (test_command == ASYNC_WR_RESP) {
        return test_buffer_handler_fvcc();
    }

    return true;
}

void kernel_main() {
    uint32_t rt_args_idx = 0;
    time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    src_endpoint_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    controller_noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t outbound_eth_chan = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    dest_device = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t rx_buf_size = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    if constexpr (ASYNC_WR & test_command) {
        base_target_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    }

    target_address = base_target_address;
    rx_addr_hi = base_target_address + rx_buf_size;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;
    test_results[TT_FABRIC_STATUS_INDEX + 1] = (uint32_t)local_pull_request;

    test_results[TT_FABRIC_MISC_INDEX] = 0xff000000;
    test_results[TT_FABRIC_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

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

    test_producer.init(data_buffer_start_addr, data_buffer_size_words);
    fvcc_test_producer.init(data_buffer_start_addr, 0x0, 0x0);

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

    // notify the controller kernel that this worker is ready to proceed
    notify_traffic_controller();

    // wait till controllrer sends start signal. This is set by controller
    // once tt_fabric kernels have been launched on all the test devices and
    // all the tx workers are ready on this chip
    while (*(volatile tt_l1_ptr uint32_t*)signal_address == 0);

    test_results[TT_FABRIC_MISC_INDEX] = 0xff000001;

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

    // initalize client
    tt_fabric_init();
    fabric_endpoint_init<RoutingType::ROUTING_TABLE>(client_interface, outbound_eth_chan);
    routing_table = reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(client_interface->routing_tables_l1_offset);

#ifndef FVC_MODE_PULL
    test_producer.register_with_routers<FVC_MODE_ENDPOINT>(dest_device & 0xFFFF, dest_device >> 16);
#ifdef LOW_LATENCY_ROUTING
    // For low latency routing mode, calculate the unicast/multicast route once and put it in header.
    // packet size and target address will get update for each packet.
    // the destination devices remain the same for every packet so the routes can be populated in
    // header once.
    if constexpr (mcast_data) {
        if constexpr (e_depth) {
            fabric_set_mcast_route(&low_latency_packet_header, eth_chan_directions::EAST, e_depth);
        } else if constexpr (w_depth) {
            fabric_set_mcast_route(&low_latency_packet_header, eth_chan_directions::WEST, w_depth);
        } else if constexpr (n_depth) {
            fabric_set_mcast_route(&low_latency_packet_header, eth_chan_directions::NORTH, n_depth);
        } else if constexpr (s_depth) {
            fabric_set_mcast_route(&low_latency_packet_header, eth_chan_directions::SOUTH, s_depth);
        }
    } else {
        uint32_t outgoing_direction =
            get_next_hop_router_direction(client_interface, 0, dest_device >> 16, dest_device & 0xFFFF);
        fabric_set_unicast_route(
            client_interface, &low_latency_packet_header, outgoing_direction, dest_device & 0xFFFF);
    }
#endif
#endif

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

        if (test_producer.get_curr_packet_valid<FVC_MODE_ENDPOINT>()) {
#ifdef FVC_MODE_PULL
            curr_packet_size =
                (test_producer.current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            uint32_t curr_data_words_sent = test_producer.pull_data_from_fvc_buffer<FVC_MODE_ENDPOINT>();
#else
            curr_packet_size =
                ((test_producer.packet_header->routing.packet_size_bytes & 0x3FFFFFFF) + PACKET_WORD_SIZE_BYTES - 1) >>
                4;
#ifdef LOW_LATENCY_ROUTING
            uint32_t curr_data_words_sent = test_producer.push_data_to_eth_router<FVC_MODE_ENDPOINT>(dest_device);
#else
            uint32_t curr_data_words_sent = test_producer.push_data_to_eth_router<FVC_MODE_ENDPOINT>();
#endif
#endif
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
        } else if (fvcc_test_producer.get_curr_packet_valid()) {
            fvcc_test_producer.fvcc_handler<FVC_MODE_ENDPOINT>();
#ifdef CHECK_TIMEOUT
            progress_timestamp = get_timestamp_32b();
#endif
        } else if (fvcc_test_producer.packet_corrupted) {
            DPRINT << "Packet Header Corrupted: packet " << packet_count
                   << " Addr: " << fvcc_test_producer.get_local_buffer_read_addr() << ENDL();
            break;
        } else if (all_packets_initialized) {
            DPRINT << "all packets done" << ENDL();
            break;
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    uint64_t num_packets = input_queue_state.get_num_packets();
    set_64b_result(test_results, data_words_sent, TT_FABRIC_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, TT_FABRIC_CYCLES_INDEX);
    set_64b_result(test_results, iter, TT_FABRIC_ITER_INDEX);
    set_64b_result(test_results, total_data_words, TX_TEST_IDX_TOT_DATA_WORDS);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);
    set_64b_result(test_results, zero_data_sent_iter, TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, few_data_sent_iter, TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, many_data_sent_iter, TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER);

    if (!timeout) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
        test_results[TT_FABRIC_MISC_INDEX] = packet_count;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_TIMEOUT;
        set_64b_result(test_results, words_flushed, TX_TEST_IDX_WORDS_FLUSHED);
    }
}
