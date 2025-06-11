// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"

#include <cstdint>
#include <cstddef>

constexpr size_t NUM_STALL_DURATIONS = get_compile_time_arg_val(0);
constexpr size_t NUM_PACKET_SIZES = get_compile_time_arg_val(1);
constexpr size_t NUM_MESSAGES = get_compile_time_arg_val(2);

static FORCE_INLINE void setup_packet_header(
    volatile PACKET_HEADER_TYPE* pkt_hdr, size_t num_hops, tt::tt_fabric::ChipSendType chip_send_type) {
    if (num_hops > 0) {
        if (chip_send_type == tt::tt_fabric::CHIP_UNICAST) {
            pkt_hdr->to_chip_unicast(num_hops);
        } else {
            pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_hops)});
        }
    }
}

static inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    const size_t fabric_write_dest_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_write_dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_write_dest_noc_y = get_arg_val<uint32_t>(arg_idx++);


    bool is_starting_worker = get_arg_val<uint32_t>(arg_idx++);
    const size_t num_times_to_connect = get_arg_val<uint32_t>(arg_idx++);
    const size_t next_worker_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t next_worker_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t next_worker_connection_token_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    auto connection_token_ptr = reinterpret_cast<volatile uint32_t*>(next_worker_connection_token_addr);

    size_t num_stall_durations = get_arg_val<uint32_t>(arg_idx++);
    size_t* stall_durations = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_stall_durations;
    size_t num_packet_sizes = get_arg_val<uint32_t>(arg_idx++);
    size_t* packet_sizes = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_packet_sizes;
    size_t num_num_messages = get_arg_val<uint32_t>(arg_idx++);
    size_t* num_messages = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_num_messages;

    const size_t source_l1_cb_index = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);
    size_t packet_size_index = get_arg_val<uint32_t>(arg_idx++);;
    size_t num_messages_index = get_arg_val<uint32_t>(arg_idx++);;
    size_t stall_duration_index = get_arg_val<uint32_t>(arg_idx++);;

    auto fabric_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    cb_reserve_back(source_l1_cb_index, 1);
    cb_reserve_back(packet_header_cb, packet_header_size_in_headers);
    const auto source_l1_buffer_address = get_write_ptr(source_l1_cb_index);
    const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);

    auto* pkt_hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);


    if (is_starting_worker) {
        DPRINT << "Is starting worker\n";
        *connection_token_ptr += 1;
    }

    // We let workers send different traffic patterns across iterations
    uint32_t num_fwd_hops = 1;
    tt::tt_fabric::ChipSendType chip_send_type = CHIP_UNICAST;

    setup_packet_header(pkt_hdr_fwd, num_fwd_hops, chip_send_type);
    auto noc0_dest_addr_fwd = get_noc_addr(
        static_cast<uint8_t>(fabric_write_dest_noc_x),
        static_cast<uint8_t>(fabric_write_dest_noc_y),
        fabric_write_dest_bank_addr,
        0);

    uint64_t next_worker_token_addr =
        get_noc_addr(next_worker_noc_x, next_worker_noc_y, next_worker_connection_token_addr);
    bool wrap_val = 4;
    for (size_t i = 0; i < num_times_to_connect; i++) {
        auto stall_duration = stall_durations[stall_duration_index];
        auto packet_size = packet_sizes[packet_size_index];
        auto num_messages_to_send = num_messages[num_messages_index];

        pkt_hdr_fwd->to_noc_unicast_write(NocUnicastCommandHeader{noc0_dest_addr_fwd}, packet_size);

        while (*connection_token_ptr != 1) {
            noc_async_write(source_l1_buffer_address, noc0_dest_addr_fwd, packet_size);
        }
        *connection_token_ptr = 0;

        if (stall_duration) {
            auto start = get_timestamp();
            while (get_timestamp() - start < stall_duration) {}
        }
        fabric_connection.open();

        for (size_t i = 0; i < num_messages_to_send; i++) {
            // Forward direction
            fabric_connection.wait_for_empty_write_slot();
            fabric_connection.send_payload_without_header_non_blocking_from_address(
                source_l1_buffer_address, packet_size);
            fabric_connection.send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));
        }

        fabric_connection.close();

        // notify the next worker
        noc_semaphore_inc(next_worker_token_addr, 1);

        if ((i & (wrap_val - 1)) == 0) {
            stall_duration_index = tt::tt_fabric::wrap_increment<NUM_STALL_DURATIONS>(stall_duration_index);
            packet_size_index = tt::tt_fabric::wrap_increment<NUM_PACKET_SIZES>(packet_size_index);
            num_messages_index = tt::tt_fabric::wrap_increment<NUM_MESSAGES>(num_messages_index);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
