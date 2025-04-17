// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>

namespace tt::tt_fabric {

struct BaseWorkerConfig {
    static constexpr uint8_t TEST_RESULTS_SIZE_BYTES = 128;
    static constexpr uint16_t PACKET_HEADER_BUFFER_SIZE_BYTES = 1024;
    static constexpr uint8_t NUM_DIRECTIONS = 4;

    // for now set the l1 per sender to be 28 KB
    static constexpr uint32_t L1_BUFFER_SIZE_PER_SENDER_BYTES = 28 * 1024;
    static constexpr uint8_t MAX_NUM_SENDERS_PER_RECEIVER = 4;

    BaseWorkerConfig(
        uint32_t base_address, uint32_t packet_payload_size_bytes, uint32_t num_packets, uint32_t time_seed) {
        // test results should always start at the base address
        this->test_results_address = base_address;
        this->base_target_address = this->test_results_address + TEST_RESULTS_SIZE_BYTES;

        this->packet_payload_size_bytes = packet_payload_size_bytes;
        this->num_packets = num_packets;
        this->time_seed = time_seed;
    }

    // common memory map
    uint32_t test_results_address = 0;
    uint32_t base_target_address = 0;

    // common test parameters
    uint32_t packet_payload_size_bytes = 0;
    uint32_t num_packets = 0;
    uint32_t time_seed = 0;
};

struct SenderWorkerConfig : BaseWorkerConfig {
    static SenderWorkerConfig build_from_args(std::size_t& arg_idx) {
        uint32_t base_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t routing_plane_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
        uint32_t time_seed = get_arg_val<uint32_t>(arg_idx++);
        uint32_t sender_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t rx_noc_encoding = get_arg_val<uint32_t>(arg_idx++);

        tt_l1_ptr uint32_t* is_mcast_enabled = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(agr_idx));
        arg_idx += NUM_DIRECTIONS;

        tt_l1_ptr uint32_t* hop_counts = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(agr_idx));
        arg_idx += NUM_DIRECTIONS;

        return SenderWorkerConfig(
            base_address,
            routing_plane_id,
            packet_payload_size_bytes,
            num_packets,
            time_seed,
            sender_id,
            rx_noc_encoding,
            is_mcast_enabled,
            hop_counts);
    }

    SenderWorkerConfig(
        uint32_t base_address,
        uint32_t routing_plane_id,
        uint32_t packet_payload_size_bytes,
        uint32_t num_packets,
        uint32_t time_seed,
        uint32_t rx_noc_encoding,
        tt_l1_ptr uint32_t* is_mcast_enabled,
        tt_l1_ptr uint32_t* hops_count) :
        BaseWorkerConfig(base_address, packet_payload_size_bytes, num_packets, time_seed) {
        this->packet_header_buffer_address = this->test_results_address + TEST_RESULTS_SIZE_BYTES;
        this->source_l1_buffer_address = this->packet_header_buffer_address + PACKET_HEADER_BUFFER_SIZE_BYTES;
        this->target_address = this->base_target_address + (L1_BUFFER_SIZE_PER_SENDER_BYTES * routing_plane_id);
        this->rx_noc_encoding = rx_noc_encoding;

        for (auto i = 0; i < NUM_DIRECTIONS; i++) {
            this->is_mcast_enabled[i] = is_mcast_enabled[i];
            this->hop_count[i] = hops_count[i];
        }
    }

    // memory map
    uint32_t packet_header_buffer_address = 0;
    uint32_t payload_buffer_address = 0;
    uint32_t target_address = 0;

    // test parameters
    uint32_t sender_id;
    uint32_t rx_noc_encoding = 0;
    std::array<bool, NUM_DIRECTIONS> is_mcast_enabled = {false};
    std::array<uint32_t, NUM_DIRECTIONS> hops_count = {0};
};

struct ReceiverWorkerConfig : BaseWorkerConfig {
    static ReceiverChannelPointers build_from_args(std::size_t& arg_idx) {
        uint32_t base_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
        uint32_t time_seed = get_arg_val<uint32_t>(arg_idx++);

        uint32_t num_senders = get_arg_val<uint32_t>(arg_idx++);
        tt_l1_ptr uint32_t* sender_ids = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(agr_idx));
        arg_idx += num_senders;

        return ReceiverWorkerConfig(
            base_address, packet_payload_size_bytes, num_packets, time_seed, num_senders, sender_ids);
    }

    ReceiverWorkerConfig(
        uint32_t base_address,
        uint32_t packet_payload_size_bytes,
        uint32_t num_packets,
        uint32_t time_seed,
        uint32_t num_senders,
        tt_l1_ptr uint32_t* sender_ids) :
        BaseWorkerConfig(base_address, packet_payload_size_bytes, num_packets, time_seed) {
        this->num_senders = num_senders;
        for (auto i = 0; i < this->num_senders; i++) {
            this->sender_ids[i] = sender_ids[i];
            this->target_addresses[i] = this->base_target_address + (L1_BUFFER_SIZE_PER_SENDER_BYTES * i);
        }
    }

    // memory map
    std::array<uint32_t, MAX_NUM_SENDERS_PER_RECEIVER> target_addresses = {0};

    // test parameters
    uint32_t num_senders = 0;
    std::array<uint32_t, MAX_NUM_SENDERS_PER_RECEIVER> sender_ids = {0};
};

}  // namespace tt::tt_fabric
