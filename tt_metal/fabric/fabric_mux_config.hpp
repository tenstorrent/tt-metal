// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <vector>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_fabric {

/*
    Full size channel supports transfers of packet header + payload.
    This should be used for cases when a payload needs to be sent to a remote end point.

    Header only channel only supports transfer of packet headers.
    This channel is for flow control and useful for sending credits back to the sender.
*/
enum class FabricMuxChannelType : uint8_t { FULL_SIZE_CHANNEL = 0, HEADER_ONLY_CHANNEL = 1 };

/*
    Channel layout: Each channel consists of buffers/slots
    ┌──────────────────────────┐
    │ ┌──┐┌──┐┌──┐        ┌──┐ │
    │ └──┘└──┘└──┘        └──┘ │
    └──────────────────────────┘

    Channel organization in L1:
    |  Full size channel  |
    |  Full size channel  |
    |  Full size channel  |
    |         .           |
    |         .           |
    | Header only channel |
    | Header only channel |
    | Header only channel |
    |          .          |
    |          .          |

    Basic operation:
        In each iteration the mux kernel round robins over all the channels, and forwards data over fabric.
        It processes the full size channels first and then the header only channels.

    Configuration parameters:
    -> Number of full size channels
    -> Number of header only channels
    -> Number of buffers/slots in a full size channel
    -> Number of buffers/slots in a header only channel
    -> Buffer size in bytes for a full size channel (for a header only channel its equal to the pre-determined packet
        header size)
    -> Base address where the channels start in the mux's L1

    Advanced configuation parameters:
    -> Number of full size channel iters
        This determines the number of full size channel iters to run per iter of header only channels.
        By default its set to 1, which indicates that the full size channels and header only channels are processed
        equally. This can be incremented in cases where the full size channels are not big enough compared to the
        buffers on the receiver. In such cases, the receiver can also accumulate credits and send them back in one shot
        instead of sending back one-by-one which may not always be the most efficient.
    -> Number of iters between teardown checks
        This determines how frequently the mux kernel checks for the termination signal. The larger this value, the less
        frequently mux kernel will check for the termination signal. Can be used to optimize performance, but very large
        values can impact teardown times.
*/
struct FabricMuxConfig {
    static constexpr std::uint8_t noc_aligned_address_size_bytes = 16;
    static constexpr std::size_t default_num_full_size_channel_iters = 1;
    static constexpr std::size_t default_num_iters_between_teardown_checks = 32;

    std::uint8_t num_full_size_channels = 0;
    std::uint8_t num_header_only_channels = 0;
    std::uint8_t num_buffers_full_size_channel = 0;
    std::uint8_t num_buffers_header_only_channel = 0;
    std::size_t buffer_size_bytes_full_size_channel = 0;

    FabricMuxConfig(
        std::uint8_t num_full_size_channels,
        std::uint8_t num_header_only_channels,
        std::uint8_t num_buffers_full_size_channel,
        std::uint8_t num_buffers_header_only_channel,
        std::size_t buffer_size_bytes_full_size_channel,
        std::size_t base_l1_address) :
        num_full_size_channels(num_full_size_channels),
        num_header_only_channels(num_header_only_channels),
        num_buffers_full_size_channel(num_buffers_full_size_channel),
        num_buffers_header_only_channel(num_buffers_header_only_channel),
        buffer_size_bytes_full_size_channel(buffer_size_bytes_full_size_channel) {
        // TODO: asserts on the max size/number of channels allowed?
        auto num_total_channels = num_full_size_channels + num_header_only_channels;

        this->full_size_channel_size_bytes = num_buffers_full_size_channel * buffer_size_bytes_full_size_channel;
        this->header_only_channel_size_bytes = num_buffers_header_only_channel * sizeof(tt::tt_fabric::PacketHeader);

        this->memory_map_start_address = base_l1_address;

        this->status_address = this->memory_map_start_address;
        this->termination_signal_address = this->status_address + noc_aligned_address_size_bytes;

        this->connection_info_base_address = this->termination_signal_address + noc_aligned_address_size_bytes;
        size_t connection_info_address_block_size_bytes =
            num_total_channels * sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo);

        this->connection_handshake_base_address =
            this->connection_info_base_address + connection_info_address_block_size_bytes;
        size_t address_block_size_bytes = num_total_channels * noc_aligned_address_size_bytes;

        this->flow_control_base_address = this->connection_handshake_base_address + address_block_size_bytes;
        this->buffer_index_base_address = this->flow_control_base_address + address_block_size_bytes;
        this->full_size_channels_base_address = this->buffer_index_base_address + address_block_size_bytes;

        this->header_only_channels_base_address =
            this->full_size_channels_base_address + (num_full_size_channels * this->full_size_channel_size_bytes);
        this->memory_map_end_address =
            this->header_only_channels_base_address + (num_header_only_channels * this->header_only_channel_size_bytes);
    }

    std::vector<uint32_t> get_fabric_mux_compile_time_args() {
        return std::vector<uint32_t>{
            this->num_full_size_channels,
            this->num_buffers_full_size_channel,
            this->buffer_size_bytes_full_size_channel,
            this->num_header_only_channels,
            this->num_buffers_header_only_channel,
            this->status_address,
            this->termination_signal_address,
            this->connection_info_base_address,
            this->connection_handshake_base_address,
            this->flow_control_base_address,
            this->full_size_channels_base_address,
            tt::tt_fabric::get_1d_fabric_config().sender_channels_num_buffers[0],
            this->num_full_size_channel_iters,
            this->num_iters_between_teardown_checks};
    }

    std::size_t get_status_address() { return this->status_address; }

    std::size_t get_termination_signal_address() { return this->termination_signal_address; }

    std::size_t get_channel_base_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        if (channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL) {
            return this->full_size_channels_base_address + (channel_id * this->full_size_channel_size_bytes);
        } else if (channel_type == FabricMuxChannelType::HEADER_ONLY_CHANNEL) {
            return this->header_only_channels_base_address + (channel_id * this->header_only_channel_size_bytes);
        } else {
            TT_THROW("Unexpected channel type: {}", channel_type);
        }

        return 0;
    }

    std::size_t get_connection_info_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(
            channel_type,
            channel_id,
            this->connection_info_base_address,
            sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo));
    }

    std::size_t get_connection_handshake_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(
            channel_type, channel_id, this->connection_handshake_base_address, noc_aligned_address_size_bytes);
    }

    std::size_t get_flow_control_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(
            channel_type, channel_id, this->flow_control_base_address, noc_aligned_address_size_bytes);
    }

    std::size_t get_buffer_index_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(
            channel_type, channel_id, this->buffer_index_base_address, noc_aligned_address_size_bytes);
    }

    std::size_t get_num_bytes_to_clear() { return this->memory_map_end_address - this->memory_map_start_address; }

    std::size_t get_start_address_to_clear() { return this->memory_map_start_address; }

    void set_num_full_size_channel_iters(std::size_t new_val) {
        if (this->num_full_size_channels > 0 && new_val == 0) {
            TT_THROW("Full size channels are present, but trying to set num iters as 0");
        }
        this->num_full_size_channel_iters = new_val;
    }

    void set_num_iters_between_teardown_checks(std::size_t new_val) {
        if (new_val == 0) {
            TT_THROW("Setting num iters b/w teardown checks to 0 will result in no data being sent over fabric");
        }
        this->num_iters_between_teardown_checks = new_val;
    }

private:
    std::size_t get_address_from_block(
        FabricMuxChannelType channel_type, uint8_t channel_id, std::size_t base_address, std::size_t unit_size_bytes) {
        std::size_t offset = 0;
        if (channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL) {
            offset = channel_id;
        } else if (channel_type == FabricMuxChannelType::HEADER_ONLY_CHANNEL) {
            offset = num_full_size_channels + channel_id;
        } else {
            TT_THROW("Unexpected channel type: {}", channel_type);
        }

        return base_address + (offset * unit_size_bytes);
    }

    std::size_t full_size_channel_size_bytes = 0;
    std::size_t header_only_channel_size_bytes = 0;

    std::size_t num_full_size_channel_iters = default_num_full_size_channel_iters;
    std::size_t num_iters_between_teardown_checks = default_num_iters_between_teardown_checks;

    // memory map
    std::size_t memory_map_start_address = 0;
    std::size_t status_address = 0;
    std::size_t termination_signal_address = 0;
    std::size_t connection_info_base_address = 0;
    std::size_t connection_handshake_base_address = 0;
    std::size_t flow_control_base_address = 0;
    std::size_t buffer_index_base_address = 0;
    std::size_t full_size_channels_base_address = 0;
    std::size_t header_only_channels_base_address = 0;
    std::size_t memory_map_end_address = 0;
};

}  // namespace tt::tt_fabric
