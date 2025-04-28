// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <vector>

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
    // Full size channel
    // Full size channel
    // Full size channel
    //         .
    //         .
    // Header only channel
    // Header only channel
    // Header only channel
    //         .
    //         .
*/
struct FabricMuxConfig {
    static constexpr std::uint8_t noc_aligned_address_size_bytes = 16;

    std::uint8_t num_full_size_channels = 0;
    std::uint8_t num_header_only_channels = 0;
    std::uint8_t num_buffers_full_size_channel = 0;
    std::uint8_t num_buffers_header_only_channel = 0;
    std::size_t buffer_size_bytes_full_size_channel = 0;
    std::size_t base_l1_address = 0;

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
        buffer_size_bytes_full_size_channel(buffer_size_bytes_full_size_channel),
        base_l1_address(base_l1_address) {
        // TODO: asserts on the max size/number of channels allowed?

        this->full_size_channel_size_bytes = num_buffers_full_size_channel * buffer_size_bytes_full_size_channel;
        this->header_only_channel_size_bytes = num_buffers_header_only_channel * sizeof(tt::tt_fabric::PacketHeader);

        size_t address_block_size_bytes =
            (num_full_size_channels + num_header_only_channels) * noc_aligned_address_size_bytes;

        this->connection_info_base_address = base_l1_address;
        this->connection_handshake_base_address = this->connection_info_base_address + address_block_size_bytes;
        this->flow_control_base_address = this->connection_handshake_base_address + address_block_size_bytes;
        this->buffer_index_base_address = this->flow_control_base_address + address_block_size_bytes;
        this->full_size_channels_base_address = this->buffer_index_base_address + address_block_size_bytes;
        this->header_only_channels_base_address =
            this->full_size_channels_base_address + (num_full_size_channels * this->full_size_channel_size_bytes);

        // TODO: setup connection with the edm kernel
    }

    std::vector<uint32_t> get_fabric_mux_compile_time_args() {
        return std::vector<uint32_t>{
            this->num_full_size_channels,
            this->num_buffers_full_size_channel,
            this->buffer_size_bytes_full_size_channel,
            this->num_header_only_channels,
            this->num_buffers_header_only_channel,
            this->connection_info_base_address,
            this->connection_handshake_base_address,
            this->flow_control_base_address,
            this->full_size_channels_base_address};
    }

    std::vector<uint32_t> get_fabric_mux_run_time_args() { return std::vector<uint32_t>{}; }

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
        return get_address_from_block(channel_type, channel_id, this->connection_info_base_address);
    }

    std::size_t get_connection_handshake_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(channel_type, channel_id, this->connection_handshake_base_address);
    }

    std::size_t get_flow_control_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(channel_type, channel_id, this->flow_control_base_address);
    }

    std::size_t get_buffer_index_address(FabricMuxChannelType channel_type, uint8_t channel_id) {
        return get_address_from_block(channel_type, channel_id, this->buffer_index_base_address);
    }

private:
    std::size_t get_address_from_block(
        FabricMuxChannelType channel_type, uint8_t channel_id, std::size_t base_address) {
        std::size_t offset = 0;
        if (channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL) {
            offset = channel_id;
        } else if (channel_type == FabricMuxChannelType::HEADER_ONLY_CHANNEL) {
            offset = num_full_size_channels + channel_id;
        } else {
            TT_THROW("Unexpected channel type: {}", channel_type);
        }

        return base_address + (offset * noc_aligned_address_size_bytes);
        ;
    }

    std::size_t full_size_channel_size_bytes = 0;
    std::size_t header_only_channel_size_bytes = 0;

    // memory map
    std::size_t connection_info_base_address = 0;
    std::size_t connection_handshake_base_address = 0;
    std::size_t flow_control_base_address = 0;
    std::size_t buffer_index_base_address = 0;
    std::size_t full_size_channels_base_address = 0;
    std::size_t header_only_channels_base_address = 0;
};

}  // namespace tt::tt_fabric
