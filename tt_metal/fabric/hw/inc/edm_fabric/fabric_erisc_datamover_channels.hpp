// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "dataflow_api.h"
#if defined(COMPILE_FOR_ERISC)
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#endif
#include "tt_metal/hw/inc/utils/utils.h"
#include "risc_attribs.h"
#include "fabric_edm_packet_header.hpp"
#include "fabric_edm_types.hpp"
#include "edm_fabric_worker_adapters.hpp"
#include "edm_fabric_flow_control_helpers.hpp"

// !!! TODO: delete this once push/pull 2D tests/code is deprecated !!!
#if (ROUTING_MODE & ROUTING_MODE_PULL) || (ROUTING_MODE & ROUTING_MODE_PUSH)
namespace tt::tt_fabric {
static constexpr uint8_t worker_handshake_noc = 0;
}  // namespace tt::tt_fabric
#endif

namespace tt::tt_fabric {

template <typename T>
FORCE_INLINE auto wrap_increment(T val, size_t max) {
    return (val == max - 1) ? 0 : val + 1;
}

template <uint8_t NUM_BUFFERS>
class EthChannelBuffer final {
public:
    // The channel structure is as follows:
    //              &header->  |----------------| channel_base_address
    //                         |    header      |
    //             &payload->  |----------------|
    //                         |                |
    //                         |    payload     |
    //                         |                |
    //                         |----------------|

    EthChannelBuffer() : buffer_size_in_bytes(0), max_eth_payload_size_in_bytes(0) {}

    /*
     * Expected that *buffer_index_ptr is initialized outside of this object
     */
    EthChannelBuffer(
        size_t channel_base_address, size_t buffer_size_bytes, size_t header_size_bytes, uint8_t channel_id) :
        buffer_size_in_bytes(buffer_size_bytes),
        max_eth_payload_size_in_bytes(buffer_size_in_bytes),
        channel_id(channel_id) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] = channel_base_address + i * this->max_eth_payload_size_in_bytes;
// need to avoid unrolling to keep code size within limits
#pragma GCC unroll 1
            for (size_t j = 0; j < sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t); j++) {
                reinterpret_cast<volatile uint32_t*>(this->buffer_addresses[i])[j] = 0;
            }
        }
        set_cached_next_buffer_slot_addr(this->buffer_addresses[0]);
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(const BufferIndex& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE volatile T* get_packet_header(const BufferIndex& buffer_index) const {
        return reinterpret_cast<volatile T*>(this->buffer_addresses[buffer_index]);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE size_t get_payload_size(const BufferIndex& buffer_index) const {
        return get_packet_header<T>(buffer_index)->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes(const BufferIndex& buffer_index) const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size() const { return this->max_eth_payload_size_in_bytes; }

    [[nodiscard]] FORCE_INLINE size_t get_id() const { return this->channel_id; }

#if defined(COMPILE_FOR_ERISC)
    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed(const BufferIndex& buffer_index) const {
        return eth_is_receiver_channel_send_acked(buffer_index) || eth_is_receiver_channel_send_done(buffer_index);
    }
#endif

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const { return this->cached_next_buffer_slot_addr; }

    FORCE_INLINE void set_cached_next_buffer_slot_addr(size_t next_buffer_slot_addr) {
        this->cached_next_buffer_slot_addr = next_buffer_slot_addr;
    }

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;

    // header + payload regions only
    const std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    const std::size_t max_eth_payload_size_in_bytes;
    std::size_t cached_next_buffer_slot_addr;
    uint8_t channel_id;
};


// A tuple of EthChannelBuffer
template <size_t... BufferSizes>
struct EthChannelBufferTuple {
    std::tuple<tt::tt_fabric::EthChannelBuffer<BufferSizes>...> channel_buffers;

    void init(
        const size_t channel_base_address[],
        const size_t buffer_size_bytes,
        const size_t header_size_bytes,
        const size_t channel_base_id) {
        size_t idx = 0;

        std::apply(
            [&](auto&... chans) {
                ((new (&chans) std::remove_reference_t<decltype(chans)>(
                      channel_base_address[idx],
                      buffer_size_bytes,
                      header_size_bytes,
                      static_cast<uint8_t>(channel_base_id + idx)),
                  ++idx),
                 ...);
            },
            channel_buffers);
    }

    template <size_t I>
    auto& get() {
        return std::get<I>(channel_buffers);
    }
};

template <auto& ChannelBuffers>
struct EthChannelBuffers {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        return EthChannelBufferTuple<ChannelBuffers[Is]...>{};
    }
};

}  // namespace tt::tt_fabric
