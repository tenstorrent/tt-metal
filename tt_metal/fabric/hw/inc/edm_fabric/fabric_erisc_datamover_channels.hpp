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
            for (size_t j = 0; j < this->max_eth_payload_size_in_bytes; j++) {
                reinterpret_cast<volatile uint8_t*>(this->buffer_addresses[i])[j] = 0;
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

    FORCE_INLINE bool needs_to_send_channel_sync() const { return this->need_to_send_channel_sync; }

    FORCE_INLINE void set_need_to_send_channel_sync(bool need_to_send_channel_sync) {
        this->need_to_send_channel_sync = need_to_send_channel_sync;
    }

    FORCE_INLINE void clear_need_to_send_channel_sync() { this->need_to_send_channel_sync = false; }

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

// Note that this class implements a mix of interfaces and will need to be separated to just be different
// interface types altogether.
//
// The two types of interfaces implemented/supported here are hardcoded by producer type (EDM or Worker)
// but they should be split based on credit exchange protocol (read/write counter vs free slots)
// Additionally, a nice to have would be if we could further create types for different credit
// storage mechanisms (e.g. L1 vs stream registers)
//
template <uint8_t NUM_BUFFERS>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(nullptr),
        sender_sync_noc_cmd_buf(write_at_cmd_buf) {}
    EdmChannelWorkerInterface(
        // TODO: PERF: See if we can make this non-volatile and then only
        // mark it volatile when we know we need to reload it (i.e. after we receive a
        // "done" message from sender)
        // Have a volatile update function that only triggers after reading the volatile
        // completion field so that way we don't have to do a volatile read for every
        // packet... Then we'll also be able to cache the uint64_t addr of the worker
        // semaphore directly (saving on regenerating it each time)
        volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t* const remote_producer_write_counter,
        volatile tt_l1_ptr uint32_t* const connection_live_semaphore,
        uint8_t sender_sync_noc_cmd_buf,
        uint8_t edm_read_counter_initial_value) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(connection_live_semaphore),
        sender_sync_noc_cmd_buf(sender_sync_noc_cmd_buf) {
        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_read_counter)) = edm_read_counter_initial_value;
        local_write_counter.reset();
        local_read_counter.reset();
    }

    // Flow control methods
    //
    // local_wrptr trails from_remote_wrptr
    // we have new data if they aren't equal

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }

    // Only used for persistent connections (i.e. upstream is EDM)
    template <bool enable_ring_support>
    FORCE_INLINE void update_persistent_connection_copy_of_free_slots(int32_t inc_val) {
        noc_inline_dw_write<true, true>(
            this->cached_worker_semaphore_address,
            inc_val << REMOTE_DEST_BUF_WORDS_FREE_INC,
            0xf,
            tt::tt_fabric::worker_handshake_noc);
    }

    FORCE_INLINE void notify_worker_of_read_counter_update() {
        noc_inline_dw_write<true, true>(
            this->cached_worker_semaphore_address,
            local_read_counter.counter,
            0xf,
            tt::tt_fabric::worker_handshake_noc);
    }

    FORCE_INLINE void increment_local_read_counter(int32_t inc_val) {
        local_read_counter.counter += inc_val;
    }

    FORCE_INLINE void copy_read_counter_to_worker_location_info() const {
        worker_location_info_ptr->edm_read_counter = local_read_counter.counter;
    }

    // Connection management methods
    //
    template <bool posted = false>
    FORCE_INLINE void teardown_worker_connection() const {
        invalidate_l1_cache();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x,
            (uint32_t)worker_info.worker_xy.y,
            worker_info.worker_teardown_semaphore_address);

        // Set connection to unused so it's available for next worker
        *this->connection_live_semaphore = tt::tt_fabric::EdmToEdmSender<0>::unused_connection_value;

        this->copy_read_counter_to_worker_location_info();

        noc_semaphore_inc<posted>(worker_semaphore_address, 1, tt::tt_fabric::worker_handshake_noc);
    }

    FORCE_INLINE void cache_producer_noc_addr() {
        invalidate_l1_cache();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        invalidate_l1_cache();
        return *connection_live_semaphore == tt::tt_fabric::EdmToEdmSender<0>::close_connection_request_value;
    }
    [[nodiscard]] FORCE_INLINE bool connection_is_live() const {
        invalidate_l1_cache();
        return *connection_live_semaphore == tt::tt_fabric::EdmToEdmSender<0>::open_connection_value;
    }

    volatile tt_l1_ptr EDMChannelWorkerLocationInfo* worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t* const connection_live_semaphore;
    uint8_t sender_sync_noc_cmd_buf;

    ChannelCounter<NUM_BUFFERS> local_write_counter;
    ChannelCounter<NUM_BUFFERS> local_read_counter;
};

// A tuple of EDM channel worker interfaces
template <size_t... BufferSizes>
struct EdmChannelWorkerInterfaceTuple {
    // tuple of EdmChannelWorkerInterface<BufferSizes>...
    std::tuple<tt::tt_fabric::EdmChannelWorkerInterface<BufferSizes>...> channel_worker_interfaces;

    template <size_t I>
    auto& get() {
        return std::get<I>(channel_worker_interfaces);
    }
};

template <auto& ChannelBuffers>
struct EdmChannelWorkerInterfaces {
    template <size_t... Is>
    static auto make(std::index_sequence<Is...>) {
        return EdmChannelWorkerInterfaceTuple<ChannelBuffers[Is]...>{};
    }
};

}  // namespace tt::tt_fabric
