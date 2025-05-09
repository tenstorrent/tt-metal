// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "debug/dprint.h"
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
        size_t channel_base_address,
        size_t buffer_size_bytes,
        size_t header_size_bytes,
        size_t eth_transaction_ack_word_addr,  // Assume for receiver channel, this address points to a chunk of memory
                                               // that can fit 2 eth_channel_syncs cfor ack
        uint8_t channel_id) :
        buffer_size_in_bytes(buffer_size_bytes),
        max_eth_payload_size_in_bytes(buffer_size_in_bytes),
        channel_id(channel_id) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] = channel_base_address + i * this->max_eth_payload_size_in_bytes;
        }
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

private:
    std::array<size_t, NUM_BUFFERS> buffer_addresses;

    // header + payload regions only
    const std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    const std::size_t max_eth_payload_size_in_bytes;
    uint8_t channel_id;
};

template <uint8_t NUM_BUFFERS>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        remote_producer_wrptr(nullptr),
        connection_live_semaphore(nullptr),
        sender_sync_noc_cmd_buf(write_at_cmd_buf),
        local_wrptr(),
        local_ackptr(),
        local_rdptr() {}
    EdmChannelWorkerInterface(
        // TODO: PERF: See if we can make this non-volatile and then only
        // mark it volatile when we know we need to reload it (i.e. after we receive a
        // "done" message from sender)
        // Have a volatile update function that only triggers after reading the volatile
        // completion field so that way we don't have to do a volatile read for every
        // packet... Then we'll also be able to cache the uint64_t addr of the worker
        // semaphore directly (saving on regenerating it each time)
        volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t* const remote_producer_wrptr,
        volatile tt_l1_ptr uint32_t* const connection_live_semaphore,
        uint8_t sender_sync_noc_cmd_buf) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        remote_producer_wrptr(remote_producer_wrptr),
        connection_live_semaphore(connection_live_semaphore),
        sender_sync_noc_cmd_buf(sender_sync_noc_cmd_buf),
        local_wrptr(),
        local_ackptr(),
        local_rdptr() {
        DPRINT << "EDM  my_x: " << (uint32_t)my_x[0] << ", my_y: " << (uint32_t)my_y[0] << " rdptr set to 0 at "
               << (uint32_t)(void*)&(worker_location_info_ptr->edm_rdptr) << "\n";
        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_rdptr)) = 0;
    }

    // Flow control methods
    //
    // local_wrptr trails from_remote_wrptr
    // we have new data if they aren't equal
    [[nodiscard]] FORCE_INLINE bool has_unsent_payload() { return local_wrptr.get_ptr() != *remote_producer_wrptr; }
    [[nodiscard]] FORCE_INLINE bool has_unacked_sends() { return local_ackptr.get_ptr() != local_wrptr.get_ptr(); }

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }

    template <bool enable_ring_support>
    FORCE_INLINE void update_worker_copy_of_read_ptr(BufferPtr new_ptr_val) {
        noc_inline_dw_write<false, true>(
            this->cached_worker_semaphore_address, new_ptr_val, 0xf, tt::tt_fabric::worker_handshake_noc);
    }

    // Connection management methods
    //
    template <bool posted = false>
    FORCE_INLINE void teardown_connection(uint32_t last_edm_rdptr_value) const {
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x,
            (uint32_t)worker_info.worker_xy.y,
            worker_info.worker_teardown_semaphore_address);

        // Set connection to unused so it's available for next worker
        *this->connection_live_semaphore = tt::tt_fabric::EdmToEdmSender<0>::unused_connection_value;

        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_rdptr)) = last_edm_rdptr_value;

        noc_semaphore_inc<posted>(worker_semaphore_address, 1, tt::tt_fabric::worker_handshake_noc);
    }

    FORCE_INLINE void cache_producer_noc_addr() {
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }

    FORCE_INLINE bool all_eth_packets_acked() const { return this->local_ackptr.is_caught_up_to(this->local_wrptr); }
    FORCE_INLINE bool all_eth_packets_completed() const { return this->local_rdptr.is_caught_up_to(this->local_wrptr); }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        return *connection_live_semaphore == tt::tt_fabric::EdmToEdmSender<0>::close_connection_request_value;
    }
    [[nodiscard]] FORCE_INLINE bool connection_is_live() const {
        return *connection_live_semaphore == tt::tt_fabric::EdmToEdmSender<0>::open_connection_value;
    }

    volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t* const remote_producer_wrptr;
    volatile tt_l1_ptr uint32_t* const connection_live_semaphore;
    uint8_t sender_sync_noc_cmd_buf;

    ChannelBufferPointer<NUM_BUFFERS> local_wrptr;
    ChannelBufferPointer<NUM_BUFFERS> local_ackptr;
    ChannelBufferPointer<NUM_BUFFERS> local_rdptr;  // also used as completion_ptr
};

}  // namespace tt::tt_fabric
