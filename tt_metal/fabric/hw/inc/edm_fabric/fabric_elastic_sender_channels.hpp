// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_outbound_sender_channel_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/datastructures/fabric_circular_buffer.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/datastructures/one_pass_iterator.hpp"
#include "fabric_edm_packet_header.hpp"
#include "fabric_erisc_datamover_channels.hpp"

#include "risc_attribs.h"

namespace tt::tt_fabric {

template <size_t CHUNK_N_PKTS, typename ChannelBuffersPoolT>
struct ElasticSenderChannel : public SenderEthChannelInterface<ElasticSenderChannel<CHUNK_N_PKTS>> {
    using chunk_t = EthChannelBuffer<PACKET_HEADER_TYPE, CHUNK_N_PKTS>;
    using chunk_iterator_t tt::tt_fabric::OnePassIteratorStaticSizes<uint32_t, CHUNK_N_PKTS, PACKET_SIZE_BYTES / sizeof(uint32_t)>;

    // A container for all the chunks in use by this sender channel.
    CircularBuffer<chunk_t*, REQUESTED_SIZE> open_chunks_window;
    
    // This iterator is responsible for iterating through the "leading" sender channel
    // chunk and advances as new packets are sent over ethernet (i.e. in `send_next_data`)
    chunk_iterator_t sending_chunk_iterator;

    // This iterator is responsible for iterating through the "trailing" sender channel
    // chunk and advances as completion acks are received.
    chunk_iterator_t completion_chunk_iterator;

    std::size_t cached_next_buffer_slot_addr;

    ChannelBuffersPoolT *channel_buffers_pool;

    // Send Side APIs (these map to the `SenderEthChannel` type)
    FORCE_INLINE void init_impl(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        static_assert(false, "Unimplemented");
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const {
        return this->cached_next_buffer_slot_addr;
        static_assert(false, "Unimplemented");
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr_impl() {
        static_assert(false, "Unimplemented");
        sending_chunk_iterator.increment();
        if (sending_chunk_iterator.is_done()) {
            
        }
    }


    // EdmChannelWorkerInterface API
    // Current, only enable elastic channels with persistent (i.e. router <-> router) connections
    // update_persistent_connection_copy_of_free_slots

    // Future work - enable elastic channels with dynamic (i.e. worker <-> router) connections
    // -> this will be moved to send side and requires updates to the dynamic connection protocol
    // notify_worker_of_read_counter_update 
    

    // Channel Buffers Pool API
    // FORCE_INLINE chunk_t* get_free_chunk();
    // FORCE_INLINE void return_chunk(chunk_t* chunk) { free_chunks.push(chunk); }
    // FORCE_INLINE bool is_empty() const { return free_chunks.is_empty(); }
    // FORCE_INLINE bool is_full() const { return free_chunks.is_full(); }
};

}  // namespace tt::tt_fabric