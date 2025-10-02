// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/datastructures/fabric_circular_buffer.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/adapters/fabric_adapter_utils.hpp"
#include "risc_attribs.h"

#include <cstdint>
#include <array>

namespace tt::tt_fabric {

/*
 * The FabricToFabricSender acts as an adapter between two routers in the fabric. It hides details
 * of the communication between the inbound (from Ethernet) and the downstream outbound (to Ethernet)
 * routers.
 * The main functionality provided is:
 * - Opening a connection with the router
 * - Closing a connection with the router
 * - Flow control protocol between the two routers
 *
 * ### Flow Control Protocol:
 * The flow control protocol is not counter or pointer based. Instead, credits indicate a change in the number of
 * slots consumed or freed.
 * For example, the consumer tracks the number of free slots available. The producer, after writing a payload, will
 * issue an atomic decrement to the consumer's free slots counter. The consumer will see its free slots count decrement
 * and will understand that a new payload has been received.
 *
 * The consumer when issuing an acknowledgement, will increment the producer's free slots counter.
 */
template <uint8_t SLOTS_PER_CHUNK, uint8_t CHUNK_SIZE_BYTES>
struct ActiveDestinationChunk {
    ActiveDestinationChunk() = default;

    uint64_t destination_noc_addr = 0;

    // if destination_noc_addr == destination_chunk_last_addr, this chunk is exhausted
    uint64_t destination_chunk_last_addr = 0;

    FORCE_INLINE void init(uint64_t destination_noc_addr) {
        this->destination_noc_addr = destination_noc_addr;
        this->destination_chunk_last_addr = destination_noc_addr + CHUNK_SIZE_BYTES * SLOTS_PER_CHUNK;
    }

    FORCE_INLINE uint64_t get_current_dest_address() const { return destination_noc_addr; }
    FORCE_INLINE uint32_t get_current_dest_address_hi() const { return destination_noc_addr >> 32; }
    FORCE_INLINE uint32_t get_current_dest_address_lo() const { return static_cast<uint32_t>(destination_noc_addr); }

    FORCE_INLINE void advance() { destination_noc_addr += CHUNK_SIZE_BYTES; }

    FORCE_INLINE bool is_full() const { return destination_noc_addr == destination_chunk_last_addr; }
};

using destination_chunk_allocation_t = uint32_t;  // bank_address;

template <uint8_t SLOTS_PER_CHUNK, uint8_t CHUNK_SIZE_BYTES>
struct RouterElasticChannelWriterAdapter {
    FORCE_INLINE void init(
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_connection_handshake_l1_id,
        std::size_t router_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
        volatile uint32_t* const worker_teardown_addr,
        uint32_t sender_channel_credits_stream_id,  // To update the downstream EDM's free slots. Sending worker or edm
                                                    // decrements over noc.

        // TODO: get the location where we see new address information from the downstream EDM
        uint8_t data_noc_cmd_buf = write_reg_cmd_buf,
        uint8_t sync_noc_cmd_buf = write_at_cmd_buf) {
        this->router_worker_location_info_addr = router_worker_location_info_addr;
        this->my_connection_teardown_addr = worker_teardown_addr;
        this->router_connection_handshake_l1_addr = edm_connection_handshake_l1_id;
        static_assert(false, "Unimplemented");
    }

    // SEND_CREDIT_ADDR: True when the EDM sender is IDLE_ETH (mux) as it doesn't have credits on L1 static address
    //                   or some legacy code which skips connection info copy on Tensix L1 static address
    template <bool SEND_CREDIT_ADDR = false, bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
    void open() {
        static_assert(false, "Unimplemented");
        open_start<SEND_CREDIT_ADDR, posted, WORKER_HANDSHAKE_NOC>(
            this->router_worker_location_info_addr,
            static_cast<size_t>(this->destination_chunk_address_ptr),
            this->my_connection_teardown_addr,
            this->router_noc_x,
            this->router_noc_y);
        open_finish<posted, WORKER_HANDSHAKE_NOC>(
            this->router_connection_handshake_l1_addr,
            this->my_connection_teardown_addr,
            this->router_noc_x,
            this->router_noc_y);
    }

    // TODO: rename to `consumer_has_space_for_packet`. Kept as `edm_has_space_for_packet` for
    // now to avoid renaming everywhere, which will make the elastic channel changes less isolated
    bool edm_has_space_for_packet() const { return !active_destination_chunk.is_full(); }

    ///
    // TODO: commonize with `update_edm_buffer_free_slots` from the `RouterChannelWriterAdapter`
    ///
    template <
        bool stateful_api = false,
        bool enable_deadlock_avoidance = false,
        bool vc1_has_different_downstream_dest = false>
    FORCE_INLINE void update_edm_buffer_free_slots(uint8_t noc = noc_index) {
        if constexpr (stateful_api) {
            if constexpr (enable_deadlock_avoidance) {
                if constexpr (vc1_has_different_downstream_dest) {
                    auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
                    noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr_, packed_val, 0xf, noc);
                } else {
                    noc_inline_dw_write_with_state<true, false, true>(
                        0,  // val unused
                        this->edm_buffer_remote_free_slots_update_addr,
                        this->sync_noc_cmd_buf,
                        noc);
                }
            } else {
                noc_inline_dw_write_with_state<false, false, true>(
                    0,  // val unused
                    0,  // addr unused
                    this->sync_noc_cmd_buf,
                    noc);
            }
        } else {
            auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
            const uint64_t noc_sem_addr = get_noc_addr(
                this->router_noc_x, this->router_noc_y, this->edm_buffer_remote_free_slots_update_addr, noc);
            noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, packed_val, 0xf, noc);
        }
        // Write to the atomic increment stream register (write of -1 will subtract 1)
        // snijjar - delete
        // increment_local_update_ptr_val(worker_credits_stream_id, -1);
    }

    ///
    // TODO: commonize with `send_payload_from_address_with_trid_impl` from the `RouterChannelWriterAdapter`
    ///
    template <
        bool enable_deadlock_avoidance,
        bool vc1_has_different_downstream_dest,
        uint8_t EDM_TO_DOWNSTREAM_NOC,
        bool stateful_api,
        bool increment_pointers>
    FORCE_INLINE void send_payload_with_trid(uint32_t source_address, size_t size_bytes, uint8_t trid) {
        ASSERT(size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));

        send_chunk_from_address_with_trid<stateful_api, vc1_has_different_downstream_dest>(
            source_address,
            size_bytes,
            active_destination_chunk.get_current_dest_address_hi(),
            active_destination_chunk.get_current_dest_address_lo(),
            trid,
            EDM_TO_DOWNSTREAM_NOC,
            this->data_noc_cmd_buf);

        if constexpr (increment_pointers) {
            this->active_destination_chunk.advance();
            this->update_edm_buffer_free_slots<
                stateful_api,
                enable_deadlock_avoidance,
                vc1_has_different_downstream_dest>(EDM_TO_DOWNSTREAM_NOC);
        }
    }

    // TODO: commonize with `update_edm_buffer_slot_word` from the `RouterChannelWriterAdapter`
    template <bool inc_pointers = true>
    FORCE_INLINE void update_edm_buffer_slot_word(uint32_t offset, uint32_t data, uint8_t noc = noc_index) {
        noc_inline_dw_write(this->active_destination_chunk.get_current_dest_address(), data, 0xf, noc);
        if constexpr (inc_pointers) {
            post_send_payload_increment_pointers(noc);
        }
    }

    FORCE_INLINE bool new_chunk_is_available() {
        // shouldn't be calling this when the chunk is not full
        // We currently only support one unwritten chunk at a time
        ASSERT(this->active_destination_chunk.is_full());
        return *destination_chunk_address_ptr;
    }

    // consider merging with above to limit number of loads
    FORCE_INLINE void get_next_chunk() {
        // shouldn't be calling this when the chunk is not full
        // We currently only support one unwritten chunk at a time
        ASSERT(this->active_destination_chunk.is_full());
        ASSERT(this->new_chunk_is_available());
        this->active_destination_chunk.init(*destination_chunk_address_ptr);
    }

private:
    // local copy of the "credit" written to by the downstream router
    volatile tt_reg_ptr destination_chunk_allocation_t* destination_chunk_address_ptr;
    ActiveDestinationChunk<SLOTS_PER_CHUNK, CHUNK_SIZE_BYTES> active_destination_chunk;
    volatile tt_l1_ptr uint32_t* my_connection_teardown_addr;
    mutable uint64_t noc_sem_addr_;  // address on the downstream router

    size_t edm_buffer_remote_free_slots_update_addr;
    size_t router_worker_location_info_addr;

    // TODO: work on removing this so we can trim down the struct size
    // this isn't needed during steady-state operation
    size_t router_connection_handshake_l1_addr;

    // noc location of the edm we are connected to (where packets are sent to)
    uint8_t router_noc_x;
    uint8_t router_noc_y;

    // the cmd buffer is used for edm-edm path
    uint8_t data_noc_cmd_buf;
    uint8_t sync_noc_cmd_buf;
};

}  // namespace tt::tt_fabric
