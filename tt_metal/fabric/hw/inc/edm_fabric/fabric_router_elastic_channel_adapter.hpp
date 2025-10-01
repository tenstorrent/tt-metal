// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "dataflow_api.h"

// #include "risc_common.h"
// #include "fabric_stream_regs.hpp"
// #include "fabric_edm_types.hpp"
// // #include "hostdevcommon/fabric_common.h"
// #include "edm_fabric_flow_control_helpers.hpp"
// #include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
// #include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
// #include "fabric_edm_packet_header_validate.hpp"
// #include "tt_metal/hw/inc/utils/utils.h"
// #include "debug/assert.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/datastructures/fabric_circular_buffer.hpp"
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

    FORCE_INLINE uint64_t get_current_dest_address() const {
        return destination_noc_addr;
    }
    FORCE_INLINE uint32_t get_current_dest_address_hi() const {
        return destination_noc_addr >> 32;
    }
    FORCE_INLINE uint32_t get_current_dest_address_lo() const {
        return static_cast<uint32_t>(destination_noc_addr);
    }

    FORCE_INLINE void advance() {
        destination_noc_addr += CHUNK_SIZE_BYTES;
    }
    
    FORCE_INLINE bool is_full() const {
        return destination_noc_addr == destination_chunk_last_addr;
    }
};

template <uint8_t SLOTS_PER_CHUNK, uint8_t CHUNK_SIZE_BYTES>
struct RouterElasticChannelWriterAdapter {
    ActiveDestinationChunk<SLOTS_PER_CHUNK, CHUNK_SIZE_BYTES> active_destination_chunk; 

    FORCE_INLINE void init(
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_connection_handshake_l1_id,
        std::size_t edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
        volatile uint32_t* const worker_teardown_addr,
        uint32_t sender_channel_credits_stream_id,  // To update the downstream EDM's free slots. Sending worker or edm
                                                    // decrements over noc.

        // TODO: get the location where we see new address information from the downstream EDM
        uint8_t data_noc_cmd_buf = write_reg_cmd_buf,
        uint8_t sync_noc_cmd_buf = write_at_cmd_buf
    ) {
        static_assert(false, "Unimplemented");
    }


    // SEND_CREDIT_ADDR: True when the EDM sender is IDLE_ETH (mux) as it doesn't have credits on L1 static address
    //                   or some legacy code which skips connection info copy on Tensix L1 static address
    template <bool SEND_CREDIT_ADDR = false, bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
    void open() {
        static_assert(false, "Unimplemented");
        open_start<SEND_CREDIT_ADDR, posted, WORKER_HANDSHAKE_NOC>();
        open_finish<posted, WORKER_HANDSHAKE_NOC>();
    }
    void close() {
        static_assert(false, "Unimplemented");
        close_start();
        close_finish();
    }

    // TODO: rename to `consumer_has_space_for_packet`. Kept as `edm_has_space_for_packet` for
    // now to avoid renaming everywhere, which will make the elastic channel changes less isolated
    bool edm_has_space_for_packet() const {
        return !active_destination_chunk.is_full();
    }

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
            const uint64_t noc_sem_addr =
                get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_remote_free_slots_update_addr, noc);
            noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, packed_val, 0xf, noc);
        }
        // Write to the atomic increment stream register (write of -1 will subtract 1)
        increment_local_update_ptr_val(worker_credits_stream_id, -1);
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
    FORCE_INLINE void send_payload_with_trid(
        uint32_t source_address, size_t size_bytes, uint8_t trid) {
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
            this->update_edm_buffer_free_slots<stateful_api, enable_deadlock_avoidance, vc1_has_different_downstream_dest>(
                EDM_TO_DOWNSTREAM_NOC);
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



private:
    mutable uint64_t noc_sem_addr_;
    std::array<uint32_t, EDM_NUM_BUFFER_SLOTS> edm_buffer_slot_addrs;

    uint32_t worker_credits_stream_id;
    // Local copy of the the free slots on the downstream router
    // Downstream router will increment this when it frees up a slot
    volatile tt_reg_ptr uint32_t* edm_buffer_local_free_slots_read_ptr;
    volatile tt_reg_ptr uint32_t* edm_buffer_local_free_slots_update_ptr;
    size_t edm_buffer_remote_free_slots_update_addr;
    size_t edm_connection_handshake_l1_addr;
    size_t edm_worker_location_info_addr;
    // Note that for persistent (fabric to fabric connections), this only gets read once and actually points to the free
    // slots addr
    size_t edm_copy_of_wr_counter_addr;

    volatile tt_l1_ptr uint32_t* worker_teardown_addr;

    uint16_t buffer_size_bytes;

    BufferIndex buffer_slot_index;

    // noc location of the edm we are connected to (where packets are sent to)
    uint8_t edm_noc_x;
    uint8_t edm_noc_y;

    // the cmd buffer is used for edm-edm path
    uint8_t data_noc_cmd_buf;
    uint8_t sync_noc_cmd_buf;

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
            const uint64_t noc_sem_addr =
                get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_remote_free_slots_update_addr, noc);
            noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, packed_val, 0xf, noc);
        }
        // Write to the atomic increment stream register (write of -1 will subtract 1)
        increment_local_update_ptr_val(worker_credits_stream_id, -1);
    }

    FORCE_INLINE uint8_t get_buffer_slot_index() const { return this->buffer_slot_index.get(); }

    FORCE_INLINE void advance_buffer_slot_write_index() {
        this->buffer_slot_index = BufferIndex{wrap_increment<EDM_NUM_BUFFER_SLOTS>(this->buffer_slot_index.get())};
    }

    template <
        bool stateful_api = false,
        bool enable_deadlock_avoidance = false,
        bool vc1_has_different_downstream_dest = false>
    FORCE_INLINE void post_send_payload_increment_pointers(uint8_t noc = noc_index) {
        this->advance_buffer_slot_write_index();
        this->update_edm_buffer_free_slots<stateful_api, enable_deadlock_avoidance, vc1_has_different_downstream_dest>(
            noc);
    }

    template <
        bool enable_deadlock_avoidance,
        bool vc1_has_different_downstream_dest,
        uint8_t EDM_TO_DOWNSTREAM_NOC,
        bool stateful_api,
        bool increment_pointers>
    FORCE_INLINE void send_payload_from_address_with_trid_impl(
        uint32_t source_address, size_t size_bytes, uint8_t trid) {
        ASSERT(size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));

        send_chunk_from_address_with_trid<stateful_api, vc1_has_different_downstream_dest>(
            source_address,
            size_bytes,
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0) >> NOC_ADDR_COORD_SHIFT,
            this->edm_buffer_slot_addrs[this->get_buffer_slot_index()],
            trid,
            EDM_TO_DOWNSTREAM_NOC,
            this->data_noc_cmd_buf);

        if constexpr (increment_pointers) {
            post_send_payload_increment_pointers<
                stateful_api,
                enable_deadlock_avoidance,
                vc1_has_different_downstream_dest>(EDM_TO_DOWNSTREAM_NOC);
        }
    }
};

template <uint8_t EDM_SENDER_CHANNEL_NUM_BUFFERS>
using EdmToEdmSender = RouterElasticChannelWriterAdapter<EDM_SENDER_CHANNEL_NUM_BUFFERS>;

}  // namespace tt::tt_fabric
