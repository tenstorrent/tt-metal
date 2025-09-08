// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

#include "risc_common.h"
#include "fabric_stream_regs.hpp"
#include "fabric_edm_types.hpp"
// #include "hostdevcommon/fabric_common.h"
#include "edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "fabric_edm_packet_header_validate.hpp"
#include "tt_metal/hw/inc/utils/utils.h"
#include "debug/assert.h"

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
template <uint8_t EDM_NUM_BUFFER_SLOTS>
struct RouterElasticChannelWriterAdapter {
    static constexpr bool ENABLE_STATEFUL_WRITE_CREDIT_TO_DOWNSTREAM_EDM =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
        true;
#else
        false;
#endif
    // Temporary flag to distinguish between worker and EDM users of this adapter until we split it into
    // two separate adapters (they've started diverging quite a bit by now)
    //  --> Not splitting yet though to avoid change conflict issues with some other in flight changes
    static constexpr bool IS_POW2_NUM_BUFFERS = is_power_of_2(EDM_NUM_BUFFER_SLOTS);
    static constexpr size_t BUFFER_SLOT_PTR_WRAP = EDM_NUM_BUFFER_SLOTS * 2;
    static constexpr uint32_t unused_connection_value = 0;
    static constexpr uint32_t open_connection_value = 1;
    static constexpr uint32_t close_connection_request_value = 2;
    // HACK: Need a way to properly set this up

    RouterElasticChannelWriterAdapter() = default;

    template <ProgrammableCoreType my_core_type = ProgrammableCoreType::ACTIVE_ETH>
    FORCE_INLINE void init(
        bool connected_to_persistent_fabric,
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_buffer_base_addr,
        uint8_t /*num_buffers_per_channel*/,  // unused
        std::size_t edm_connection_handshake_l1_id,
        std::size_t edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
        uint16_t buffer_size_bytes,
        size_t edm_buffer_index_id,
        volatile uint32_t* const
            from_remote_buffer_free_slots_ptr,  // For worker to locally track downstream EDM's read counter. Only used
                                                // by Worker. Downstream EDM increments over noc when a slot is freed.
        volatile uint32_t* const worker_teardown_addr,
        uint32_t local_buffer_index_addr,
        uint32_t sender_channel_credits_stream_id,  // To update the downstream EDM's free slots. Sending worker or edm
                                                    // decrements over noc.
        StreamId
            worker_credits_stream_id,  // To locally track downstream EDM's free slots. Only used by EDM. Sending EDM
                                       // decrements locally. Downstream EDM increments over noc when a slot is freed.
        uint8_t data_noc_cmd_buf = write_reg_cmd_buf,
        uint8_t sync_noc_cmd_buf = write_at_cmd_buf) {
        this->worker_credits_stream_id = worker_credits_stream_id.get();

        this->edm_buffer_local_free_slots_read_ptr =
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_stream_reg_read_addr(this->worker_credits_stream_id));
        this->edm_buffer_remote_free_slots_update_addr = get_stream_reg_write_addr(sender_channel_credits_stream_id);
        this->edm_buffer_local_free_slots_update_ptr =
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(get_stream_reg_write_addr(this->worker_credits_stream_id));
        this->edm_connection_handshake_l1_addr = connected_to_persistent_fabric
                                                     ? edm_connection_handshake_l1_id
                                                     : get_semaphore<my_core_type>(edm_connection_handshake_l1_id);
        this->edm_worker_location_info_addr = edm_worker_location_info_addr;
        this->edm_copy_of_wr_counter_addr =
            connected_to_persistent_fabric ? edm_buffer_index_id : get_semaphore<my_core_type>(edm_buffer_index_id);
        this->worker_teardown_addr = worker_teardown_addr;
        this->buffer_size_bytes = buffer_size_bytes;
        this->edm_noc_x = edm_worker_x;
        this->edm_noc_y = edm_worker_y;
        this->data_noc_cmd_buf = data_noc_cmd_buf;
        this->sync_noc_cmd_buf = sync_noc_cmd_buf;

        // The EDM is guaranteed to know the number of free slots of the downstream EDM
        // becausen all EDMs are brought up/initialized at the same time
        init_ptr_val(this->worker_credits_stream_id, EDM_NUM_BUFFER_SLOTS);
        for (size_t i = 0; i < EDM_NUM_BUFFER_SLOTS; ++i) {
            this->edm_buffer_slot_addrs[i] = edm_buffer_base_addr + (i * this->buffer_size_bytes);
        }
    }

    template <ProgrammableCoreType my_core_type = ProgrammableCoreType::ACTIVE_ETH>
    FORCE_INLINE RouterElasticChannelWriterAdapter(
        bool connected_to_persistent_fabric,
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_buffer_base_addr,
        uint8_t num_buffers_per_channel,
        std::size_t edm_connection_handshake_l1_id,
        std::size_t edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
        uint16_t buffer_size_bytes,
        size_t edm_buffer_index_id,
        volatile uint32_t* const from_remote_buffer_free_slots_ptr,
        volatile uint32_t* const worker_teardown_addr,
        uint32_t local_buffer_index_addr,
        uint32_t sender_channel_credits_stream_id,
        StreamId worker_credits_stream_id,
        uint8_t data_noc_cmd_buf = write_reg_cmd_buf,
        uint8_t sync_noc_cmd_buf = write_at_cmd_buf) {
        this->init<my_core_type>(
            connected_to_persistent_fabric,
            edm_worker_x,
            edm_worker_y,
            edm_buffer_base_addr,
            num_buffers_per_channel,
            edm_connection_handshake_l1_id,
            edm_worker_location_info_addr,
            buffer_size_bytes,
            edm_buffer_index_id,
            from_remote_buffer_free_slots_ptr,
            worker_teardown_addr,
            local_buffer_index_addr,
            sender_channel_credits_stream_id,
            worker_credits_stream_id,
            data_noc_cmd_buf,
            sync_noc_cmd_buf);
    }

    static constexpr size_t edm_sender_channel_field_stride_bytes = 16;

    template <bool SEND_CREDIT_ADDR = false, bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
    void open_start() {
        const auto dest_noc_addr_coord_only = get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0);

        tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr =
            reinterpret_cast<tt::tt_fabric::EDMChannelWorkerLocationInfo*>(edm_worker_location_info_addr);

        const uint64_t dest_edm_location_info_addr =
            dest_noc_addr_coord_only |
            reinterpret_cast<size_t>(
                edm_worker_location_info_addr +
                offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, worker_semaphore_address));
        // write the address of our local copy of read counter (that EDM is supposed to update)

        noc_inline_dw_write<InlineWriteDst::DEFAULT, posted>(
            dest_edm_location_info_addr,
            reinterpret_cast<size_t>(edm_buffer_local_free_slots_update_ptr),
            0xf,
            WORKER_HANDSHAKE_NOC);
        const uint64_t edm_teardown_semaphore_address_address =
            dest_noc_addr_coord_only |
            reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_teardown_semaphore_address));
        // Write our local teardown ack address to EDM
        noc_inline_dw_write<InlineWriteDst::DEFAULT, posted>(
            edm_teardown_semaphore_address_address,
            reinterpret_cast<size_t>(worker_teardown_addr),
            0xf,
            WORKER_HANDSHAKE_NOC);
        // Write out core noc-xy coord to EDM
        const uint64_t connection_worker_xy_address =
            dest_noc_addr_coord_only | reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_xy));
        noc_inline_dw_write<InlineWriteDst::DEFAULT, posted>(
            connection_worker_xy_address, WorkerXY(my_x[0], my_y[0]).to_uint32(), 0xf, WORKER_HANDSHAKE_NOC);
    }

    template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
    void open_finish() {
        const uint64_t edm_connection_handshake_noc_addr =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, edm_connection_handshake_l1_addr);
        noc_async_read_barrier(WORKER_HANDSHAKE_NOC);
        // Order here is important
        // We need to write our read counter value to the register before we signal the EDM
        // As EDM will potentially increment the register as well
        this->buffer_slot_index = BufferIndex(0);

        noc_inline_dw_write<InlineWriteDst::DEFAULT, posted>(
            edm_connection_handshake_noc_addr, open_connection_value, 0xf, WORKER_HANDSHAKE_NOC);
        *this->worker_teardown_addr = 0;
    }

    void close_start() {
        const auto dest_noc_addr_coord_only =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0) & ~(uint64_t)NOC_COORDINATE_MASK;

        // buffer index stored at location after handshake addr
        const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
        noc_inline_dw_write(remote_buffer_index_addr, this->get_buffer_slot_index());

        const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write(dest_edm_connection_state_addr, close_connection_request_value);
    }

    void close_finish() {
        WAYPOINT("FCFW");
        // Need to wait for the ack to teardown notice, from edm
        while (*this->worker_teardown_addr != 1) {
            invalidate_l1_cache();
        }
        WAYPOINT("FCFD");
        noc_async_write_barrier();
        *(this->worker_teardown_addr) = 0;
    }

public:
    template <uint8_t EDM_TO_DOWNSTREAM_NOC = noc_index, uint8_t EDM_TO_DOWNSTREAM_NOC_VC = NOC_UNICAST_WRITE_VC>
    FORCE_INLINE void setup_edm_noc_cmd_buf() const {
        uint64_t edm_noc_addr = get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0, EDM_TO_DOWNSTREAM_NOC);
        noc_async_write_one_packet_with_trid_set_state<true>(
            edm_noc_addr, this->data_noc_cmd_buf, EDM_TO_DOWNSTREAM_NOC, EDM_TO_DOWNSTREAM_NOC_VC);
        const uint64_t noc_sem_addr = get_noc_addr(
            this->edm_noc_x, this->edm_noc_y, this->edm_buffer_remote_free_slots_update_addr, EDM_TO_DOWNSTREAM_NOC);
        noc_sem_addr_ = noc_sem_addr;
        noc_inline_dw_write_set_state<true, true>(
            noc_sem_addr,
            (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC,
            0xF,
            this->sync_noc_cmd_buf,
            EDM_TO_DOWNSTREAM_NOC,
            EDM_TO_DOWNSTREAM_NOC_VC);
    }

    FORCE_INLINE bool edm_has_space_for_packet() const {
        invalidate_l1_cache();
        return *this->edm_buffer_local_free_slots_read_ptr != 0;
    }

    template <
        bool enable_deadlock_avoidance,
        bool vc1_has_different_downstream_dest,
        uint8_t EDM_TO_DOWNSTREAM_NOC,
        bool stateful_api,
        bool increment_pointers>
    FORCE_INLINE void send_payload_non_blocking_from_address_with_trid(
        uint32_t source_address, size_t size_bytes, uint8_t trid) {
        send_payload_from_address_with_trid_impl<
            enable_deadlock_avoidance,
            vc1_has_different_downstream_dest,
            EDM_TO_DOWNSTREAM_NOC,
            stateful_api,
            increment_pointers>(source_address, size_bytes, trid);
    }

    template <bool inc_pointers = true>
    FORCE_INLINE void update_edm_buffer_slot_word(uint32_t offset, uint32_t data, uint8_t noc = noc_index) {
        uint64_t noc_addr;
        noc_addr = get_noc_addr(
            this->edm_noc_x, this->edm_noc_y, this->edm_buffer_slot_addrs[this->get_buffer_slot_index()] + offset, noc);

        noc_inline_dw_write(noc_addr, data, 0xf, noc);
        if constexpr (inc_pointers) {
            post_send_payload_increment_pointers(noc);
        }
    }

    // SEND_CREDIT_ADDR: True when the EDM sender is IDLE_ETH (mux) as it doesn't have credits on L1 static address
    //                   or some legacy code which skips connection info copy on Tensix L1 static address
    template <bool SEND_CREDIT_ADDR = false, bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
    void open() {
        open_start<SEND_CREDIT_ADDR, posted, WORKER_HANDSHAKE_NOC>();
        open_finish<posted, WORKER_HANDSHAKE_NOC>();
    }
    void close() {
        close_start();
        close_finish();
    }

    // Only for debug/watcher asserts
    FORCE_INLINE uint32_t get_worker_credits_stream_id() const { return this->worker_credits_stream_id; }

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
                    noc_inline_dw_write<InlineWriteDst::REG>(
                        noc_sem_addr_, (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xf, noc);
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
            const uint64_t noc_sem_addr =
                get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_remote_free_slots_update_addr, noc);
            noc_inline_dw_write<InlineWriteDst::REG>(noc_sem_addr, (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xf, noc);
        }
        // Write to the atomic increment stream register (write of -1 will subtract 1)
        *edm_buffer_local_free_slots_update_ptr = (-1) << REMOTE_DEST_BUF_WORDS_FREE_INC;
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
            1,
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
