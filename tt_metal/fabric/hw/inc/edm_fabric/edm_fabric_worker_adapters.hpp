// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include "internal/tt-1xx/risc_common.h"
#include "internal/ethernet/dataflow_api.h"
#include "edm_fabric_utils.hpp"
#include "fabric_edm_packet_header_validate.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "fabric_stream_regs.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "hostdevcommon/fabric_common.h"
#include "edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "api/alignment.h"
#include "api/debug/assert.h"

#include <cstdint>
#include <array>

namespace tt::tt_fabric {

/*
 * The WorkerToFabricEdmSenderImpl acts as an adapter between the worker and the EDM, it hides details
 * of the communication between worker and EDM to provide flexibility for the implementation to change
 * over time without kernel updates. Additionally, details for adapter setup w.r.t runtime args is also hidden.
 * The main functionality provided is:
 * - Opening a connection with the EDM
 * - Closing a connection with the EDM
 * - Flow control protocol between worker and EDM
 *
 * ### Flow Control Protocol:
 * The flow control protocol is rd/wr ptr based and is implemented as follows (from the worker's perspective):
 * The adapter has a local write pointer (wrptr) which is used to track the next buffer slot to write to. The adapter
 * also has a local memory slot that holds the remote read pointer (rdptr) of the EDM. The adapter uses the difference
 * between these two pointers (where rdptr trails wrptr) to determine if the EDM has space to accept a new packet.
 *
 * As the adapter writes into the EDM, it updates the local wrptr. As the EDM reads from its local L1 channel buffer,
 * it will notify the worker/adapter (here) by updating the worker remote_rdptr to carry the value of the EDM rdptr.
 */
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS = 0>
struct WorkerToFabricEdmSenderImpl {
    static constexpr bool ENABLE_STATEFUL_WRITE_CREDIT_TO_DOWNSTREAM_EDM =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
        true;
#else
        false;
#endif
    static constexpr bool USER_DEFINED_NUM_BUFFER_SLOTS = EDM_NUM_BUFFER_SLOTS != 0;
    // Temporary flag to distinguish between worker and EDM users of this adapter until we split it into
    // two separate adapters (they've started diverging quite a bit by now)
    //  --> Not splitting yet though to avoid change conflict issues with some other in flight changes
    static constexpr bool IS_WORKER = !I_USE_STREAM_REG_FOR_CREDIT_RECEIVE;
    static constexpr bool IS_POW2_NUM_BUFFERS = USER_DEFINED_NUM_BUFFER_SLOTS && is_power_of_2(EDM_NUM_BUFFER_SLOTS);
    static constexpr size_t BUFFER_SLOT_PTR_WRAP = EDM_NUM_BUFFER_SLOTS * 2;
    // HACK: Need a way to properly set this up

    WorkerToFabricEdmSenderImpl() = default;

    template <ProgrammableCoreType my_core_type>
    static WorkerToFabricEdmSenderImpl build_from_args(std::size_t& arg_idx) {
        constexpr bool is_persistent_fabric = true;
        uint8_t direction;
        uint8_t edm_worker_x;
        uint8_t edm_worker_y;
        uint32_t edm_buffer_base_addr;
        uint8_t num_buffers_per_channel;
        uint32_t edm_l1_sem_id;
        uint32_t edm_connection_handshake_l1_addr;
        uint32_t edm_worker_location_info_addr;
        uint16_t buffer_size_bytes;
        uint32_t edm_copy_of_wr_counter_addr;
        volatile uint32_t* writer_send_sem_addr;
        uint32_t worker_free_slots_stream_id;  // used to update the available buffer slot on the receiving router
                                               // (decrement by 1 from the sending side for each packet)

        // TODO: https://github.com/tenstorrent/tt-metal/issues/24959
        // remove redundant nested constructor to avoid copy
        if constexpr (my_core_type == ProgrammableCoreType::TENSIX) {
            tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
                reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(MEM_TENSIX_FABRIC_CONNECTIONS_BASE);
            uint32_t eth_channel = get_arg_val<uint32_t>(arg_idx++);
            const auto conn = &connection_info->read_only[eth_channel];
            const auto aligned_conn = &connection_info->read_write[eth_channel];
            direction = conn->edm_direction;
            edm_worker_x = conn->edm_noc_x;
            edm_worker_y = conn->edm_noc_y;
            edm_buffer_base_addr = conn->edm_buffer_base_addr;
            num_buffers_per_channel = conn->num_buffers_per_channel;
            edm_connection_handshake_l1_addr = conn->edm_connection_handshake_addr;
            edm_worker_location_info_addr = conn->edm_worker_location_info_addr;
            buffer_size_bytes = conn->buffer_size_bytes;
            edm_copy_of_wr_counter_addr = conn->buffer_index_semaphore_id;
            writer_send_sem_addr = reinterpret_cast<volatile uint32_t*>(
                reinterpret_cast<uintptr_t>(&aligned_conn->worker_flow_control_semaphore));
            worker_free_slots_stream_id = static_cast<uint32_t>(conn->worker_free_slots_stream_id);
        } else {
            // TODO: will be deprecated. currently for ethernet dispatch case
            //       ethernet core need to have same memory mapping as worker
            direction = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
            auto edm_worker_xy = WorkerXY::from_uint32(get_arg_val<uint32_t>(arg_idx++));
            edm_worker_x = edm_worker_xy.x;
            edm_worker_y = edm_worker_xy.y;
            edm_buffer_base_addr = get_arg_val<uint32_t>(arg_idx++);
            num_buffers_per_channel = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
            edm_l1_sem_id = get_arg_val<uint32_t>(arg_idx++);
            edm_connection_handshake_l1_addr = get_arg_val<uint32_t>(arg_idx++);
            edm_worker_location_info_addr = get_arg_val<uint32_t>(arg_idx++);
            buffer_size_bytes = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
            edm_copy_of_wr_counter_addr = get_arg_val<uint32_t>(arg_idx++);
            auto writer_send_sem_id = get_arg_val<uint32_t>(arg_idx++);
            writer_send_sem_addr =
                reinterpret_cast<volatile uint32_t*>(get_semaphore<my_core_type>(writer_send_sem_id));
            worker_free_slots_stream_id = tt::tt_fabric::connection_interface::sender_channel_0_free_slots_stream_id;
        }

        // DEAD CODE
        // Workers don't have a local stream ID, so we set to a placeholder (unused) value until the worker and EDM
        // codepaths are split
        const StreamId my_fc_stream_channel_id = StreamId{std::numeric_limits<uint32_t>::max()};

        auto worker_teardown_sem_addr =
            reinterpret_cast<volatile uint32_t* const>(get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++)));
        const auto worker_buffer_index_semaphore_addr = get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++));
        return WorkerToFabricEdmSenderImpl(
            is_persistent_fabric,
            edm_worker_x,
            edm_worker_y,
            edm_buffer_base_addr,
            num_buffers_per_channel,
            edm_connection_handshake_l1_addr,
            edm_worker_location_info_addr,  // The EDM's location for `EDMChannelWorkerLocationInfo`
            buffer_size_bytes,
            edm_copy_of_wr_counter_addr,
            writer_send_sem_addr,
            worker_teardown_sem_addr,
            worker_buffer_index_semaphore_addr,
            worker_free_slots_stream_id,
            my_fc_stream_channel_id,
            write_reg_cmd_buf,
            write_at_cmd_buf);
    }

    template <ProgrammableCoreType my_core_type = ProgrammableCoreType::ACTIVE_ETH>
    FORCE_INLINE void init(
        bool connected_to_persistent_fabric,
        uint8_t edm_worker_x,
        uint8_t edm_worker_y,
        std::size_t edm_buffer_base_addr,
        uint8_t num_buffers_per_channel,
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
        this->edm_buffer_addr = edm_buffer_base_addr;
        this->worker_credits_stream_id = worker_credits_stream_id.get();

        this->edm_buffer_local_free_slots_read_ptr =
            !I_USE_STREAM_REG_FOR_CREDIT_RECEIVE
                ? reinterpret_cast<volatile tt_reg_ptr uint32_t*>(from_remote_buffer_free_slots_ptr)
                : reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                      get_stream_reg_read_addr(this->worker_credits_stream_id));
        this->edm_buffer_remote_free_slots_update_addr = get_stream_reg_write_addr(sender_channel_credits_stream_id);
        this->edm_buffer_local_free_slots_update_ptr =
            !I_USE_STREAM_REG_FOR_CREDIT_RECEIVE
                ? reinterpret_cast<volatile tt_reg_ptr uint32_t*>(from_remote_buffer_free_slots_ptr)
                : reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                      get_stream_reg_write_addr(this->worker_credits_stream_id));
        this->edm_connection_handshake_l1_addr =
            connected_to_persistent_fabric
                ? edm_connection_handshake_l1_id
                : get_semaphore<my_core_type>(edm_connection_handshake_l1_id);
        ASSERT(is_l1_address(edm_connection_handshake_l1_addr));  // must be a L1 address
        this->edm_worker_location_info_addr = edm_worker_location_info_addr;
        ASSERT(is_l1_address(edm_worker_location_info_addr));  // must be a L1 address
        this->edm_copy_of_wr_counter_addr =
            connected_to_persistent_fabric ? edm_buffer_index_id : get_semaphore<my_core_type>(edm_buffer_index_id);
        ASSERT(is_l1_address(edm_copy_of_wr_counter_addr));  // must be a L1 address
        this->worker_teardown_addr = worker_teardown_addr;
        ASSERT(is_l1_address(reinterpret_cast<size_t>(worker_teardown_addr)));  // must be a L1 address
        this->edm_buffer_base_addr = edm_buffer_base_addr;
        this->buffer_size_bytes = buffer_size_bytes;
        this->num_buffers_per_channel = num_buffers_per_channel;
        this->edm_noc_x = edm_worker_x;
        this->edm_noc_y = edm_worker_y;
        this->data_noc_cmd_buf = data_noc_cmd_buf;
        this->sync_noc_cmd_buf = sync_noc_cmd_buf;

        if constexpr (I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            // The EDM is guaranteed to know the number of free slots of the downstream EDM
            // becausen all EDMs are brought up/initialized at the same time
            init_ptr_val(this->worker_credits_stream_id, EDM_NUM_BUFFER_SLOTS);
        }
        if constexpr (USER_DEFINED_NUM_BUFFER_SLOTS) {
            for (size_t i = 0; i < EDM_NUM_BUFFER_SLOTS; ++i) {
                this->edm_buffer_slot_addrs[i] = this->edm_buffer_base_addr + (i * this->buffer_size_bytes);
            }
        }
    }

    template <ProgrammableCoreType my_core_type = ProgrammableCoreType::ACTIVE_ETH>
    FORCE_INLINE WorkerToFabricEdmSenderImpl(
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

    // templatized num_slots to let callers implement bubble flow control without runtime overheads.
    template <size_t num_slots = 1>
    FORCE_INLINE bool edm_has_space_for_packet() const {
        invalidate_l1_cache();
        if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            auto used_slots = this->buffer_slot_write_counter.counter - *this->edm_buffer_local_free_slots_read_ptr;
            if constexpr (num_slots == 1) {
                return used_slots < this->num_buffers_per_channel;
            } else {
                return used_slots <= this->num_buffers_per_channel - num_slots;
            }
        } else {
            return get_ptr_val(worker_credits_stream_id) >= num_slots;
        }
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        WAYPOINT("FWSW");
        while (!this->edm_has_space_for_packet<1>());
        WAYPOINT("FWSD");
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<EDM_IO_BLOCKING_MODE::BLOCKING>(cb_id, num_pages, page_size);
    }
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes) {
        send_payload_without_header_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_flush_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }
    FORCE_INLINE void send_payload_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::BLOCKING>(source_address, size_bytes);
    }

    /*
     * No CB
     */
    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }

    static constexpr size_t edm_sender_channel_field_stride_bytes = 16;

    // Advanced usage API:
    // Starts the connection opening process but doesn't wait for the process complete. This avoids waiting
    // for the read barrier to complete before returning, saving some cycles for advanced users.
    // !!! IMPORTANT !!!
    // Must be called alongside (before) open_finish().
    template <
        bool SEND_CREDIT_ADDR = false,
        bool posted = false,
        uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void open_start() {
        const auto dest_noc_addr_coord_only = get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0, WORKER_HANDSHAKE_NOC);

        tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr =
            reinterpret_cast<tt::tt_fabric::EDMChannelWorkerLocationInfo*>(edm_worker_location_info_addr);

        if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
            // piggy back off of worker_teardown_addr just to temporarily store the read-back write pointer
            // then once we get it we will use that address for the teardown ack
            // Note this is safe because only the worker can initiate teardown (and it will not do it until)
            // some time atleast after it copied the wrptr out of the worker_teardown_addr
            noc_async_read(
                remote_buffer_index_addr,
                reinterpret_cast<size_t>(this->worker_teardown_addr),
                sizeof(uint32_t),
                WORKER_HANDSHAKE_NOC);

            const uint64_t edm_read_free_slots_or_read_counter_addr =
                dest_noc_addr_coord_only | reinterpret_cast<size_t>(
                                               edm_worker_location_info_addr +
                                               offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter));
            // Read the read/pointer or buffer free slots
            noc_async_read(
                edm_read_free_slots_or_read_counter_addr,
                reinterpret_cast<size_t>(this->edm_buffer_local_free_slots_read_ptr),
                sizeof(uint32_t),  // also want to read the local write counter
                WORKER_HANDSHAKE_NOC);
        }
        const uint64_t dest_edm_location_info_addr =
            dest_noc_addr_coord_only |
            reinterpret_cast<size_t>(
                edm_worker_location_info_addr +
                offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, worker_semaphore_address));
        // write the address of our local copy of read counter (that EDM is supposed to update)
        if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            noc_inline_dw_write<InlineWriteDst::L1, posted>(
                dest_edm_location_info_addr,
                reinterpret_cast<size_t>(edm_buffer_local_free_slots_update_ptr),
                0xf,
                WORKER_HANDSHAKE_NOC);
        } else {
            noc_inline_dw_write<InlineWriteDst::L1, posted>(
                dest_edm_location_info_addr,
                reinterpret_cast<size_t>(edm_buffer_local_free_slots_update_ptr),
                0xf,
                WORKER_HANDSHAKE_NOC);
        }
        const uint64_t edm_teardown_semaphore_address_address =
            dest_noc_addr_coord_only |
            reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_teardown_semaphore_address));
        // Write our local teardown ack address to EDM
        noc_inline_dw_write<InlineWriteDst::L1, posted>(
            edm_teardown_semaphore_address_address,
            reinterpret_cast<size_t>(worker_teardown_addr),
            0xf,
            WORKER_HANDSHAKE_NOC);
        // Write out core noc-xy coord to EDM
        const uint64_t connection_worker_xy_address =
            dest_noc_addr_coord_only | reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_xy));
        noc_inline_dw_write<InlineWriteDst::L1, posted>(
            connection_worker_xy_address, WorkerXY(my_x[0], my_y[0]).to_uint32(), 0xf, WORKER_HANDSHAKE_NOC);
    }

    // Advanced usage API:
    // Completes the connection opening process. Induces a read barrier
    // !!! IMPORTANT !!!
    // Must be called alongside (after) open_start().
    template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void open_finish() {
        const uint64_t edm_connection_handshake_noc_addr =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, edm_connection_handshake_l1_addr, WORKER_HANDSHAKE_NOC);
        noc_async_read_barrier(WORKER_HANDSHAKE_NOC);
        // Order here is important
        // We need to write our read counter value to the register before we signal the EDM
        // As EDM will potentially increment the register as well
        if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            this->buffer_slot_write_counter.reset();
            this->buffer_slot_write_counter.counter = *this->worker_teardown_addr;
            this->buffer_slot_write_counter.index = BufferIndex{static_cast<uint8_t>(this->buffer_slot_write_counter.counter % static_cast<uint32_t>(this->num_buffers_per_channel))};
            this->buffer_slot_index = this->buffer_slot_write_counter.get_buffer_index();
        } else {
            this->buffer_slot_index = BufferIndex(0);
        }

        noc_inline_dw_write<InlineWriteDst::L1, posted>(
            edm_connection_handshake_noc_addr,
            tt::tt_fabric::connection_interface::open_connection_value,
            0xf,
            WORKER_HANDSHAKE_NOC);
        *this->worker_teardown_addr = 0;
        if constexpr (!USER_DEFINED_NUM_BUFFER_SLOTS) {
            this->edm_buffer_addr =
                this->edm_buffer_base_addr + (this->get_buffer_slot_index() * this->buffer_size_bytes);
        }
    }

    // SEND_CREDIT_ADDR: True when the EDM sender is IDLE_ETH (mux) as it doesn't have credits on L1 static address
    //                   or some legacy code which skips connection info copy on Tensix L1 static address
    template <
        bool SEND_CREDIT_ADDR = false,
        bool posted = false,
        uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void open() {
        open_start<SEND_CREDIT_ADDR, posted, WORKER_HANDSHAKE_NOC>();
        open_finish<posted, WORKER_HANDSHAKE_NOC>();
    }

    // Advanced usage API:
    // Starts the connection closing process but doesn't wait for the process to complete. This avoids waiting
    // for the ack from the fabric before returning, saving some cycles for advanced users.
    // !!! IMPORTANT !!!
    // Must be called alongside (before) close_finish().
    void close_start() {
        const uint8_t noc = get_fabric_worker_noc();
        const auto dest_noc_addr_coord_only =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0, noc) & ~(uint64_t)NOC_COORDINATE_MASK;

        // buffer index stored at location after handshake addr
        if (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
            noc_inline_dw_write<InlineWriteDst::L1>(
                remote_buffer_index_addr, this->buffer_slot_write_counter.counter, 0xF, noc);
        } else {
            const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
            noc_inline_dw_write<InlineWriteDst::L1>(remote_buffer_index_addr, this->get_buffer_slot_index(), 0xF, noc);
        }
        const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write<InlineWriteDst::L1>(
            dest_edm_connection_state_addr,
            tt::tt_fabric::connection_interface::close_connection_request_value,
            0xF,
            noc);
    }

    // Advanced usage API:
    // Completes the connection closing process. Induces a write barrier
    // !!! IMPORTANT !!!
    // Must be called alongside (after) close_start().
    void close_finish() {
        WAYPOINT("FCFW");
        // Need to wait for the ack to teardown notice, from edm
        while (*this->worker_teardown_addr != 1) {
            invalidate_l1_cache();
        }
        WAYPOINT("FCFD");
        noc_async_write_barrier(get_fabric_worker_noc());
        *(this->worker_teardown_addr) = 0;
    }

    void close() {
        close_start();
        close_finish();
    }

    uint32_t edm_buffer_addr;

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
    size_t edm_buffer_base_addr;

    BufferIndex buffer_slot_index;

    // WORKER ONLY
    ChannelCounter<EDM_NUM_BUFFER_SLOTS> buffer_slot_write_counter;

    uint16_t buffer_size_bytes;
    uint8_t num_buffers_per_channel;

    // noc location of the edm we are connected to (where packets are sent to)
    uint8_t edm_noc_x;
    uint8_t edm_noc_y;

    // the cmd buffer is used for edm-edm path
    uint8_t data_noc_cmd_buf;
    uint8_t sync_noc_cmd_buf;

private:
    template <bool stateful_api = false, bool enable_deadlock_avoidance = false>
    FORCE_INLINE void update_edm_buffer_free_slots(uint8_t noc = get_fabric_worker_noc()) {
        if constexpr (stateful_api) {
            if constexpr (enable_deadlock_avoidance) {
                noc_inline_dw_write_with_state<true, false, true, false, false, InlineWriteDst::REG>(
                    0,  // val unused
                    this->edm_buffer_remote_free_slots_update_addr,
                    this->sync_noc_cmd_buf,
                    noc);
            } else {
                noc_inline_dw_write_with_state<false, false, true, false, false, InlineWriteDst::REG>(
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
        if constexpr (I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            // Write to the atomic increment stream register (write of -1 will subtract 1)
            increment_local_update_ptr_val(worker_credits_stream_id, -1);
        }
    }

    FORCE_INLINE uint8_t get_buffer_slot_index() const { return this->buffer_slot_index.get(); }

    FORCE_INLINE void advance_buffer_slot_write_index() {
        if constexpr (USER_DEFINED_NUM_BUFFER_SLOTS) {
            if (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
                // Mux uses this path
                buffer_slot_write_counter.counter++;
            }
            this->buffer_slot_index = BufferIndex{wrap_increment<EDM_NUM_BUFFER_SLOTS>(this->buffer_slot_index.get())};
        } else {
            if (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
                buffer_slot_write_counter.counter++;
                this->buffer_slot_index =
                    BufferIndex{wrap_increment(this->buffer_slot_index.get(), this->num_buffers_per_channel)};
                this->edm_buffer_addr =
                    this->edm_buffer_base_addr + (this->get_buffer_slot_index() * this->buffer_size_bytes);
            } else {
                this->buffer_slot_index = BufferIndex{wrap_increment(this->buffer_slot_index.get(), this->num_buffers_per_channel)};
                this->edm_buffer_addr =
                    this->edm_buffer_base_addr + (this->get_buffer_slot_index() * this->buffer_size_bytes);
            }
        }
    }

    FORCE_INLINE uint64_t compute_dest_buffer_slot_noc_addr() const {
        // TODO: Worth it to precompute the full noc addr?
        if constexpr (USER_DEFINED_NUM_BUFFER_SLOTS) {
            return get_noc_addr(
                this->edm_noc_x,
                this->edm_noc_y,
                this->edm_buffer_slot_addrs[this->get_buffer_slot_index()],
                get_fabric_worker_noc());
        } else {
            return get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr, get_fabric_worker_noc());
        }
    }

    template <bool stateful_api = false, bool enable_deadlock_avoidance = false>
    FORCE_INLINE void post_send_payload_increment_pointers(uint8_t noc = get_fabric_worker_noc()) {
        this->advance_buffer_slot_write_index();
        this->update_edm_buffer_free_slots<stateful_api, enable_deadlock_avoidance>(noc);
    }

    template <EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_without_header_from_address_impl(uint32_t source_address, size_t size_bytes) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr();

        // skip past the first part of the buffer which will be occupied by the packet header
        send_chunk_from_address<blocking_mode>(
            source_address, 1, size_bytes, buffer_address + sizeof(PACKET_HEADER_TYPE));
    }
    template <EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_from_address_impl(uint32_t source_address, size_t size_bytes) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr();
        ASSERT(size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));
        send_chunk_from_address<blocking_mode>(source_address, 1, size_bytes, buffer_address);
        post_send_payload_increment_pointers();
    }

    template <EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_impl(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr();
        ASSERT(num_pages * page_size <= this->buffer_size_bytes);
        send_chunk<blocking_mode>(cb_id, num_pages, page_size, buffer_address);
        post_send_payload_increment_pointers();
    }
};

using WorkerToFabricEdmSender = WorkerToFabricEdmSenderImpl<false, 0>;


}  // namespace tt::tt_fabric
