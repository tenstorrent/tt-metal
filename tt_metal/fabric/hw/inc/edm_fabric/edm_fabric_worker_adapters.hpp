// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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

template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS = 0, uint8_t VC_ID = 0>
struct WorkerToFabricEdmSenderBase;

// VC0/VC1: connection info read from L1 conn table populated by device-init.
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS = 0>
using WorkerToFabricEdmSenderImpl =
    WorkerToFabricEdmSenderBase<I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, EDM_NUM_BUFFER_SLOTS>;

using WorkerToFabricEdmSender = WorkerToFabricEdmSenderImpl<false, 0>;

// VC2: infrastructure connection — addresses passed as runtime args, stream ID 30.
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS = 0>
using WorkerToFabricEdmSenderVC2Impl =
    WorkerToFabricEdmSenderBase<I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, EDM_NUM_BUFFER_SLOTS, 2>;

using WorkerToFabricEdmSenderVC2 = WorkerToFabricEdmSenderVC2Impl<false, 0>;

namespace fabric_detail{
    template <bool STATEFUL_NOC>
    void update_credits_and_slots(WorkerToFabricEdmSender*);
}

/*
 * Layout of the producer-position word that a worker leaves in the router's L1 at
 * `edm_copy_of_wr_counter_addr` when it closes a connection, and that the next worker to connect
 * picks up. One 32-bit inline write, one 32-bit read -- the router itself never touches it.
 *
 *     [31:24] slot index    - which buffer slot the next producer must write first
 *     [23: 0] write counter - total packets ever injected on this channel, modulo 2^24
 *
 * WHY THE INDEX IS STORED RATHER THAN DERIVED
 * -------------------------------------------
 * This word used to hold the bare write counter, and the reconnecting producer recovered its slot
 * index as `counter % num_buffers`. That is only correct while the counter has never wrapped,
 * because the router's own cursor is a pure mod-num_buffers counter that never passes through a
 * uint32 (fabric_erisc_datamover_channels.hpp: it is zeroed once at bring-up and bumped per packet).
 * After P packets the router sits at `P % num_buffers` but the producer computes
 * `(P % 2^W) % num_buffers`, and those agree for every P only if num_buffers divides 2^W -- i.e.
 * only for power-of-two depths. At depth 18, `2^32 % 18 == 4`, so the first wrap left the producer
 * permanently 4 slots behind the router and the channel silently transmitted stale slots with
 * credits still balanced. Storing the index removes the constraint for every depth.
 *
 * WHY TRUNCATING THE COUNTER TO 24 BITS IS LOSSLESS
 * -------------------------------------------------
 * 24 bits cannot hold the write counter -- it is monotonic for the life of the fabric and is exactly
 * the thing that runs past 2^24 and 2^32. But this word does not have to carry P; it only has to
 * carry enough of P to disambiguate it, because the reader already holds a full-width anchor.
 *
 * At open the producer has R (the router's read counter, exact, 32-bit) and counter_field = P mod
 * 2^24, and it knows 0 <= P - R <= num_buffers. Then
 *     (counter_field - R) mod 2^24 == (P - R) mod 2^24   [counter_field == P mod 2^24, so the
 *                                                         congruence survives subtracting R]
 *                                  == P - R              [0 <= P - R <= 255 < 2^24, so the
 *                                                         reduction is a no-op -- the ONLY step
 *                                                         that needs the occupancy bound]
 * and P = R + that. See open_finish. Truncating to W bits aliases P with P +/- k*2^W; that is
 * resolvable against an exact R iff every possible P - R is below 2^W. The largest is 255, so the
 * minimum workable W is 8 -- 24 is simply what is left after the 8-bit slot index.
 *
 * The 0 <= P - R <= num_buffers bound is the ring occupancy, enforced by the producer's own
 * admission control: wait_for_empty_write_slot() spins until used_slots < num_buffers and exactly one
 * cursor advance follows each admitted packet. Exceeding it would mean overwriting a slot the router
 * has not drained, so this is a pre-existing invariant, not a new requirement.
 *
 * The producer's live counter stays full width while connected and may cross 2^24 freely; only this
 * snapshot is truncated, and it is re-anchored to R on the next open. The per-packet path is
 * unaffected -- get_num_free_write_slots() keeps its plain wrap-safe uint32 subtraction, no mask.
 */
namespace connection_handoff {
constexpr uint32_t COUNTER_BITS = 24;
constexpr uint32_t COUNTER_MASK = (1u << COUNTER_BITS) - 1;

constexpr uint32_t pack(uint8_t slot_index, uint32_t write_counter) {
    return (static_cast<uint32_t>(slot_index) << COUNTER_BITS) | (write_counter & COUNTER_MASK);
}
constexpr uint8_t slot_index_of(uint32_t packed) { return static_cast<uint8_t>(packed >> COUNTER_BITS); }
constexpr uint32_t counter_of(uint32_t packed) { return packed & COUNTER_MASK; }

// A freshly zeroed word must decode to "slot 0, counter 0", which is the correct state for the
// first producer to connect after fabric bring-up.
static_assert(slot_index_of(0) == 0 && counter_of(0) == 0);
}  // namespace connection_handoff
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
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS, uint8_t VC_ID>
struct WorkerToFabricEdmSenderBase {
    static_assert(VC_ID == 0 || VC_ID == 2, "Only VC_ID 0 and 2 are supported");
    // VC0 uses stream 22 (sender_channel_0 free slots); VC2 uses stream 30.
    static constexpr uint32_t STREAM_ID =
        VC_ID == 2 ? tt::tt_fabric::connection_interface::vc2_sender_free_slots_stream_id
                   : tt::tt_fabric::connection_interface::sender_channel_0_free_slots_stream_id;
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

    WorkerToFabricEdmSenderBase() = default;

    template <ProgrammableCoreType my_core_type>
    static WorkerToFabricEdmSenderBase build_from_args(std::size_t& arg_idx) {
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
        if constexpr (my_core_type == ProgrammableCoreType::TENSIX && VC_ID == 0) {
            // VC0: connection info is populated into the L1 conn table by device-init;
            // read it by eth channel index.
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
            // VC2 (TENSIX or ETH): addresses are passed directly as runtime args — no L1 conn table.
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
            worker_free_slots_stream_id = STREAM_ID;
        }

        // DEAD CODE
        // Workers don't have a local stream ID, so we set to a placeholder (unused) value until the worker and EDM
        // codepaths are split
        const StreamId my_fc_stream_channel_id = StreamId{std::numeric_limits<uint32_t>::max()};

        auto worker_teardown_sem_addr =
            reinterpret_cast<volatile uint32_t* const>(get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++)));
        const auto worker_buffer_index_semaphore_addr = get_semaphore<my_core_type>(get_arg_val<uint32_t>(arg_idx++));
        return WorkerToFabricEdmSenderBase(
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
    FORCE_INLINE WorkerToFabricEdmSenderBase(
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

    FORCE_INLINE uint32_t get_num_free_write_slots() const {
        /*
        Without this l1 invalidation `FlowControlAllToAllMeshLowLatency_size_1024_ntype_atomic_inc_ftype_mcast` fabric
        test hangs, while sending packets, waiting for space in the EDM buffer. This is despite disabling the use of the
        l1 data cache. More investigation is needed to discover the underlying issue.
        */
        invalidate_l1_cache();
        if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            auto used_slots = this->buffer_slot_write_counter.counter - *this->edm_buffer_local_free_slots_read_ptr;
            return used_slots >= this->num_buffers_per_channel ? 0 : this->num_buffers_per_channel - used_slots;
        } else {
            return get_ptr_val(worker_credits_stream_id);
        }
    }

    // templatized num_slots to let callers implement bubble flow control without runtime overheads.
    template <size_t num_slots = 1>
    FORCE_INLINE bool edm_has_space_for_packet() const {
        return this->get_num_free_write_slots() >= num_slots;
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        WAYPOINT("FWSW");
        while (!this->edm_has_space_for_packet<1>());
        WAYPOINT("FWSD");
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<EDM_IO_BLOCKING_MODE::BLOCKING>(cb_id, num_pages, page_size);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        send_payload_without_header_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING, posted>(
            source_address, size_bytes, noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_flush_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::FLUSH_BLOCKING, posted>(source_address, size_bytes, noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING, posted>(source_address, size_bytes, noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::BLOCKING, posted>(source_address, size_bytes, noc);
    }

    /*
     * No CB
     */
    // Does not wait for CB. Assumes caller handles CB data availability
    template <bool posted = false>
    FORCE_INLINE void send_payload_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        send_payload_from_address_impl<EDM_IO_BLOCKING_MODE::NON_BLOCKING, posted>(source_address, size_bytes, noc);
    }

    // Non-stateful current-slot helper for payload+header pairs.
    // This avoids recomputing the destination EDM slot address for the header write.
    template <bool posted = false>
    FORCE_INLINE void send_current_slot_non_blocking(
        uint32_t payload_source_l1_addr,
        size_t payload_size_bytes,
        uint32_t header_source_l1_addr,
        uint8_t noc = get_fabric_worker_noc()) {
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_source_l1_addr))));

        const uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr(noc);
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING, posted>(
            payload_source_l1_addr, 1, payload_size_bytes, buffer_address + sizeof(PACKET_HEADER_TYPE), noc);
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING, posted>(
            header_source_l1_addr, 1, sizeof(PACKET_HEADER_TYPE), buffer_address, noc);
        post_send_payload_increment_pointers(noc);
    }

    template <bool posted = false>
    FORCE_INLINE void setup_stateful_send_cmd_bufs(uint8_t noc = get_fabric_worker_noc()) const {
        // In DM_DYNAMIC_NOC, write and write_reg traffic on a worker RISC alias to the same physical cmd buf.
        // Program the state only after generic worker-side NOC setup is complete, and avoid unrelated writes on this
        // RISC while the stateful send loop is active.
        const uint64_t edm_core_noc_addr = get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0, noc);
        ncrisc_noc_write_set_state</*posted=*/posted, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, edm_core_noc_addr, 0, NOC_UNICAST_WRITE_VC);

        const uint64_t credit_noc_addr =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_remote_free_slots_update_addr, noc);
        const uint32_t packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        // Keep the credit inline write non-posted for now; only the payload/header transport path is toggled by
        // `posted`.
        noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
            credit_noc_addr, packed_val, 0xF, this->sync_noc_cmd_buf, noc, NOC_UNICAST_WRITE_VC);
    }

    template <bool posted = false>
    FORCE_INLINE void send_current_slot_stateful_non_blocking(
        uint32_t payload_source_l1_addr,
        uint32_t payload_size_bytes,
        uint32_t header_source_l1_addr,
        uint8_t noc = get_fabric_worker_noc()) {
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_source_l1_addr))));

        const uint32_t slot_l1_addr = this->current_buffer_slot_l1_addr();
        this->issue_payload_to_current_slot_stateful<posted>(
            slot_l1_addr, payload_source_l1_addr, payload_size_bytes, noc);
        this->issue_header_to_current_slot_stateful<posted>(slot_l1_addr, header_source_l1_addr, noc);
        this->post_send_payload_increment_pointers</*stateful_api=*/true>(noc);
    }

    template <bool posted = false>
    FORCE_INLINE void send_current_slot_stateful_non_blocking_from_address(
        uint32_t packet_source_l1_addr, uint32_t packet_size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        ASSERT(packet_size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_source_l1_addr))));

        const uint32_t slot_l1_addr = this->current_buffer_slot_l1_addr();
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, packet_source_l1_addr, slot_l1_addr, packet_size_bytes);
        this->post_send_payload_increment_pointers</*stateful_api=*/true>(noc);
    }

    template <bool posted = false>
    FORCE_INLINE void send_current_slot_stateful_non_blocking_from_address_with_trid(
        uint32_t packet_source_l1_addr,
        uint32_t packet_size_bytes,
        uint32_t trid,
        uint8_t noc = get_fabric_worker_noc()) {
        ASSERT(packet_size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_source_l1_addr))));

        const uint32_t slot_l1_addr = this->current_buffer_slot_l1_addr();
        noc_async_write_one_packet_with_trid_with_state</*update_counter=*/true, posted>(
            packet_source_l1_addr, slot_l1_addr, packet_size_bytes, trid, this->data_noc_cmd_buf, noc);
        this->post_send_payload_increment_pointers</*stateful_api=*/true>(noc);
    }

    FORCE_INLINE uint8_t get_stateful_send_data_noc_cmd_buf() const { return this->data_noc_cmd_buf; }

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
            // some time at least after it copied the wrptr out of the worker_teardown_addr
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
            // Restore the producer position from the packed handoff word the previous producer left
            // behind (see connection_handoff above for the layout and the reasoning).
            //
            // The slot index is taken verbatim -- it is NOT re-derived as `counter % num_buffers`,
            // because the counter only survives modulo 2^COUNTER_BITS and re-deriving would only be
            // correct when num_buffers divides that modulus, i.e. only for power-of-two depths.
            //
            // The full-width counter is then reconstructed from the router's read counter:
            //     outstanding = (P - R) mod 2^COUNTER_BITS, and outstanding <= num_buffers < 2^24,
            //     so R + outstanding == P exactly.
            // That keeps `counter - edm_read_counter` in get_num_free_write_slots() a plain
            // wrap-safe uint32 subtraction with no masking in the hot path.
            invalidate_l1_cache();
            const uint32_t handoff = *this->worker_teardown_addr;
            const uint32_t edm_read_counter = *this->edm_buffer_local_free_slots_read_ptr;
            const uint32_t outstanding =
                (connection_handoff::counter_of(handoff) - edm_read_counter) & connection_handoff::COUNTER_MASK;
            this->buffer_slot_write_counter.reset();
            this->buffer_slot_write_counter.counter = edm_read_counter + outstanding;
            this->buffer_slot_write_counter.index = BufferIndex{connection_handoff::slot_index_of(handoff)};
            this->buffer_slot_index = this->buffer_slot_write_counter.get_buffer_index();
            ASSERT(this->buffer_slot_index.get() < this->num_buffers_per_channel);
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
    template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void close_start() {
        const auto dest_noc_addr_coord_only =
            get_noc_addr(this->edm_noc_x, this->edm_noc_y, 0, WORKER_HANDSHAKE_NOC) & ~(uint64_t)NOC_COORDINATE_MASK;

        // buffer index stored at location after handshake addr
        if (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
            const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
            // Hand off BOTH the slot index and the write counter. The index must be carried
            // explicitly; it cannot be recovered from the counter alone (see connection_handoff).
            noc_inline_dw_write<InlineWriteDst::L1, posted>(
                remote_buffer_index_addr,
                connection_handoff::pack(this->get_buffer_slot_index(), this->buffer_slot_write_counter.counter),
                0xF,
                WORKER_HANDSHAKE_NOC);
        } else {
            const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_copy_of_wr_counter_addr;
            noc_inline_dw_write<InlineWriteDst::L1, posted>(
                remote_buffer_index_addr, this->get_buffer_slot_index(), 0xF, WORKER_HANDSHAKE_NOC);
        }
        const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write<InlineWriteDst::L1, posted>(
            dest_edm_connection_state_addr,
            tt::tt_fabric::connection_interface::close_connection_request_value,
            0xF,
            WORKER_HANDSHAKE_NOC);
    }

    // Advanced usage API:
    // Completes the connection closing process. Induces a write barrier
    // !!! IMPORTANT !!!
    // Must be called alongside (after) close_start().
    template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void close_finish() {
        WAYPOINT("FCFW");
        if constexpr (posted) {
            noc_async_posted_writes_flushed(WORKER_HANDSHAKE_NOC);
        }
        noc_async_write_barrier(WORKER_HANDSHAKE_NOC);

        // Need to wait for the ack to teardown notice, from edm
        while (*this->worker_teardown_addr != 1) {
            invalidate_l1_cache();
        }
        WAYPOINT("FCFD");
        *(this->worker_teardown_addr) = 0;
    }

    template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = get_fabric_worker_noc()>
    void close() {
        close_start<posted, WORKER_HANDSHAKE_NOC>();
        close_finish<posted, WORKER_HANDSHAKE_NOC>();
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
    template <bool STATEFUL_NOC>
    friend void fabric_detail::update_credits_and_slots(WorkerToFabricEdmSender*);

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
                noc_inline_dw_write_with_state<false, true, false, false, false, InlineWriteDst::REG>(
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

    FORCE_INLINE uint64_t compute_dest_buffer_slot_noc_addr(uint8_t noc = get_fabric_worker_noc()) const {
        // TODO: Worth it to precompute the full noc addr?
        if constexpr (USER_DEFINED_NUM_BUFFER_SLOTS) {
            return get_noc_addr(
                this->edm_noc_x, this->edm_noc_y, this->edm_buffer_slot_addrs[this->get_buffer_slot_index()], noc);
        } else {
            return get_noc_addr(this->edm_noc_x, this->edm_noc_y, this->edm_buffer_addr, noc);
        }
    }

    FORCE_INLINE uint32_t current_buffer_slot_l1_addr() const {
        if constexpr (USER_DEFINED_NUM_BUFFER_SLOTS) {
            return this->edm_buffer_slot_addrs[this->get_buffer_slot_index()];
        } else {
            return this->edm_buffer_addr;
        }
    }

    template <bool posted = false>
    FORCE_INLINE void issue_payload_to_current_slot_stateful(
        uint32_t slot_l1_addr,
        uint32_t payload_source_l1_addr,
        uint32_t payload_size_bytes,
        uint8_t noc = get_fabric_worker_noc()) const {
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc,
            this->data_noc_cmd_buf,
            payload_source_l1_addr,
            slot_l1_addr + sizeof(PACKET_HEADER_TYPE),
            payload_size_bytes);
    }

    template <bool posted = false>
    FORCE_INLINE void issue_header_to_current_slot_stateful(
        uint32_t slot_l1_addr, uint32_t header_source_l1_addr, uint8_t noc = get_fabric_worker_noc()) {
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, header_source_l1_addr, slot_l1_addr, sizeof(PACKET_HEADER_TYPE));
    }

    template <bool stateful_api = false, bool enable_deadlock_avoidance = false>
    FORCE_INLINE void post_send_payload_increment_pointers(uint8_t noc = get_fabric_worker_noc()) {
        this->update_edm_buffer_free_slots<stateful_api, enable_deadlock_avoidance>(noc);
        this->advance_buffer_slot_write_index();
    }

    template <EDM_IO_BLOCKING_MODE blocking_mode, bool posted = false>
    FORCE_INLINE void send_payload_without_header_from_address_impl(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr(noc);

        // skip past the first part of the buffer which will be occupied by the packet header
        send_chunk_from_address<blocking_mode, posted>(
            source_address, 1, size_bytes, buffer_address + sizeof(PACKET_HEADER_TYPE), noc);
    }
    template <EDM_IO_BLOCKING_MODE blocking_mode, bool posted = false>
    FORCE_INLINE void send_payload_from_address_impl(
        uint32_t source_address, size_t size_bytes, uint8_t noc = get_fabric_worker_noc()) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr(noc);
        ASSERT(size_bytes <= this->buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));
        send_chunk_from_address<blocking_mode, posted>(source_address, 1, size_bytes, buffer_address, noc);
        post_send_payload_increment_pointers(noc);
    }

    template <EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_impl(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        uint64_t buffer_address = this->compute_dest_buffer_slot_noc_addr();
        ASSERT(num_pages * page_size <= this->buffer_size_bytes);
        send_chunk<blocking_mode>(cb_id, num_pages, page_size, buffer_address);
        post_send_payload_increment_pointers();
    }
};

namespace fabric_detail{
    template <bool STATEFUL_NOC>
    void update_credits_and_slots(WorkerToFabricEdmSender* conn){
        conn->advance_buffer_slot_write_index();
        conn->update_edm_buffer_free_slots<STATEFUL_NOC>();
    }
} // namespace fabric_detail

}  // namespace tt::tt_fabric
