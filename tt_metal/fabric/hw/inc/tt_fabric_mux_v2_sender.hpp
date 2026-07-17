// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

namespace tt::tt_fabric {

enum class FabricMuxV2SenderState : uint8_t {
    Disconnected = 0,
    Staging = 1,
    Connected = 2,
};

static constexpr uint8_t kInvalidStatusReadTrid = 0xFF;

/*
 * Worker client for transient Mux V2.
 *
 * Mux-core roles:
 *  - Forwarder: data path. Worker signals a pending packet by decrementing the
 *    per-channel stream reg (id == logical_channel_id); forwarder increments after send.
 *  - Manager: open/close handshake, publishes monotonic read counter for flow control,
 *    and acks teardown on close.
 *
 * Flow control: free_slots = num_buffers - (write_counter - published_read_counter).
 *
 * States: Disconnected -> (optional Staging if EAGER_STAGING) -> Connected -> close.
 *
 * Usage:
 *  - Default (EAGER_STAGING=false): open() -> send* -> close(). flush() is unused.
 *  - Eager staging: open() enters Staging and may fill slots without signaling the
 *    forwarder. Transition to Connected with flush() (blocking or poll), or let
 *    ring-full / close() do it. After Connected, send* commits normally.
 *  - flush<false>() overlaps READY polls with local work; flush<true>() (or close())
 *    when you must be Connected before continuing.
 *  - status_trid on open() is only for non-blocking READY polls during Staging;
 *    omit / kInvalidStatusReadTrid if you only use blocking flush / close.
 *
 * Runtime args (FabricMuxV2Config::append_client_connection_rt_args):
 *   mux_x, mux_y, logical_channel_id, num_buffers, channel_buffer_size_bytes,
 *   channel_base, connection_info, handshake, flow_control_sem, teardown_sem, mux_status
 */
template <bool EAGER_STAGING = false, uint8_t NUM_BUFFERS = 0>
class FabricMuxV2Sender {
public:
    static constexpr bool USER_DEFINED_NUM_BUFFERS = NUM_BUFFERS != 0;

    FabricMuxV2Sender() = default;

    static FabricMuxV2Sender build_from_args(std::size_t& arg_idx) {
        FabricMuxV2Sender sender;
        sender.mux_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        sender.mux_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        sender.logical_channel_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
        const uint32_t num_buffers_rt = get_arg_val<uint32_t>(arg_idx++);
        sender.channel_buffer_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        sender.channel_base_address = get_arg_val<uint32_t>(arg_idx++);
        sender.connection_info_address = get_arg_val<uint32_t>(arg_idx++);
        sender.connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t flow_control_sem_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t teardown_sem_id = get_arg_val<uint32_t>(arg_idx++);
        sender.mux_status_address = get_arg_val<uint32_t>(arg_idx++);

        sender.num_buffers = USER_DEFINED_NUM_BUFFERS ? NUM_BUFFERS : static_cast<uint8_t>(num_buffers_rt);
        // Clients are always Tensix workers.
        sender.flow_control_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore<ProgrammableCoreType::TENSIX>(flow_control_sem_id));
        sender.teardown_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore<ProgrammableCoreType::TENSIX>(teardown_sem_id));
        sender.credit_stream_reg_write_addr = get_stream_reg_write_addr(sender.logical_channel_id);
        return sender;
    }

    // Wait for READY (unless EAGER_STAGING), publish location info, request open.
    // With EAGER_STAGING: enter Staging immediately; optional status_trid enables
    // non-blocking READY polls during staging.
    void open(uint8_t status_trid = kInvalidStatusReadTrid) {
        if constexpr (EAGER_STAGING) {
            open_staging(status_trid);
        } else {
            open_blocking();
        }
    }

    // Staging -> Connected when READY. No-op if already Connected.
    // Blocking: wait for READY, then transition (always returns true).
    // Non-blocking: return true if Connected / READY observed, else false.
    template <bool Blocking = true>
    bool flush() {
        if (state != FabricMuxV2SenderState::Staging) {
            return true;
        }
        if constexpr (Blocking) {
            if (status_read_in_flight) {
                drain_status_read_trid();
                if (check_scratch_ready()) {
                    open_finish();
                    return true;
                }
                status_read_in_flight = false;
            }
            wait_until_ready_blocking();
            open_finish();
            return true;
        } else {
            if (status_read_in_flight) {
                if (!poll_status_read_trid()) {
                    return false;
                }
                if (check_scratch_ready()) {
                    open_finish();
                    return true;
                }
            }
            issue_status_read();
            return false;
        }
    }

    // From Staging, flush first. Request teardown and block on manager ack.
    void close() {
        if constexpr (EAGER_STAGING) {
            flush<true>();
            noc_async_writes_flushed();
        }

        const uint64_t handshake_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, connection_handshake_address, noc_index);
        noc_inline_dw_write<InlineWriteDst::L1>(
            handshake_noc_addr, connection_interface::close_connection_request_value, 0xf, noc_index);

        WAYPOINT("MV2C");
        while (*teardown_ptr != 1) {
            invalidate_l1_cache();
        }
        WAYPOINT("MV2D");
        noc_async_write_barrier(noc_index);
        *teardown_ptr = 0;
    }

    // --- Non-stateful send lane (generic fabric helpers) ---

    FORCE_INLINE uint32_t get_num_free_write_slots() const {
        invalidate_l1_cache();
        const uint32_t used_slots = write_counter - *flow_control_ptr;
        return used_slots >= num_buffers ? 0 : (num_buffers - used_slots);
    }

    template <size_t num_slots = 1>
    FORCE_INLINE bool has_space_for_packet() const {
        return this->get_num_free_write_slots() >= num_slots;
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        WAYPOINT("MWSW");
        while (!this->has_space_for_packet<1>());
        WAYPOINT("MWSD");
    }

    // Write payload after the header region; does not commit (pair with flush below).
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes) {
        const uint64_t slot_noc_addr = this->current_slot_noc_addr();
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(
            source_address, 1, size_bytes, slot_noc_addr + sizeof(PACKET_HEADER_TYPE));
    }

    // Write header (+ optional co-located payload) at slot start, then commit.
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        ASSERT(size_bytes <= channel_buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));
        const uint64_t slot_noc_addr = this->current_slot_noc_addr();
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, 1, size_bytes, slot_noc_addr);
        if constexpr (EAGER_STAGING) {
            commit_or_stage_non_stateful();
        } else {
            this->commit_current_slot();
        }
    }

    // --- Stateful send lane ---
    // DATA cmd buf: mux slot destination. SYNC cmd buf: credit stream reg (-1).
    // SYNC state is clobbered by noc_inline_dw_write / noc_semaphore_inc on write_at_cmd_buf.
    // open_finish() reprograms SYNC after batched credit notify if setup was done; do not
    // use write_at_cmd_buf ops between stateful sends without re-setup.
    template <bool posted = false>
    FORCE_INLINE void setup_stateful_send_cmd_bufs(uint8_t noc = noc_index) {
        this->data_noc_cmd_buf = write_reg_cmd_buf;
        this->sync_noc_cmd_buf = write_at_cmd_buf;

        const uint64_t mux_core_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, 0, noc);
        ncrisc_noc_write_set_state</*posted=*/posted, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, mux_core_noc_addr, 0, NOC_UNICAST_WRITE_VC);

        const uint64_t credit_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, credit_stream_reg_write_addr, noc);
        const uint32_t packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
            credit_noc_addr, packed_val, 0xF, this->sync_noc_cmd_buf, noc, NOC_UNICAST_WRITE_VC);
        stateful_setup_done = true;
    }

    template <bool posted = false>
    FORCE_INLINE void send_current_slot_stateful_non_blocking_from_address(
        uint32_t packet_source_l1_addr, uint32_t packet_size_bytes, uint8_t noc = noc_index) {
        ASSERT(packet_size_bytes <= channel_buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_source_l1_addr))));

        const uint32_t slot_l1_addr = this->current_slot_l1_addr();
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, packet_source_l1_addr, slot_l1_addr, packet_size_bytes);
        if constexpr (EAGER_STAGING) {
            commit_or_stage_stateful(noc);
        } else {
            this->commit_current_slot_stateful(noc);
        }
    }

    template <bool posted = false>
    FORCE_INLINE void send_current_slot_stateful_non_blocking(
        uint32_t payload_source_l1_addr,
        uint32_t payload_size_bytes,
        uint32_t header_source_l1_addr,
        uint8_t noc = noc_index) {
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_source_l1_addr))));

        const uint32_t slot_l1_addr = this->current_slot_l1_addr();
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc,
            this->data_noc_cmd_buf,
            payload_source_l1_addr,
            slot_l1_addr + sizeof(PACKET_HEADER_TYPE),
            payload_size_bytes);
        ncrisc_noc_write_with_state<noc_mode, /*posted=*/posted, /*update_counter=*/true, /*one_packet=*/false>(
            noc, this->data_noc_cmd_buf, header_source_l1_addr, slot_l1_addr, sizeof(PACKET_HEADER_TYPE));
        if constexpr (EAGER_STAGING) {
            commit_or_stage_stateful(noc);
        } else {
            this->commit_current_slot_stateful(noc);
        }
    }

    FORCE_INLINE uint8_t get_stateful_send_data_noc_cmd_buf() const { return this->data_noc_cmd_buf; }

    FORCE_INLINE bool is_staging_ring_full() const {
        return state == FabricMuxV2SenderState::Staging && deferred_count == num_buffers;
    }

private:
    void publish_worker_location_info() {
        const uint64_t worker_semaphore_field = get_noc_addr(
            mux_noc_x,
            mux_noc_y,
            connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_semaphore_address),
            noc_index);
        noc_inline_dw_write<InlineWriteDst::L1>(
            worker_semaphore_field, reinterpret_cast<size_t>(flow_control_ptr), 0xf, noc_index);

        const uint64_t worker_teardown_field = get_noc_addr(
            mux_noc_x,
            mux_noc_y,
            connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_teardown_semaphore_address),
            noc_index);
        noc_inline_dw_write<InlineWriteDst::L1>(
            worker_teardown_field, reinterpret_cast<size_t>(teardown_ptr), 0xf, noc_index);

        const uint64_t worker_xy_field = get_noc_addr(
            mux_noc_x,
            mux_noc_y,
            connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_xy),
            noc_index);
        noc_inline_dw_write<InlineWriteDst::L1>(
            worker_xy_field, WorkerXY(my_x[0], my_y[0]).to_uint32(), 0xf, noc_index);
    }

    void request_connection_open() {
        const uint64_t handshake_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, connection_handshake_address, noc_index);
        noc_inline_dw_write<InlineWriteDst::L1>(
            handshake_noc_addr, connection_interface::open_connection_value, 0xf, noc_index);
    }

    FORCE_INLINE uint32_t current_slot_l1_addr() const {
        return channel_base_address + (current_slot * channel_buffer_size_bytes);
    }

    FORCE_INLINE uint64_t current_slot_noc_addr() const {
        return get_noc_addr(mux_noc_x, mux_noc_y, this->current_slot_l1_addr(), noc_index);
    }

    FORCE_INLINE void advance_local_cursor() {
        write_counter++;
        advance_slot();
    }

    FORCE_INLINE void signal_pending_non_stateful(uint32_t count) {
        const uint64_t credit_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, credit_stream_reg_write_addr, noc_index);
        const int32_t packed_val = pack_value_for_inc_on_write_stream_reg_write(-static_cast<int32_t>(count));
        noc_inline_dw_write<InlineWriteDst::REG>(credit_noc_addr, packed_val, 0xf, noc_index);
    }

    FORCE_INLINE void signal_pending_stateful(uint32_t count, uint8_t noc) {
        if (count == 1) {
            noc_inline_dw_write_with_state</*posted=*/false, /*update=*/true, false, false, false, InlineWriteDst::REG>(
                0, 0, this->sync_noc_cmd_buf, noc);
        } else {
            // Batched credit notify (open_finish); SYNC is reprogrammed afterward if needed.
            signal_pending_non_stateful(count);
        }
    }

    FORCE_INLINE void advance_slot() {
        const uint8_t slot_count = USER_DEFINED_NUM_BUFFERS ? NUM_BUFFERS : num_buffers;
        current_slot = (current_slot + 1 == slot_count) ? 0 : static_cast<uint8_t>(current_slot + 1);
    }

    FORCE_INLINE void commit_current_slot() {
        signal_pending_non_stateful(1);
        advance_local_cursor();
    }

    FORCE_INLINE void commit_current_slot_stateful(uint8_t noc) {
        signal_pending_stateful(1, noc);
        advance_local_cursor();
    }

    void open_blocking() {
        wait_until_ready_blocking();
        init_local_state();
        publish_worker_location_info();
        request_connection_open();
        state = FabricMuxV2SenderState::Connected;
    }

    void open_staging(uint8_t status_trid) {
        init_local_state();
        publish_worker_location_info();
        status_read_trid = status_trid;
        if (status_read_trid != kInvalidStatusReadTrid) {
            issue_status_read();
        }
        state = FabricMuxV2SenderState::Staging;
    }

    void init_local_state() {
        write_counter = 0;
        current_slot = 0;
        deferred_count = 0;
        *flow_control_ptr = 0;
        *teardown_ptr = 0;
    }

    void open_finish() {
        if (status_read_in_flight) {
            drain_status_read_trid();
            status_read_in_flight = false;
        }
        *teardown_ptr = 0;
        request_connection_open();
        if (deferred_count > 0) {
            signal_pending_non_stateful(deferred_count);
            deferred_count = 0;
        }
        if constexpr (EAGER_STAGING) {
            if (stateful_setup_done) {
                reprogram_stateful_sync_cmd_buf();
            }
        }
        state = FabricMuxV2SenderState::Connected;
    }

    FORCE_INLINE void commit_or_stage_non_stateful() {
        if (state == FabricMuxV2SenderState::Staging) {
            if (deferred_count == num_buffers) {
                flush<true>();
            }
            if (state == FabricMuxV2SenderState::Staging) {
                advance_local_cursor();
                deferred_count++;
                ready_check_opportunistic();
                return;
            }
        }
        commit_current_slot();
    }

    FORCE_INLINE void commit_or_stage_stateful(uint8_t noc) {
        if (state == FabricMuxV2SenderState::Staging) {
            if (deferred_count == num_buffers) {
                flush<true>();
            }
            if (state == FabricMuxV2SenderState::Staging) {
                advance_local_cursor();
                deferred_count++;
                ready_check_opportunistic();
                return;
            }
        }
        commit_current_slot_stateful(noc);
    }

    FORCE_INLINE void ready_check_opportunistic() {
        if (status_read_trid == kInvalidStatusReadTrid) {
            return;
        }
        if (!status_read_in_flight) {
            issue_status_read();
            return;
        }
        if (poll_status_read_trid()) {
            if (check_scratch_ready()) {
                open_finish();
            } else {
                issue_status_read();
            }
        }
    }

    void wait_until_ready_blocking() {
        const uint64_t status_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, mux_status_address, noc_index);
        volatile tt_l1_ptr uint32_t* scratch = teardown_ptr;
        WAYPOINT("MV2W");
        do {
            noc_async_read(status_noc_addr, reinterpret_cast<size_t>(scratch), sizeof(uint32_t), noc_index);
            noc_async_read_barrier(noc_index);
            invalidate_l1_cache();
        } while (*scratch != static_cast<uint32_t>(FabricMuxStatus::READY_FOR_TRAFFIC));
        WAYPOINT("MV2R");
    }

    void issue_status_read() {
        if (status_read_trid == kInvalidStatusReadTrid) {
            return;
        }
        const uint64_t status_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, mux_status_address, noc_index);
        noc_async_read_set_trid(status_read_trid, noc_index);
        noc_async_read(status_noc_addr, reinterpret_cast<size_t>(teardown_ptr), sizeof(uint32_t), noc_index);
        status_read_in_flight = true;
    }

    bool poll_status_read_trid() { return ncrisc_noc_read_with_transaction_id_flushed(noc_index, status_read_trid); }

    void drain_status_read_trid() {
        while (!poll_status_read_trid()) {
        }
        invalidate_l1_cache();
    }

    bool check_scratch_ready() {
        invalidate_l1_cache();
        return *teardown_ptr == static_cast<uint32_t>(FabricMuxStatus::READY_FOR_TRAFFIC);
    }

    void reprogram_stateful_sync_cmd_buf() {
        const uint64_t credit_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, credit_stream_reg_write_addr, noc_index);
        const uint32_t packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
            credit_noc_addr, packed_val, 0xF, this->sync_noc_cmd_buf, noc_index, NOC_UNICAST_WRITE_VC);
    }

    uint8_t mux_noc_x = 0;
    uint8_t mux_noc_y = 0;
    uint8_t logical_channel_id = 0;
    uint8_t num_buffers = 0;
    uint32_t channel_buffer_size_bytes = 0;
    uint32_t channel_base_address = 0;
    uint32_t connection_info_address = 0;
    uint32_t connection_handshake_address = 0;
    uint32_t mux_status_address = 0;
    uint32_t credit_stream_reg_write_addr = 0;

    // flow_control: manager-published read counter. teardown: close ack; also READY scratch.
    volatile tt_l1_ptr uint32_t* flow_control_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* teardown_ptr = nullptr;

    uint32_t write_counter = 0;
    uint8_t current_slot = 0;

    uint8_t data_noc_cmd_buf = write_reg_cmd_buf;
    uint8_t sync_noc_cmd_buf = write_at_cmd_buf;

    FabricMuxV2SenderState state = FabricMuxV2SenderState::Disconnected;
    uint8_t deferred_count = 0;
    uint8_t status_read_trid = kInvalidStatusReadTrid;
    bool status_read_in_flight = false;
    bool stateful_setup_done = false;
};

}  // namespace tt::tt_fabric
