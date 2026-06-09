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

/*
 * FabricMuxV2Sender: worker-facing client adapter for the transient self-poll Mux V2.
 *
 * This is a fresh, V2-owned adapter (it does not embed/inherit the V1
 * WorkerToFabricEdmSenderBase). It reuses only the low-level NOC/stream-register
 * primitives; the connection state machine is V2-specific and tuned for the
 * single-use transient lifecycle (cheap open/close, no buffer-index
 * read-back/write-back).
 *
 * Roles it talks to on the mux core:
 *  - Forwarder (RISCV_0): owns the downstream send path. The worker signals a
 *    pending packet by decrementing the forwarder's per-channel credit stream
 *    register (stream id == logical_channel_id). The forwarder increments it
 *    back after forwarding.
 *  - Manager  (RISCV_1): owns the upstream control path. It reads the worker
 *    location info on `open`, publishes a monotonic read counter to the worker's
 *    flow-control word as packets retire downstream, and acks teardown on
 *    `close` by incrementing the worker's teardown word.
 *
 * Flow control is counter-based (mirrors the V1 worker path): free slots =
 * num_buffers - (local_write_counter - published_read_counter).
 *
 * Runtime-arg layout (must match FabricMuxV2Config::append_client_connection_rt_args):
 *   0  mux_x
 *   1  mux_y
 *   2  logical_channel_id
 *   3  num_buffers_per_channel
 *   4  channel_buffer_size_bytes
 *   5  channel_base_address           (mux L1 base of this channel's slot ring)
 *   6  connection_info_address        (mux EDMChannelWorkerLocationInfo)
 *   7  connection_handshake_address   (mux per-channel handshake scalar)
 *   8  flow_control_sem_id            (local: manager publishes read counter here)
 *   9  teardown_sem_id                (local: teardown ack; also status scratch)
 *   10 mux_status_address             (mux-global status word)
 */
template <uint8_t NUM_BUFFERS = 0>
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

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------

    // Gate on the mux being ready, then open the transient connection. Cheap by
    // design: no buffer-index read-back and no blocking on the manager's
    // establish (free slots start full, and the manager only reads location info
    // after observing the open value, which is written last in program order).
    // No write barrier here: the inline handshake writes are non-posted and the
    // manager polls for them; first-send correctness does not depend on them
    // having landed (mirrors WorkerToFabricEdmSenderBase::open).
    void open() {
        wait_until_ready();

        write_counter = 0;
        current_slot = 0;
        *flow_control_ptr = 0;  // read counter the manager will publish into
        *teardown_ptr = 0;      // teardown ack word

        publish_worker_location_info();
        request_connection_open();
    }

    // Request teardown and wait for the manager's ack. Valid even with zero
    // packets sent; the mux drains any staged packets before retiring credit.
    void close() {
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

    // ---------------------------------------------------------------------
    // Data plane: generic fabric-helper contract (implemented natively)
    // ---------------------------------------------------------------------

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

    // Stage the payload into the current slot, after the packet header region.
    // Does not commit (no credit/pointer advance); the matching flush commits.
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes) {
        const uint64_t slot_noc_addr = this->current_slot_noc_addr();
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(
            source_address, 1, size_bytes, slot_noc_addr + sizeof(PACKET_HEADER_TYPE));
    }

    // Write `size_bytes` (the packet header, possibly with a co-located payload)
    // to the start of the current slot, then commit the slot.
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        ASSERT(size_bytes <= channel_buffer_size_bytes);
        ASSERT(tt::tt_fabric::is_valid(
            *const_cast<PACKET_HEADER_TYPE*>(reinterpret_cast<volatile PACKET_HEADER_TYPE*>(source_address))));
        const uint64_t slot_noc_addr = this->current_slot_noc_addr();
        send_chunk_from_address<EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, 1, size_bytes, slot_noc_addr);
        this->commit_current_slot();
    }

    // ---------------------------------------------------------------------
    // Data plane: stateful perf lane
    // ---------------------------------------------------------------------

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
        this->commit_current_slot_stateful(noc);
    }

    FORCE_INLINE uint8_t get_stateful_send_data_noc_cmd_buf() const { return this->data_noc_cmd_buf; }

private:
    void wait_until_ready() {
        const uint64_t status_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, mux_status_address, noc_index);
        // Reuse the teardown word as the readback scratch; it is cleared in open().
        volatile tt_l1_ptr uint32_t* scratch = teardown_ptr;
        WAYPOINT("MV2W");
        do {
            noc_async_read(status_noc_addr, reinterpret_cast<size_t>(scratch), sizeof(uint32_t), noc_index);
            noc_async_read_barrier(noc_index);
            invalidate_l1_cache();
        } while (*scratch != static_cast<uint32_t>(FabricMuxStatus::READY_FOR_TRAFFIC));
        WAYPOINT("MV2R");
    }

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

    FORCE_INLINE void advance_slot() {
        const uint8_t slot_count = USER_DEFINED_NUM_BUFFERS ? NUM_BUFFERS : num_buffers;
        current_slot = (current_slot + 1 == slot_count) ? 0 : static_cast<uint8_t>(current_slot + 1);
    }

    // Publish half of the stage/commit seam: signal a pending packet to the
    // forwarder (decrement its per-channel credit stream reg), then advance our
    // local write counter and slot cursor.
    FORCE_INLINE void commit_current_slot() {
        const uint64_t credit_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, credit_stream_reg_write_addr, noc_index);
        const int32_t packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        noc_inline_dw_write<InlineWriteDst::REG>(credit_noc_addr, packed_val, 0xf, noc_index);
        write_counter++;
        this->advance_slot();
    }

    FORCE_INLINE void commit_current_slot_stateful(uint8_t noc) {
        noc_inline_dw_write_with_state</*posted=*/false, /*update=*/true, false, false, false, InlineWriteDst::REG>(
            0, 0, this->sync_noc_cmd_buf, noc);
        write_counter++;
        this->advance_slot();
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

    // Local L1 words (2 semaphores). flow_control receives the manager-published
    // read counter; teardown receives the teardown ack and doubles as the
    // ready-status readback scratch during connect.
    volatile tt_l1_ptr uint32_t* flow_control_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* teardown_ptr = nullptr;

    uint32_t write_counter = 0;
    uint8_t current_slot = 0;

    uint8_t data_noc_cmd_buf = write_reg_cmd_buf;
    uint8_t sync_noc_cmd_buf = write_at_cmd_buf;
};

}  // namespace tt::tt_fabric
