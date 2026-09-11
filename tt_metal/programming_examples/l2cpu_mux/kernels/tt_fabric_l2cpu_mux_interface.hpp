// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Worker-side client for the L2CPU-resident fabric mux (x280/fw_mux.c).
//
// Same call surface as the V1 mux client in tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp:
//   build_connection_to_l2cpu_mux(...)        (mirrors build_connection_to_fabric_endpoint)
//   wait_for_fabric_endpoint_ready(...)       (reused as is: status word is plain memory)
//   fabric_client_connect / _disconnect       (overloads for WorkerToL2cpuMuxSender)
//   fabric_async_write / fabric_atomic_inc    (overloads)
//   fabric_endpoint_terminate(...)            (reused as is: termination word is plain memory)
//
// ALIGNMENT CONTRACT (measured on Blackhole, see l2cpu_noc_transfer/l2cpu_align_probe):
// NOC WRITES from Tensix L1 into L2CPU memory land correctly at any offsets, so payload
// and header sources (CB pages, packet header pool) need no special alignment. NOC READS
// from L2CPU memory fetch from (src & ~63) + (landing & 63), i.e. the L1 landing address
// must share the source's offset within 64 B. The client makes exactly three reads
// (status word, cursor block, read-counter seed) and derives their landing zones from
// ONE local scratch region of at least 192 B: the `local_buffer_index_address` argument
// is rounded up to 64 B and used as   +0x00 cursor landing (LM_CURSOR is +0 mod 64),
// +0x40 status landing (LM_STATUS is +0 mod 64), +0x70 flow-control word (matches
// conn_info+0x30). The `local_flow_control_address` argument is used only if it already
// sits at +0x30 mod 64. Semaphores (16 B strided) are fine for the teardown word.
//
// Protocol: identical to WorkerToFabricEdmSender against a mux channel — cursor adopt,
// location info, handshake 1/2, read-counter credits pushed into local L1 — with ONE
// substitution. A Tensix/eth mux is committed to by decrementing a stream register on
// the mux core; the L2CPU tile has no stream registers, so this client commits by
// writing its (incremented) write counter into the channel's write-counter word with a
// plain inline write. Single producer per channel makes that lossless and ordered
// behind the slot writes on the same NOC.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "fabric/fabric_edm_packet_header.hpp"

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric {

template <uint8_t NUM_BUFFERS>
struct WorkerToL2cpuMuxSender {
    // Mux (L2CPU) tile and channel geometry
    uint8_t edm_noc_x = 0;
    uint8_t edm_noc_y = 0;
    uint32_t channel_base_address = 0;
    uint32_t buffer_size_bytes = 0;
    uint32_t connection_info_address = 0;
    uint32_t connection_handshake_address = 0;
    uint32_t producer_write_counter_address = 0;  // "flow control" address: where we commit
    uint32_t buffer_index_address = 0;            // SenderChannelProducerCursor on the mux
    // Local words the mux writes into
    volatile tt_l1_ptr uint32_t* local_flow_control_ptr = nullptr;  // mux read counter lands here
    volatile tt_l1_ptr uint32_t* local_teardown_ptr = nullptr;      // mux writes 1 on close
    uint32_t local_cursor_address = 0;                              // 64 B-aligned landing zone for the cursor
    uint32_t local_status_scratch = 0;                              // 64 B-aligned landing zone for the status word
    // Producer state
    uint32_t write_counter = 0;
    uint8_t write_index = 0;

    FORCE_INLINE uint64_t mux_addr(uint32_t a) const { return get_noc_addr(edm_noc_x, edm_noc_y, a); }
    FORCE_INLINE uint32_t current_slot_address() const {
        return channel_base_address + write_index * buffer_size_bytes;
    }

    // ---- open: same steps as WorkerToFabricEdmSender::open_start/open_finish ----
    void open_start() {
        noc_async_read(mux_addr(buffer_index_address), local_cursor_address, sizeof(SenderChannelProducerCursor));
        // 16 B read (the counter is the first word of a 16 B-strided field, and the local
        // semaphore slot is 16 B): sub-16 B NOC reads from the L2CPU tile are not relied on.
        noc_async_read(
            mux_addr(connection_info_address + offsetof(EDMChannelWorkerLocationInfo, edm_read_counter)),
            reinterpret_cast<uint32_t>(local_flow_control_ptr),
            16);
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_semaphore_address)),
            reinterpret_cast<uint32_t>(local_flow_control_ptr));
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(
                connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_teardown_semaphore_address)),
            reinterpret_cast<uint32_t>(local_teardown_ptr));
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(connection_info_address + offsetof(EDMChannelWorkerLocationInfo, worker_xy)),
            WorkerXY(my_x[0], my_y[0]).to_uint32());
    }
    void open_finish() {
        noc_async_read_barrier();
        invalidate_l1_cache();
        auto* cursor = reinterpret_cast<volatile tt_l1_ptr SenderChannelProducerCursor*>(local_cursor_address);
        write_counter = cursor->write_counter;
        write_index = cursor->write_index < NUM_BUFFERS ? static_cast<uint8_t>(cursor->write_index) : 0;
        *local_teardown_ptr = 0;
        noc_async_write_barrier();  // location info must land before the handshake
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(connection_handshake_address), connection_interface::open_connection_value);
    }
    template <bool SEND_CREDIT_ADDR = false>
    void open() {
        open_start();
        open_finish();
    }

    // Wait for the mux to be READY_FOR_TRAFFIC. If `config_gen` is nonzero, also wait until
    // the mux reports it built its router connection from that FF_CONN_VALID nonce
    // (LM_CONFIG_GEN) — lets a program deliver the connection block itself (conn_setup
    // kernel) and have its clients wait for the reconfiguration to land.
    void wait_for_ready(uint32_t mux_status_address, uint32_t config_gen_address = 0, uint32_t config_gen = 0) {
        auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_status_scratch);
        for (;;) {
            noc_async_read(mux_addr(mux_status_address), local_status_scratch, 16);
            if (config_gen != 0) {
                noc_async_read(mux_addr(config_gen_address), local_status_scratch + 16, 16);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            const bool ready = scratch[0] == static_cast<uint32_t>(EDMStatus::READY_FOR_TRAFFIC);
            const bool gen_ok = config_gen == 0 || scratch[4] == config_gen;
            if (ready && gen_ok) {
                return;
            }
        }
    }

    // ---- flow control ----
    FORCE_INLINE uint32_t get_num_free_write_slots() const {
        invalidate_l1_cache();
        const uint32_t used = write_counter - *local_flow_control_ptr;
        return used >= NUM_BUFFERS ? 0 : NUM_BUFFERS - used;
    }
    template <size_t num_slots = 1>
    FORCE_INLINE bool edm_has_space_for_packet() const {
        return get_num_free_write_slots() >= num_slots;
    }
    FORCE_INLINE void wait_for_empty_write_slot() const {
        WAYPOINT("LMSW");
        while (!edm_has_space_for_packet<1>());
        WAYPOINT("LMSD");
    }

    // ---- sends (same names/semantics as the EDM adapter) ----
    template <bool posted = false>
    FORCE_INLINE void send_payload_without_header_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = noc_index) {
        noc_async_write(source_address, mux_addr(current_slot_address() + sizeof(PACKET_HEADER_TYPE)), size_bytes, noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_flush_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = noc_index) {
        noc_async_write(source_address, mux_addr(current_slot_address()), size_bytes, noc);
        noc_async_writes_flushed(noc);
        commit(noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_flush_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = noc_index) {
        noc_async_write(source_address, mux_addr(current_slot_address()), size_bytes, noc);
        commit(noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_payload_non_blocking_from_address(
        uint32_t source_address, size_t size_bytes, uint8_t noc = noc_index) {
        send_payload_flush_non_blocking_from_address<posted>(source_address, size_bytes, noc);
    }
    template <bool posted = false>
    FORCE_INLINE void send_current_slot_non_blocking(
        uint32_t payload_source_l1_addr,
        size_t payload_size_bytes,
        uint32_t header_source_l1_addr,
        uint8_t noc = noc_index) {
        const uint32_t slot = current_slot_address();
        noc_async_write(payload_source_l1_addr, mux_addr(slot + sizeof(PACKET_HEADER_TYPE)), payload_size_bytes, noc);
        noc_async_write(header_source_l1_addr, mux_addr(slot), sizeof(PACKET_HEADER_TYPE), noc);
        commit(noc);
    }

    // ---- close: same steps as close_start/close_finish ----
    void close_start() {
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(buffer_index_address + offsetof(SenderChannelProducerCursor, write_counter)), write_counter);
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(buffer_index_address + offsetof(SenderChannelProducerCursor, write_index)),
            static_cast<uint32_t>(write_index));
        noc_inline_dw_write<InlineWriteDst::L1>(
            mux_addr(connection_handshake_address), connection_interface::close_connection_request_value);
    }
    void close_finish() {
        noc_async_write_barrier();
        WAYPOINT("LMCW");
        while (*local_teardown_ptr != 1) {
            invalidate_l1_cache();
        }
        WAYPOINT("LMCD");
        *local_teardown_ptr = 0;
    }
    void close() {
        close_start();
        close_finish();
    }

private:
    // THE substitution: publish the new write counter with a plain write (the V1 mux
    // client decrements a stream register here). Ordered behind the slot writes.
    FORCE_INLINE void commit(uint8_t noc) {
        write_counter++;
        noc_inline_dw_write<InlineWriteDst::L1>(mux_addr(producer_write_counter_address), write_counter, 0xf, noc);
        write_index = (write_index + 1 == NUM_BUFFERS) ? 0 : static_cast<uint8_t>(write_index + 1);
    }
};

// Same contract as wait_for_fabric_endpoint_ready(), but reads the status word with a
// 16 B NOC read into a scratch that MUST be +0 mod 64 (see the alignment contract);
// prefer WorkerToL2cpuMuxSender::wait_for_ready(), which uses its own aligned scratch.
FORCE_INLINE void wait_for_l2cpu_mux_ready(
    uint8_t mux_x, uint8_t mux_y, size_t mux_status_address, uint32_t local_scratch_address) {
    const uint64_t noc_addr = get_noc_addr(mux_x, mux_y, mux_status_address);
    auto* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_address);
    scratch[0] = EDMStatus::TERMINATED;
    do {
        noc_async_read(noc_addr, local_scratch_address, 16);
        noc_async_read_barrier();
        invalidate_l1_cache();
    } while (scratch[0] != EDMStatus::READY_FOR_TRAFFIC);
}

// Mirrors build_connection_to_fabric_endpoint(): same argument order, so host code that
// emits V1 mux connection args needs no change beyond the addresses it passes.
template <uint8_t NUM_BUFFERS>
WorkerToL2cpuMuxSender<NUM_BUFFERS> build_connection_to_l2cpu_mux(
    uint8_t mux_x,
    uint8_t mux_y,
    uint8_t /*channel_id (implied by the addresses)*/,
    uint8_t /*num_buffers_per_channel (= NUM_BUFFERS)*/,
    size_t channel_buffer_size_bytes,
    size_t channel_base_address,
    size_t connection_info_address,
    size_t connection_handshake_address,
    size_t flow_control_address,  // producer write-counter word on the mux
    size_t buffer_index_address,
    uint32_t local_flow_control_address,
    uint32_t local_teardown_address,
    uint32_t local_buffer_index_address) {
    WorkerToL2cpuMuxSender<NUM_BUFFERS> s;
    s.edm_noc_x = mux_x;
    s.edm_noc_y = mux_y;
    s.buffer_size_bytes = channel_buffer_size_bytes;
    s.channel_base_address = channel_base_address;
    s.connection_info_address = connection_info_address;
    s.connection_handshake_address = connection_handshake_address;
    s.producer_write_counter_address = flow_control_address;
    s.buffer_index_address = buffer_index_address;
    // Local scratch: >= 192 B starting at local_buffer_index_address (rounded up to 64 B).
    const uint32_t scratch = (local_buffer_index_address + 63u) & ~63u;
    s.local_cursor_address = scratch;         // LM_CURSOR(ch) is +0 mod 64
    s.local_status_scratch = scratch + 0x40;  // LM_STATUS / LM_CONFIG_GEN are +0 mod 64
    s.local_flow_control_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        (local_flow_control_address % 64u == 0x30u) ? local_flow_control_address : scratch + 0x70);
    s.local_teardown_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_teardown_address);
    return s;
}

}  // namespace tt::tt_fabric

// Register the client with the fabric API's sender-type traits so the generic
// linear/mesh helpers (fabric_unicast_noc_unicast_write*, *_with_state, *_atomic_inc*)
// accept a WorkerToL2cpuMuxSender exactly like a WorkerToFabricMuxSender.
namespace tt::tt_fabric::common::experimental {
template <uint8_t N>
struct is_mux_sender<tt::tt_fabric::WorkerToL2cpuMuxSender<N>> : std::true_type {};
}  // namespace tt::tt_fabric::common::experimental

namespace tt::tt_fabric {

// V1-named helpers, overloaded for the L2CPU client.
template <uint8_t N>
FORCE_INLINE void fabric_client_connect(WorkerToL2cpuMuxSender<N>& h) {
    h.open();
}
template <uint8_t N>
FORCE_INLINE void fabric_client_connect_start(WorkerToL2cpuMuxSender<N>& h) {
    h.open_start();
}
template <uint8_t N>
FORCE_INLINE void fabric_client_connect_finish(WorkerToL2cpuMuxSender<N>& h) {
    h.open_finish();
}
template <uint8_t N>
FORCE_INLINE void fabric_client_disconnect(WorkerToL2cpuMuxSender<N>& h) {
    h.close();
}

// assumes packet header is correctly populated
template <uint8_t N>
FORCE_INLINE void fabric_async_write(
    WorkerToL2cpuMuxSender<N>& h,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_payload_address,
    uint32_t packet_payload_size_bytes) {
    h.wait_for_empty_write_slot();
    h.send_payload_without_header_non_blocking_from_address(source_payload_address, packet_payload_size_bytes);
    h.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// assumes packet header is correctly populated (header-only packet, e.g. an atomic inc)
template <uint8_t N>
FORCE_INLINE void fabric_atomic_inc(
    WorkerToL2cpuMuxSender<N>& h, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
    h.wait_for_empty_write_slot();
    h.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric
