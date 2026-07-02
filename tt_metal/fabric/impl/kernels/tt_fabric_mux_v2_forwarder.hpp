// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_kernel_common.hpp"
#include "api/debug/device_print.h"

namespace tt::tt_fabric::mux_v2::kernel {

constexpr uint32_t kForwarderServiceBurstSize = ct_args::forwarder_service_burst_size;

static_assert(
    ct_args::shared_trid_ring_capacity <= (NOC_MAX_TRANSACTION_ID + 1),
    "Forwarder shared TRID ring capacity exceeds available transaction IDs");

struct ForwarderSharedTridRingPublisher {
    volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingHeader* header_ptr = nullptr;
    volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingEntry* entries_ptr = nullptr;
    uint32_t cached_read_count = 0;
    uint32_t write_count = 0;

    void initialize() {
        header_ptr = get_shared_ring_header_ptr();
        entries_ptr = get_shared_ring_entries_ptr();
        cached_read_count = header_ptr->read_count;
        write_count = header_ptr->write_count;
    }

    bool has_free_slot() const { return (write_count - cached_read_count) < ct_args::shared_trid_ring_capacity; }

    void refresh_read_count() { cached_read_count = header_ptr->read_count; }

    uint32_t get_next_transaction_id() const { return write_count & ct_args::shared_trid_ring_mask; }

    void publish_logical_channel_id(uint32_t logical_channel_id) {
        entries_ptr[write_count & ct_args::shared_trid_ring_mask] = logical_channel_id;
        write_count += 1;
        header_ptr->write_count = write_count;
    }
};

inline void wait_until_downstream_slot_available(
    ForwarderSharedTridRingPublisher& shared_trid_ring,
    uint32_t& cached_downstream_free_slots,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection) {
    while (cached_downstream_free_slots == 0) {
        cached_downstream_free_slots = fabric_connection.get_num_free_write_slots();
        shared_trid_ring.refresh_read_count();
    }
}

inline void wait_for_trid_slot_available(ForwarderSharedTridRingPublisher& shared_trid_ring) {
    while (!shared_trid_ring.has_free_slot()) {
        invalidate_l1_cache();
        shared_trid_ring.refresh_read_count();
    }
}

template <bool DrainMode>
inline void try_forward_channel_packet(
    uint32_t logical_channel_id,
    ForwarderChannel& channel,
    ForwarderSharedTridRingPublisher& shared_trid_ring,
    uint32_t& cached_downstream_free_slots,
    uint8_t noc,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection) {
    if constexpr (DrainMode) {
        WAYPOINT("FDWS");
        while (cached_downstream_free_slots == 0) {
            fabric_connection.wait_for_empty_write_slot();
            cached_downstream_free_slots = fabric_connection.get_num_free_write_slots();
        }
        WAYPOINT("FDWE");
    } else {
        if (cached_downstream_free_slots == 0) {
            wait_until_downstream_slot_available(shared_trid_ring, cached_downstream_free_slots, fabric_connection);
        }
        if (!shared_trid_ring.has_free_slot()) {
            wait_for_trid_slot_available(shared_trid_ring);
        }
    }

    // Mirror V1's freshness point: invalidate just before consuming the worker-populated packet header.
    // invalidate_l1_cache();
    const uint32_t packet_l1_address = channel.get_current_slot_address();
    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_l1_address);
    const uint32_t packet_size_bytes = packet_header->get_payload_size_including_header();

    if constexpr (!DrainMode) {
        const uint32_t trid = shared_trid_ring.get_next_transaction_id();
        fabric_connection.send_current_slot_stateful_non_blocking_from_address_with_trid(
            packet_l1_address, packet_size_bytes, trid, noc);
        shared_trid_ring.publish_logical_channel_id(logical_channel_id);
    } else {
        fabric_connection.send_current_slot_stateful_non_blocking_from_address(
            packet_l1_address, packet_size_bytes, noc);
    }

    increment_local_update_ptr_val(channel.pending_packets_stream_id, 1);
    cached_downstream_free_slots -= 1;
    channel.advance_slot();
}

template <bool DrainMode>
inline bool service_channels(
    ForwarderContext& context,
    ForwarderSharedTridRingPublisher& shared_trid_ring,
    uint32_t& cached_downstream_free_slots,
    uint8_t noc,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection) {
    bool any_pending_channel = false;

    for (uint32_t channel_idx = 0; channel_idx < ct_args::num_channels; ++channel_idx) {
        auto& channel = context.channels[channel_idx];
        invalidate_l1_cache();
        if (!channel.has_pending_packets()) {
            continue;
        }

        any_pending_channel = true;
        try_forward_channel_packet<DrainMode>(
            channel_idx, channel, shared_trid_ring, cached_downstream_free_slots, noc, fabric_connection);
    }

    return any_pending_channel;
}

inline void run_forwarder(ForwarderContext& context) {
    auto forwarder_ready_sem_ptr = get_forwarder_ready_sem_ptr();
    auto manager_init_done_sem_ptr = get_manager_init_done_sem_ptr();
    auto shared_control_ptr = get_shared_control_ptr();
    ForwarderSharedTridRingPublisher shared_trid_ring{};

    initialize_pending_stream_state(context);
    noc_semaphore_set(forwarder_ready_sem_ptr, 1);
    auto fabric_connection = build_downstream_sender();
    // Tracked downstream writes are issued on the forwarder's local noc_index directly.
    // The manager retires them from the sibling RISC by polling 1 - noc_index.
    const uint8_t noc = noc_index;
    fabric_connection.open<false>();
    noc_semaphore_wait(manager_init_done_sem_ptr, 1);

    shared_trid_ring.initialize();
    fabric_connection.setup_stateful_send_cmd_bufs</*posted=*/false>(noc);
    const uint8_t data_noc_cmd_buf = fabric_connection.get_stateful_send_data_noc_cmd_buf();
    uint32_t cached_downstream_free_slots = fabric_connection.get_num_free_write_slots();

    WAYPOINT("FSTY");
    while (shared_control_ptr->drain_initiated == 0) {
        for (uint32_t service_pass = 0;
             service_pass < kForwarderServiceBurstSize && shared_control_ptr->drain_initiated == 0;
             ++service_pass) {
            if (!service_channels<false>(
                    context, shared_trid_ring, cached_downstream_free_slots, noc, fabric_connection)) {
                break;
            }
        }
    }

    WAYPOINT("FDRN");
    shared_control_ptr->forwarder_stop_tracking = 1;
    DEVICE_PRINT("Forwarder entering drain phase after manager initiated drain\n");
    while (true) {
        if (!service_channels<true>(context, shared_trid_ring, cached_downstream_free_slots, noc, fabric_connection)) {
            break;
        }
    }
    noc_clear_packet_tag(noc, data_noc_cmd_buf);
    WAYPOINT("FEMP");
    WAYPOINT("FCLS");
    DEVICE_PRINT("Forwarder finished draining all channels, closing connection\n");
    fabric_connection.close();
    DEVICE_PRINT("Forwarder closed connection\n");
    WAYPOINT("FCLD");
    WAYPOINT("FBRW");
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    WAYPOINT("FBRD");

    WAYPOINT("FDON");
    shared_control_ptr->forwarder_done = 1;

    DEVICE_PRINT("Forwarder exiting\n");
}

}  // namespace tt::tt_fabric::mux_v2::kernel
