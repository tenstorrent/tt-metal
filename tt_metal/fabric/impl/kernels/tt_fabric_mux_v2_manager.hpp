// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_kernel_common.hpp"

namespace tt::tt_fabric::mux_v2::kernel {

inline uint8_t get_forwarder_noc() {
    // The forwarder issues tracked downstream writes on its local noc_index directly.
    // This manager kernel runs on the sibling RISC, so it retires those TRIDs on 1 - noc_index.
    return static_cast<uint8_t>(1 - noc_index);
}

struct ManagerClientState {
    uint32_t local_read_counter = 0;
    uint32_t published_read_counter = 0;
    uint64_t cached_worker_semaphore_address = 0;
    bool connection_established = false;
    bool finalized = false;
};

inline void cache_worker_semaphore_address(
    ManagerClientState& state,
    volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr) {
    const auto& worker_info = *worker_location_info_ptr;
    state.cached_worker_semaphore_address = get_noc_addr(
        static_cast<uint32_t>(worker_info.worker_xy.x),
        static_cast<uint32_t>(worker_info.worker_xy.y),
        worker_info.worker_semaphore_address);
}

inline void publish_read_counter_to_worker(ManagerClientState& state) {
    noc_inline_dw_write<InlineWriteDst::L1, true, true>(
        state.cached_worker_semaphore_address, state.local_read_counter, 0xf, tt::tt_fabric::worker_handshake_noc);
    state.published_read_counter = state.local_read_counter;
}

inline void publish_pending_read_counter_to_worker(ManagerClientState& state) {
    if (state.published_read_counter == state.local_read_counter) {
        return;
    }

    publish_read_counter_to_worker(state);
}

inline void establish_worker_connection(
    ManagerClientState& state,
    volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr) {
    state.connection_established = true;
    cache_worker_semaphore_address(state, worker_location_info_ptr);
    publish_read_counter_to_worker(state);
}

inline void teardown_worker_connection(
    ManagerClientState& state,
    uint32_t logical_channel_id,
    volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr) {
    const auto& worker_info = *worker_location_info_ptr;
    const uint64_t worker_teardown_semaphore_address = get_noc_addr(
        static_cast<uint32_t>(worker_info.worker_xy.x),
        static_cast<uint32_t>(worker_info.worker_xy.y),
        worker_info.worker_teardown_semaphore_address);

    auto connection_live_semaphore = get_connection_handshake_ptr(logical_channel_id);
    connection_live_semaphore[0] = tt::tt_fabric::connection_interface::unused_connection_value;
    worker_location_info_ptr->edm_read_counter = state.local_read_counter;
    noc_semaphore_inc(worker_teardown_semaphore_address, 1, tt::tt_fabric::worker_handshake_noc);
    state.connection_established = false;
    state.published_read_counter = state.local_read_counter;
}

inline void record_retired_worker_credit(ManagerClientState& state) { state.local_read_counter += 1; }

inline void retire_published_ring_entries(
    volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingHeader* shared_ring_header_ptr,
    uint32_t& read_count,
    uint32_t write_count_snapshot,
    std::array<ManagerClientState, kMaxRuntimeChannels>& client_states) {
    auto shared_ring_entries_ptr = get_shared_ring_entries_ptr();
    const uint32_t starting_read_count = read_count;

    while ((write_count_snapshot - read_count) != 0) {
        const uint32_t ring_index = read_count & ct_args::shared_trid_ring_mask;
        const uint32_t logical_channel_id = shared_ring_entries_ptr[ring_index];
        const uint32_t transaction_id = read_count & ct_args::forwarder_in_flight_trid_mask;
        if (!ncrisc_noc_nonposted_write_with_transaction_id_sent(get_forwarder_noc(), transaction_id)) {
            break;
        }

        record_retired_worker_credit(client_states[logical_channel_id]);
        read_count += 1;
    }

    if (read_count != starting_read_count) {
        shared_ring_header_ptr->read_count = read_count;
    }
}

inline void service_client(uint32_t logical_channel_id, ManagerClientState& state, uint32_t& finalized_client_count) {
    if (state.finalized) {
        return;
    }

    auto worker_location_info_ptr = get_connection_info_ptr(logical_channel_id);
    auto connection_handshake_ptr = get_connection_handshake_ptr(logical_channel_id);
    const uint32_t connection_state = connection_handshake_ptr[0];

    if (connection_state == tt::tt_fabric::connection_interface::close_connection_request_value) {
        if (!state.finalized) {
            state.finalized = true;
            finalized_client_count += 1;
        }

        teardown_worker_connection(state, logical_channel_id, worker_location_info_ptr);
        return;
    }

    if (!state.connection_established &&
        connection_state == tt::tt_fabric::connection_interface::open_connection_value) {
        establish_worker_connection(state, worker_location_info_ptr);
        return;
    }

    if (!state.connection_established) {
        return;
    }

    publish_pending_read_counter_to_worker(state);
}

inline void run_manager() {
    auto status_ptr = get_status_ptr();
    auto forwarder_ready_sem_ptr = get_forwarder_ready_sem_ptr();
    auto manager_init_done_sem_ptr = get_manager_init_done_sem_ptr();
    auto shared_control_ptr = get_shared_control_ptr();
    auto shared_ring_header_ptr = get_shared_ring_header_ptr();
    std::array<ManagerClientState, kMaxRuntimeChannels> client_states{};
    uint32_t finalized_client_count = 0;
    uint32_t shared_ring_read_count = 0;

    zero_per_channel_scalar_region(
        ct_args::connection_handshake_region_base_address,
        ct_args::per_channel_scalar_region_stride_bytes,
        ct_args::num_channels);
    initialize_shared_ring_header();
    initialize_shared_control_block(shared_control_ptr);
    noc_semaphore_set(manager_init_done_sem_ptr, 1);
    noc_semaphore_wait(forwarder_ready_sem_ptr, 1);

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;

    // Phase 1: Steady state — service clients + opportunistic TRID retirement.
    while (true) {
        invalidate_l1_cache();
        const uint32_t ring_write_count_snapshot = shared_ring_header_ptr->write_count;

        for (uint32_t logical_channel_id = 0; logical_channel_id < ct_args::num_channels; ++logical_channel_id) {
            retire_published_ring_entries(
                shared_ring_header_ptr, shared_ring_read_count, ring_write_count_snapshot, client_states);

            service_client(logical_channel_id, client_states[logical_channel_id], finalized_client_count);
        }

        if (finalized_client_count == ct_args::num_channels) {
            shared_control_ptr->drain_initiated = 1;
            break;
        }
    }

    // Phase 2: Drain — retire TRIDs until forwarder done, skip all client servicing.
    // Once drain is initiated, clients are finalized and no future traffic can arrive.
    while (shared_control_ptr->forwarder_done == 0) {
        invalidate_l1_cache();
        if (shared_control_ptr->forwarder_stop_tracking == 0) {
            retire_published_ring_entries(
                shared_ring_header_ptr, shared_ring_read_count, shared_ring_header_ptr->write_count, client_states);
        }
    }

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}

}  // namespace tt::tt_fabric::mux_v2::kernel
