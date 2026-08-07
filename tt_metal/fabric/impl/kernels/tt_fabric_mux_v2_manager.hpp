// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2_kernel_common.hpp"

namespace tt::tt_fabric::mux_v2::kernel {

// noc_index is a compile-time constant (NOC_INDEX); forwarder owns the sibling NOC.
constexpr uint8_t get_forwarder_noc() { return static_cast<uint8_t>(1 - noc_index); }

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

inline void publish_pending_read_counter_to_worker(ManagerClientState& state, uint32_t logical_channel_id) {
    if (state.published_read_counter == state.local_read_counter) {
        return;
    }

    // Posted + flush=false: skip the BH spoof-write dual *_writes_sent wait (~tens of cycles).
    // Per-channel scratch avoids cross-worker counter clobber; same-channel overwrite with a
    // newer free counter is benign (matches fabric router credit-notify rationale).
    // On Wormhole, flush/customized_src are unused by the inline-write path.
    noc_inline_dw_write<InlineWriteDst::L1, /*posted=*/true, /*flush=*/false>(
        state.cached_worker_semaphore_address,
        state.local_read_counter,
        0xf,
        noc_index,
        NOC_UNICAST_WRITE_VC,
        get_credit_notify_scratch_address(logical_channel_id));
    state.published_read_counter = state.local_read_counter;
}

inline void establish_worker_connection(
    ManagerClientState& state,
    volatile tt_l1_ptr tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr) {
    state.connection_established = true;
    cache_worker_semaphore_address(state, worker_location_info_ptr);
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
    noc_semaphore_inc(worker_teardown_semaphore_address, 1, noc_index);
}

inline void record_retired_worker_credit(ManagerClientState& state) { state.local_read_counter += 1; }

// Bounded head poll: with flush=false credit notify, the next retire often races ahead of
// HW `_sent`. A short spaced retry absorbs that gap without returning to service_client.
constexpr uint32_t kTridHeadSentRetryAttempts = 10;

inline void retire_published_ring_entries(
    volatile tt_l1_ptr tt::tt_fabric::FabricMuxV2SharedTridRingHeader* shared_ring_header_ptr,
    uint32_t& read_count,
    uint32_t write_count_snapshot,
    std::array<ManagerClientState, kMaxRuntimeChannels>& client_states) {
    auto shared_ring_entries_ptr = get_shared_ring_entries_ptr();
    const uint32_t starting_read_count = read_count;
    constexpr uint8_t forwarder_noc = get_forwarder_noc();

    while ((write_count_snapshot - read_count) != 0) {
        const uint32_t ring_index_and_trid = read_count & ct_args::shared_trid_ring_mask;
        const uint32_t logical_channel_id = shared_ring_entries_ptr[ring_index_and_trid];

        uint32_t attempt = 0;
        for (; attempt < kTridHeadSentRetryAttempts; ++attempt) {
            if (ncrisc_noc_nonposted_write_with_transaction_id_sent(forwarder_noc, ring_index_and_trid)) {
                break;
            }
            asm volatile("nop");
            asm volatile("nop");
        }
        if (attempt == kTridHeadSentRetryAttempts) {
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
        state.finalized = true;
        finalized_client_count += 1;

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

    publish_pending_read_counter_to_worker(state, logical_channel_id);
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

    // Steady state: service clients + opportunistic TRID retirement.
    while (true) {
        for (uint32_t logical_channel_id = 0; logical_channel_id < ct_args::num_channels; ++logical_channel_id) {
            retire_published_ring_entries(
                shared_ring_header_ptr, shared_ring_read_count, shared_ring_header_ptr->write_count, client_states);

            service_client(logical_channel_id, client_states[logical_channel_id], finalized_client_count);
        }

        if (finalized_client_count == ct_args::num_channels) {
            shared_control_ptr->drain_initiated = 1;
            break;
        }
    }

    // Drain: retire TRIDs until forwarder done; skip all client servicing.
    // Once drain is initiated, clients are finalized and no future traffic can arrive.
    while (shared_control_ptr->forwarder_done == 0) {
        if (shared_control_ptr->forwarder_stop_tracking == 0) {
            retire_published_ring_entries(
                shared_ring_header_ptr, shared_ring_read_count, shared_ring_header_ptr->write_count, client_states);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}

}  // namespace tt::tt_fabric::mux_v2::kernel
