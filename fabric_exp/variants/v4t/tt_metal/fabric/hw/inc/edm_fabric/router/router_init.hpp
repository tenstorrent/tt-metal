// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Init and teardown, the same skeleton for every role. On a two-ERISC router both ERISCs run this file's sequence, each
// doing its own role's part of every phase, and meet at the end of each phase (sync_erisc). The sender ERISC owns the
// sender side (sender channels, worker connections, static connections from upstream routers); the receiver ERISC
// owns the receiver side (receiver channels, downstream connections). The phase number travels in the sync value, so
// the two can never pass different syncs as one.

#include "router_common.hpp"
#include "router_downstream.hpp"

namespace fabric_router {

constexpr bool is_sender_role = any_sender_channel_serviced;
constexpr bool is_receiver_role = any_receiver_channel_serviced;
constexpr bool is_teardown_master = MY_ERISC_ID == 0;

// ---------------------------------------------------------------------------------------------------------------------
// ERISC sync
// ---------------------------------------------------------------------------------------------------------------------

enum class Phase : uint32_t {
    STREAMS_SET = 1,       // stream registers initialized
    SETUP_DONE = 2,        // channels and connections set up locally
    HANDSHAKE_DONE = 3,    // Ethernet and local-router handshakes done
    DOWNSTREAMS_OPEN = 4,  // downstream connections opened
    CMD_BUFS_SET = 5,      // forwarding command buffers programmed
    CONNECTED = 6,         // static upstream connections in
    TEARDOWN_START = 7,
    TRANSACTIONS_DONE = 8,
    TEARDOWN_DONE = 9,
};

// Both ERISCs of a router pass every phase together: the teardown master writes the phase, the other answers. No
// termination check here: both see the same termination signal, and one leaving a sync early would leave the other
// waiting for a value that never comes.
__attribute__((noinline)) void sync_erisc(Phase phase) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        const uint32_t start = 0x0A00 | static_cast<uint32_t>(phase);
        const uint32_t answer = 0x0B00 | static_cast<uint32_t>(phase);
        if constexpr (is_teardown_master) {
            write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(start);
            while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) != answer) {
                router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
            }
            write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(0);
        } else {
            while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) != start) {
                router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
            }
            write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(answer);
        }
    }
}

FORCE_INLINE void postcode(tt::tt_fabric::EDMStatus status) {
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(status);
#endif
}

FORCE_INLINE volatile tt_l1_ptr tt::tt_fabric::EDMStatus* edm_status() {
    return reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);
}

// ---------------------------------------------------------------------------------------------------------------------
// Ethernet TX queues
// ---------------------------------------------------------------------------------------------------------------------

// Keepalive: the TX queues' sequence-update timeout. A TX queue with nothing to send returns its paired RX queue's
// acknowledgements only in a keepalive, so one-way traffic waits on this period once the sender's window is full. The
// teardown master sets it on both queues at init and restores the firmware's values at teardown; 0 (the builder's
// default off Blackhole) leaves the firmware's value (RESULTS §15, §98: Galaxy +5 to +7 without it).
constexpr uint32_t keepalive_timeout = get_named_compile_time_arg_val("ETH_TXQ_SEQ_UPDATE_TIMEOUT");
constexpr bool sets_keepalive = keepalive_timeout != 0 && is_teardown_master;
uint32_t firmware_keepalive[2];

__attribute__((noinline)) void set_keepalive() {
    if constexpr (sets_keepalive) {
        firmware_keepalive[0] = eth_txq_reg_read(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, keepalive_timeout);
        if constexpr (receiver_txq_id != sender_txq_id) {
            firmware_keepalive[1] = eth_txq_reg_read(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, keepalive_timeout);
        }
    }
}

__attribute__((noinline)) void restore_keepalive() {
    if constexpr (sets_keepalive) {
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_keepalive[0]);
        if constexpr (receiver_txq_id != sender_txq_id) {
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_keepalive[1]);
        }
    }
}

__attribute__((noinline)) void init_txqs() {
    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    set_keepalive();
    static_assert(receiver_txq_id == sender_txq_id || receiver_txq_id == 1);
    if constexpr (receiver_txq_id != sender_txq_id) {
        if constexpr (is_receiver_role) {
            // The receiver's queue carries completion credits; its counters start at 0.
            eth_enable_packet_mode(receiver_txq_id);
            for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
                reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)[i] = 0;
                reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)[i] = 0;
            }
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
        }
        if constexpr (is_sender_role) {
            for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
                reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)[i] = 0;
                reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address)[i] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------------------------------

// Before the first sync: the credit registers both link ends may touch, so they are set before the other side can.
__attribute__((noinline)) void init_link_streams() {
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_stream_if_used<to_sender_packets_acked_streams[0]>(0);
    init_stream_if_used<to_sender_packets_acked_streams[1]>(0);
    init_stream_if_used<to_sender_packets_completed_streams[0]>(0);
    init_stream_if_used<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<sender_free_slots_streams[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);
    init_ptr_val<sender_free_slots_streams[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);
    if constexpr (enable_context_switch) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(0);  // ContextSwitchState::NORMAL_EXECUTION
    }
}

// After the first sync (2D): the other receiver channels and sender channels.
__attribute__((noinline)) void init_2d_streams() {
    if constexpr (is_2d) {
        init_stream_if_used<to_receiver_packets_sent_streams[1]>(0);
        init_stream_if_used<to_receiver_packets_sent_streams[2]>(0);
        init_stream_if_used<to_sender_packets_acked_streams[2]>(0);
        init_stream_if_used<to_sender_packets_acked_streams[3]>(0);
        init_stream_if_used<to_sender_packets_acked_streams[4]>(0);
        for_each_index<8>([&](auto i) {
            constexpr size_t POSITION = decltype(i)::value + 2;
            init_stream_if_used<to_sender_packets_completed_streams[POSITION]>(0);
            if constexpr (services_fabric_position(POSITION)) {
                init_ptr_val<sender_free_slots_streams[POSITION]>(
                    SENDER_NUM_BUFFERS_ARRAY[compact_sender_for_fabric_position(POSITION)]);
            }
        });
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Telemetry block
// ---------------------------------------------------------------------------------------------------------------------

__attribute__((noinline)) void init_telemetry() {
    volatile tt_l1_ptr FabricTelemetry* telemetry =
        reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
    const auto* routing_table = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    memset(const_cast<FabricTelemetry*>(telemetry), 0, sizeof(FabricTelemetry));
    telemetry->static_info.mesh_id = routing_table->my_mesh_id;
    telemetry->static_info.device_id = routing_table->my_device_id;
    telemetry->static_info.direction = static_cast<uint8_t>(my_direction);
    telemetry->static_info.version = FABRIC_TELEMETRY_VERSION;
    telemetry->static_info.fabric_config = 0;
    telemetry->static_info.neighbor_mesh_id = 0xFFFF;  // set at the handshake
    telemetry->static_info.neighbor_device_id = 0xFF;
    telemetry->static_info.supported_stats = static_cast<DynamicStatistics>(
        DynamicStatistics::BANDWIDTH | DynamicStatistics::ROUTER_STATE | DynamicStatistics::HEARTBEAT_TX |
        DynamicStatistics::HEARTBEAT_RX);
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = RouterState::INITIALIZING;
    } else {
        telemetry->dynamic_info.erisc[0].router_state = RouterState::INITIALIZING;
        telemetry->dynamic_info.erisc[1].router_state = RouterState::INITIALIZING;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Runtime args
// ---------------------------------------------------------------------------------------------------------------------

// The router's runtime args (erisc_datamover_builder.cpp get_runtime_args): sender-channel connection semaphores and
// producer-cursor blocks, then per VC the downstream connections (see DownstreamArgs), then the downstream teardown
// semaphores, then the UDM relay block.
struct RuntimeArgs {
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> connection_semaphores;
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> producer_cursors;
    uint32_t vc0_mask;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> vc0_ring_bases;
    uint32_t vc0_noc_x;
    uint32_t vc0_noc_y;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> vc0_registration;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> vc0_location_info;
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> teardown_sems;

    DownstreamArgs vc0() const {
        return {vc0_mask & 0xF, vc0_ring_bases.data(), vc0_location_info.data(), vc0_registration.data(),
                teardown_sems.data(), vc0_noc_x, vc0_noc_y};
    }
};

__attribute__((noinline)) void read_runtime_args(RuntimeArgs& a) {
    size_t i = 0;
    for (auto& v : a.connection_semaphores) {
        v = get_arg_val<uint32_t>(i++);
    }
    for (auto& v : a.producer_cursors) {
        v = get_arg_val<uint32_t>(i++);
    }
    a.vc0_mask = get_arg_val<uint32_t>(i++);
    for (auto& v : a.vc0_ring_bases) {
        v = get_arg_val<uint32_t>(i++);
    }
    a.vc0_noc_x = get_arg_val<uint32_t>(i++);
    a.vc0_noc_y = get_arg_val<uint32_t>(i++);
    for (auto& v : a.vc0_registration) {
        v = get_arg_val<uint32_t>(i++);
    }
    for (auto& v : a.vc0_location_info) {
        v = get_arg_val<uint32_t>(i++);
    }
    i += NUM_DOWNSTREAM_SENDERS_VC0;  // buffer-index semaphores (unused by router-to-router connections)
    for (auto& v : a.teardown_sems) {
        v = get_arg_val<uint32_t>(i++);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Handshakes
// ---------------------------------------------------------------------------------------------------------------------

// The Ethernet handshake with the far router, then (with the host's start signal) the handshake among this chip's
// routers: the local master collects every router's arrival, releases them, and passes on READY_FOR_TRAFFIC.
__attribute__((noinline)) void run_handshakes() {
    if constexpr (enable_ethernet_handshake) {
        const auto* routing_table = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
        auto* termination = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
        if constexpr (is_handshake_sender) {
            erisc::datamover::handshake::fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr, routing_table->my_mesh_id, routing_table->my_device_id, termination,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        } else {
            erisc::datamover::handshake::fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr, routing_table->my_mesh_id, routing_table->my_device_id, termination,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        }
        auto* info = reinterpret_cast<volatile tt_l1_ptr erisc::datamover::handshake::handshake_info_t*>(handshake_addr);
        auto* telemetry =
            reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
        telemetry->static_info.neighbor_mesh_id = info->neighbor_mesh_id;
        telemetry->static_info.neighbor_device_id = info->neighbor_device_id;
        *edm_status() = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;

        if constexpr (wait_for_host_signal) {
            const uint32_t sync = edm_local_sync_ptr_addr;
            if constexpr (is_local_handshake_master) {
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(sync, num_local_edms - 1, termination);
                notify_subordinate_routers(edm_channels_mask, std::numeric_limits<uint32_t>::max(), sync, num_local_edms);
            } else {
                notify_master_router(local_handshake_master_eth_chan, sync);
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(sync, num_local_edms, termination);
            }
            *edm_status() = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;
            wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                (uint32_t)edm_status(), tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC, termination);
            if constexpr (is_local_handshake_master) {
                notify_subordinate_routers(
                    edm_channels_mask, local_handshake_master_eth_chan, (uint32_t)edm_status(),
                    tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------------------------------------------------

template <size_t CH>
using TeardownTrids = WriteTransactionIdTracker<
    RECEIVER_NUM_BUFFERS_ARRAY[CH], NUM_TRANSACTION_IDS, RX_CH_TRID_STARTS[CH], tt::tt_fabric::edm_to_local_chip_noc,
    tt::tt_fabric::edm_to_downstream_noc>;

__attribute__((noinline)) void teardown() {
    sync_erisc(Phase::TEARDOWN_START);
    if constexpr (is_receiver_channel_serviced[0]) {
        TeardownTrids<0>{}.all_buffer_slot_transactions_acked();  // waits on every transaction ID of the channel
    }
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Dynamic noc mode not supported");
    ncrisc_noc_counters_init();  // the router's writes do not all update the software counters
    sync_erisc(Phase::TRANSACTIONS_DONE);
    if constexpr (wait_for_host_signal && is_local_handshake_master) {
        auto* termination = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
        notify_subordinate_routers(edm_channels_mask, local_handshake_master_eth_chan, (uint32_t)termination, *termination);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#if defined(ARCH_BLACKHOLE)
    noc_clear_packet_tags(noc_index);  // tagged writes do not reset the command buffers' transaction IDs
#endif
    sync_erisc(Phase::TEARDOWN_DONE);
    restore_keepalive();
    if constexpr (is_teardown_master) {
        *edm_status() = tt::tt_fabric::EDMStatus::TERMINATED;
    }
}

}  // namespace fabric_router
