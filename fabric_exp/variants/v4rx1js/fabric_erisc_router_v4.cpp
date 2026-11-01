// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric ERISC router: moves fabric packets across one Ethernet link.
//
// A router has sender channels, which take packets from a worker on this chip (channel 0 of a VC) or from the router
// facing another direction (forwarding channels) and send them over the link, and receiver channels, which take packets
// off the link and write them to this chip, forward them to the next router, or both. Credits flow back the same way.
//
// On a two-ERISC Blackhole router, ERISC 0 runs the sender program (router/router_sender.hpp) and ERISC 1 the receiver
// program (router/router_receiver.hpp). Both run one init and teardown skeleton (router/router_init.hpp), each doing
// its own role's part, and meet at every phase.
//
// Bring-up status: two-ERISC Blackhole routers, 1D and 2D (meshes, lines, rings, tori: bubble flow control and
// first-level acks), VC0 and VC1 (multi-mesh fabrics: intermesh landing, crossover and exits), strict atomics;
// single-ERISC Blackhole routers (one ERISC runs both roles, router/router_single.hpp; functional, tuned on Wormhole);
// and Tensix mux mode on them, VC0 (the fabric's muxes feed the worker channel and take the forwards); UDM mode on 2D
// meshes (packets for this chip go to the fabric's relay; router/router_downstream.hpp); VC2 on 2D single-mesh routers
// (neighbour exchange: a second worker channel, and a receiver channel that only delivers); and the speedy path (VC0
// worker-only routers whose receiver only delivers, e.g. neighbour exchange: the generic loop specializes to them, and
// VC0's receiver channel skips decode and admission); with the host's telemetry options (the perf recorder and the
// dynamic stats; router/router_telemetry.hpp); and channel trimming (the capture, router/router_receiver.hpp's
// record_forwarded; a replay's routers with channels compiled out, an ERISC left with nothing to service running rounds
// only, router/router_round.hpp's run_idle).
// Other configurations fail to compile here (static_assert below) until their family is brought up.

#define FABRIC_ERISC_ROUTER_V4 1

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"

#include "hostdev/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/edm_fabric_counters.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_tmp_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_packet_recorder.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_channel_traits.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include "noc_overlay_parameters.h"
#include "api/alignment.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
#include "hostdevcommon/fabric_telemetry_msgs.h"
#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#include <limits>

using namespace tt::tt_fabric;

#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_init.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp"

namespace fabric_router {

// VC1 traffic would go router to router around the muxes, but the host gives VC1's sender channels no slots there.
static_assert(!(tensix_mux_mode && has_vc1), "fabric router v4: VC1 with the Tensix mux extension is not supported");

// UDM as the host builds it (compute_mesh_router_builder.cpp, connect_to_local_tensix_builder): one ERISC (any Tensix
// extension), 2D meshes, one local Tensix connection (the relay), mux mode off.
constexpr bool udm_config_supported = NUM_ACTIVE_ERISCS == 1 && is_2d && !has_vc1 && !enable_deadlock_avoidance &&
                                      !tensix_mux_mode && num_ds_or_local_tensix_connections == 1;

// VC2 as the host builds it (FabricBuilderConfig: Blackhole 2D, not UDM or the mux), here without VC1: one worker channel
// into a receiver channel of its own, counter credits.
constexpr bool vc2_config_supported = is_2d && !has_vc1 && ACTUAL_VC2_SENDER_CHANNELS == 1 &&
                                      receiver_channel_uses_counter_credits(VC2_RECEIVER_CHANNEL);

// The speedy path as the host selects it (ENABLE_SPEEDY_VC0; main's speedy-path static_asserts): no bubble flow control
// or first-level acks on VC0, VC0's receiver forwarding off. Its credit amortization args do not apply here (credits
// go back per completion group, as everywhere).
constexpr bool speedy_config_supported =
    !enable_deadlock_avoidance && !ENABLE_FIRST_LEVEL_ACK_VC0 && disable_rx_ch0_forwarding && !udm_mode;

// Bring-up gate: the configurations this router supports so far.
constexpr bool brought_up =
    is_blackhole_arch &&
    (NUM_ACTIVE_ERISCS == 2 ? multi_txq_enabled && !(is_sender_role && is_receiver_role) : NUM_ACTIVE_ERISCS == 1) &&
    (!has_vc2 || vc2_config_supported) && NUM_RECEIVER_CHANNELS == 1 + (has_vc1 ? 1 : 0) + (has_vc2 ? 1 : 0) &&
    (!has_vc1 || is_2d) &&
    !ENABLE_FIRST_LEVEL_ACK_VC1 && (!udm_mode || udm_config_supported) &&
    (!super_speedy_mode || speedy_config_supported) &&
    !ENABLE_RISC_CPU_DATA_CACHE && !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA &&
    (perf_telemetry_mode == PerfTelemetryRecorderType::NONE ||
     perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH) &&
    (!tensix_mux_mode || NUM_ACTIVE_ERISCS == 1) &&
    (num_ds_or_local_tensix_connections == 0 || tensix_mux_mode || udm_mode);
static_assert(brought_up, "fabric router v4: this configuration is not brought up yet (see the file header)");

// ---------------------------------------------------------------------------------------------------------------------
// Sender side: worker interfaces and static connections
// ---------------------------------------------------------------------------------------------------------------------

using WorkerInterfaces = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
    std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

// The worker interface of every sender channel: its location info, connection semaphore, starting credits (0 for a
// worker channel, whose credits are counters; the ring size for a forwarding channel) and the source word of its worker
// credits (a worker channel's: router_sender.hpp, WorkerCredit).
__attribute__((noinline)) void init_worker_interfaces(WorkerInterfaces& interfaces, const RuntimeArgs& args) {
    for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
        constexpr size_t CH = decltype(i)::value;
        constexpr bool vc2_worker = CH == VC2_LOCAL_CHANNEL_START;
        auto* info =
            reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(sender_connection_info_addrs[CH]);
        info->edm_read_counter = 0;
        new (&interfaces.template get<CH>())
            tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY[CH]>(
                info,
                0,
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(args.connection_semaphores[CH]),
                sender_channel_ack_cmd_buf_ids[CH],
                CH == 0 || vc2_worker ? 0 : SENDER_NUM_BUFFERS_ARRAY[CH],
                vc2_worker ? sender::WorkerCredit<CH>::source : notify_worker_of_read_counter_update_src_address);
    });
}

// Of VC0's and VC1's downstream connections, VC's.
template <size_t VC, typename Vc0, typename Vc1>
FORCE_INLINE auto& of_vc(Vc0& vc0, Vc1& vc1) {
    if constexpr (VC == 0) {
        return vc0;
    } else {
        return vc1;
    }
}

// Wait until an upstream router asks to connect on `connection_semaphore`, or termination. One body for every channel.
__attribute__((noinline)) void wait_for_connect_request(volatile tt_l1_ptr uint32_t* connection_semaphore) {
    uint32_t polls = 0;
    while (!connect_is_requested(*connection_semaphore) && !terminating()) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (polls++ == 100000) {
            polls = 0;
            run_routing();
        }
    }
}

// Wait for each forwarding channel's upstream router to connect, and cache its credit address.
__attribute__((noinline)) void wait_for_upstream_routers(WorkerInterfaces& interfaces, sender::UpstreamCredits& upstream) {
    for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
        constexpr size_t CH = decltype(i)::value;
        if constexpr (is_sender_channel_serviced[CH] && sender_ch_wait_static_connection[CH]) {
            auto& interface = interfaces.template get<CH>();
            wait_for_connect_request(interface.connection_live_semaphore);
            interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
            static_assert(((sender::all_forwarding >> CH) & 1u) != 0, "a static upstream connection is a forwarding channel's");
            upstream.addr[sender::record_index<CH>()] = interface.cached_worker_semaphore_address;
        }
    });
}

}  // namespace fabric_router

void kernel_main() {
    using namespace fabric_router;
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.reset();
    }
    postcode(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
    router_state()->state = RouterState::INITIALIZING;
    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    init_telemetry();
    telemetry::init();

    // Phase: stream registers.
    init_txqs();
    postcode(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);
    init_link_streams();
    sync_erisc(Phase::STREAMS_SET);
    init_2d_streams();
    postcode(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);
    if constexpr (code_profiling_enabled_timers_bitfield != 0) {
        clear_code_profiling_buffer(code_profiling_buffer_base_addr);
    }

    // Phase: local setup.
    RuntimeArgs args;
    read_runtime_args(args);
    WorkerInterfaces interfaces{};
    // The receiver's downstream links per VC: VC0's, and VC1's on VC1 routers (an unused stand-in otherwise).
    DownstreamLinks<0> links_vc0{};
    DownstreamLinks<vc1_links_vc> links_vc1{};
    if constexpr (is_sender_role) {
        for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
            constexpr size_t CH = decltype(i)::value;
            if constexpr (is_sender_channel_serviced[CH]) {
                *reinterpret_cast<volatile uint32_t*>(args.connection_semaphores[CH]) = 0;
                // The producer reads the whole cursor block at connection open, so every word starts at 0.
                auto* cursor = reinterpret_cast<volatile uint32_t*>(args.producer_cursors[CH]);
                for (size_t w = 0; w < sizeof(tt::tt_fabric::SenderChannelProducerCursor) / sizeof(uint32_t); ++w) {
                    cursor[w] = 0;
                }
            }
        });
    }
    postcode(tt::tt_fabric::EDMStatus::STARTED);
    *edm_status() = tt::tt_fabric::EDMStatus::STARTED;
    postcode(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
    if constexpr (is_receiver_role) {
        init_downstreams(links_vc0, args.vc0());
        if constexpr (has_vc1) {
            init_downstreams(links_vc1, args.vc1());
        }
        if constexpr (udm_mode) {
            init_relay(links_vc0.relay, args.relay);
        }
        auto receiver_channels =
            ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();
        receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
    }
    postcode(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);
    if constexpr (is_sender_role) {
        auto sender_channels =
            SenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>::make();
        sender_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
        init_worker_interfaces(interfaces, args);
    }
    sync_erisc(Phase::SETUP_DONE);

    // Phase: handshakes.
    run_handshakes();
    sync_erisc(Phase::HANDSHAKE_DONE);
    postcode(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);

    // Phase: downstream connections. As main, a VC's connections are opened when the host's open gate says so: VC0's
    // when receiver channel 0 is serviced, also where no packet uses them (an intermesh router's VC0 connections with
    // crossover), since the downstream routers wait for them; VC1's when the receiver channel that forwards on them is
    // serviced (channel 0 with crossover, else channel 1; channel trimming may leave a crossover router's channel 1
    // unserviced). In Tensix mux mode the downstreams are the other directions' muxes, opened once they are set up; in
    // UDM the relay is opened after the downstream routers, once it is set up (as main).
    if constexpr (is_receiver_role) {
        wait_for_downstream_muxes();
        if constexpr (open_downstream_vc0_connections) {
            open_downstreams<0>(args.vc0());
        }
        if constexpr (has_vc1 && open_downstream_vc1_connections) {
            open_downstreams<1>(args.vc1());
        }
        if constexpr (udm_mode) {
            open_relay(args.relay);
        }
    }
    sync_erisc(Phase::DOWNSTREAMS_OPEN);
    postcode(tt::tt_fabric::EDMStatus::VCS_OPENED);
    if constexpr (is_receiver_role) {
        preset_downstream_cmd_bufs<0>(args.vc0());
    }
    sync_erisc(Phase::CMD_BUFS_SET);
    postcode(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);

    // Phase: upstream connections.
    WAYPOINT("FSCW");
    sender::UpstreamCredits upstream{};
    if constexpr (is_sender_role) {
        wait_for_upstream_routers(interfaces, upstream);
    }
    WAYPOINT("FSCD");
    sync_erisc(Phase::CONNECTED);
    postcode(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE);
    if constexpr (is_teardown_master) {
        router_state()->state = RouterState::INITIALIZING;
    }

    // One ERISC runs the single-ERISC loop whatever channel trimming left it (a role without channels makes empty
    // passes); on two ERISCs, an ERISC trimming left nothing to service runs rounds only.
    if constexpr (NUM_ACTIVE_ERISCS == 1) {
        Downstreams<0> downstreams_vc0(links_vc0);
        Downstreams<vc1_links_vc> downstreams_vc1(links_vc1);
        single::run(
            interfaces,
            upstream,
            of_vc<receiver::Channel<0>::downstream_vc>(downstreams_vc0, downstreams_vc1),
            downstreams_vc1);
    } else if constexpr (is_sender_role) {
        sender::run(interfaces, upstream);
    } else if constexpr (is_receiver_role) {
        // Receiver channel 0 forwards on VC0's connections, or on VC1's with crossover; channel 1 on VC1's.
        Downstreams<0> downstreams_vc0(links_vc0);
        Downstreams<vc1_links_vc> downstreams_vc1(links_vc1);
        receiver::run(of_vc<receiver::Channel<0>::downstream_vc>(downstreams_vc0, downstreams_vc1), downstreams_vc1);
    } else {
        run_idle();
    }
    WAYPOINT("LPDN");
    teardown();
    WAYPOINT("TEDN");
}
