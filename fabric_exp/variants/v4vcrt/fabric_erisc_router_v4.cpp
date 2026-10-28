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
// first-level acks), VC0 and VC1 (multi-mesh fabrics: intermesh landing, crossover and exits), strict atomics. Other
// configurations fail to compile here (static_assert below) until their family is brought up.

#define FABRIC_ERISC_ROUTER_V4 1

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"

#include "fabric/fabric_edm_packet_header.hpp"
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
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_rt.hpp"

namespace fabric_router {

// Bring-up gate: the configurations this router supports so far.
constexpr bool brought_up =
    is_blackhole_arch && NUM_ACTIVE_ERISCS == 2 && multi_txq_enabled && (is_sender_role != is_receiver_role) &&
    ACTUAL_VC2_SENDER_CHANNELS == 0 && NUM_RECEIVER_CHANNELS == (has_vc1 ? 2 : 1) && (!has_vc1 || is_2d) &&
    !ENABLE_FIRST_LEVEL_ACK_VC1 && !udm_mode && !super_speedy_mode &&
    !ENABLE_RISC_CPU_DATA_CACHE && !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !FABRIC_TELEMETRY_ANY_DYNAMIC_STAT &&
    perf_telemetry_mode == PerfTelemetryRecorderType::NONE && num_ds_or_local_tensix_connections == 0;
static_assert(brought_up, "fabric router v4: this configuration is not brought up yet (see the file header)");

// ---------------------------------------------------------------------------------------------------------------------
// Sender side: worker interfaces and static connections
// ---------------------------------------------------------------------------------------------------------------------

using WorkerInterfaces = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
    std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

constexpr std::array<size_t, MAX_NUM_SENDER_CHANNELS> connection_info_addrs = {
    local_sender_channel_0_connection_info_addr, local_sender_channel_1_connection_info_addr,
    local_sender_channel_2_connection_info_addr, local_sender_channel_3_connection_info_addr,
    local_sender_channel_4_connection_info_addr, local_sender_channel_5_connection_info_addr,
    local_sender_channel_6_connection_info_addr, local_sender_channel_7_connection_info_addr,
    local_sender_channel_8_connection_info_addr, local_sender_channel_9_connection_info_addr};

// The worker interface of every sender channel: its location info, connection semaphore and starting credits (0 for a
// worker channel, whose credits are counters; the ring size for a forwarding channel).
__attribute__((noinline)) void init_worker_interfaces(WorkerInterfaces& interfaces, const RuntimeArgs& args) {
    for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
        constexpr size_t CH = decltype(i)::value;
        auto* info = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(connection_info_addrs[CH]);
        info->edm_read_counter = 0;
        new (&interfaces.template get<CH>())
            tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY[CH]>(
                info,
                0,
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(args.connection_semaphores[CH]),
                sender_channel_ack_cmd_buf_ids[CH],
                CH == 0 ? 0 : SENDER_NUM_BUFFERS_ARRAY[CH],
                notify_worker_of_read_counter_update_src_address);
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
            upstream.addr[CH] = interface.cached_worker_semaphore_address;
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
    rt::init();  // round-trip timing (router_rt.hpp)

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

    // Phase: downstream connections. As main: a VC's connections are opened when its receiver channel is serviced, also
    // where no packet uses them (an intermesh router's VC0 connections with crossover), since the downstream routers wait
    // for them.
    if constexpr (is_receiver_role) {
        if constexpr (is_receiver_channel_serviced[0]) {
            open_downstreams<0>(args.vc0());
        }
        if constexpr (has_vc1 && is_receiver_channel_serviced[1]) {
            open_downstreams<1>(args.vc1());
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
    if constexpr (MY_ERISC_ID == 0) {
        router_state()->state = RouterState::INITIALIZING;
    }

    if constexpr (is_sender_role) {
        sender::run(interfaces, upstream);
    } else {
        // Receiver channel 0 forwards on VC0's connections, or on VC1's with crossover; channel 1 on VC1's.
        Downstreams<0> downstreams_vc0(links_vc0);
        Downstreams<vc1_links_vc> downstreams_vc1(links_vc1);
        receiver::run(of_vc<receiver::Channel<0>::downstream_vc>(downstreams_vc0, downstreams_vc1), downstreams_vc1);
    }
    WAYPOINT("LPDN");
    teardown();
    WAYPOINT("TEDN");
}
