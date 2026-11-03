// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric ERISC router: moves fabric packets across one Ethernet link.
//
// A router has sender channels, which take packets from a worker on this chip (a VC's worker channel) or from the router
// facing another direction (forwarding channels) and send them over the link, and receiver channels, which take packets
// off the link and write them to this chip, forward them to the next router, or both. Credits flow back the same way.
//
// On a two-ERISC Blackhole router, ERISC 0 runs the sender program (router/router_sender.hpp) and ERISC 1 the receiver
// program (router/router_receiver.hpp); on a single-ERISC router one ERISC runs both (router/router_single.hpp).
// kernel_main is the init and teardown skeleton every ERISC runs (its helpers: router/router_init.hpp): at each phase
// each role does its own part, and the ERISCs meet at the phase's end.
//
// The configurations this router supports are listed at the bring-up gate below; others fail to compile there.

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
// into a receiver channel of its own that only delivers (neighbour exchange; VC2 line mode, which forwards, is not
// brought up here), counter credits.
constexpr bool vc2_config_supported = is_2d && !has_vc1 && ACTUAL_VC2_SENDER_CHANNELS == 1 &&
                                      is_vc_forwarding_disabled[2] &&
                                      receiver_channel_uses_counter_credits(VC2_RECEIVER_CHANNEL);

// The speedy path as the host selects it (ENABLE_SPEEDY_VC0; main's speedy-path static_asserts): no bubble flow control
// or first-level acks on VC0, VC0's receiver forwarding off. Its credit amortization args do not apply here (credits
// go back per completion group, as everywhere).
constexpr bool speedy_config_supported =
    !enable_deadlock_avoidance && !ENABLE_FIRST_LEVEL_ACK_VC0 && disable_rx_ch0_forwarding && !udm_mode;

// Bring-up gate: the configurations this router supports so far.
//   - Blackhole; two ERISCs (one role each, a TX queue each) or one ERISC (both roles);
//   - 1D and 2D: lines, meshes, rings and tori (bubble flow control; first-level acks on VC0); strict atomics;
//   - VC0; VC1 on 2D multi-mesh fabrics (intermesh landing, crossover and exits);
//   - VC2 on 2D single-mesh routers: neighbour exchange (a worker channel into a receiver channel that only delivers);
//   - Tensix mux mode on one ERISC, VC0 (the fabric's muxes feed the worker channel and take the forwards);
//   - UDM on 2D meshes, one ERISC (packets for this chip go to the fabric's relay);
//   - the speedy path (VC0 worker-only routers whose receiver channel only delivers: it skips decode and admission);
//   - the host's telemetry options: the perf recorder at low resolution and the dynamic stats;
//   - channel trimming: the capture, and a replay's routers with channels compiled out (an ERISC left with nothing to
//     service runs rounds only, router_round.hpp's run_idle).
constexpr bool brought_up =
    is_blackhole_arch &&
    (NUM_ACTIVE_ERISCS == 2 ? multi_txq_enabled && !(is_sender_role && is_receiver_role) : NUM_ACTIVE_ERISCS == 1) &&
    (!has_vc2 || vc2_config_supported) && NUM_RECEIVER_CHANNELS == 1 + (has_vc1 ? 1 : 0) + (has_vc2 ? 1 : 0) &&
    (!has_vc1 || is_2d) &&
    !ENABLE_FIRST_LEVEL_ACK_VC1 && !ENABLE_FIRST_LEVEL_ACK_VC2 && !is_vc_crossover_to_vc1[2] &&
    (!udm_mode || udm_config_supported) &&
    (!super_speedy_mode || speedy_config_supported) &&
    !ENABLE_RISC_CPU_DATA_CACHE && !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA &&
    (perf_telemetry_mode == PerfTelemetryRecorderType::NONE ||
     perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH) &&
    (!tensix_mux_mode || NUM_ACTIVE_ERISCS == 1) &&
    (num_ds_or_local_tensix_connections == 0 || tensix_mux_mode || udm_mode);
static_assert(brought_up, "fabric router v4: this configuration is not brought up yet (see the bring-up gate)");

}  // namespace fabric_router

void kernel_main() {
    using namespace fabric_router;
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.reset();
    }
    postcode(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
    router_state()->state = RouterState::INITIALIZING;
    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    init_telemetry_block();
    telemetry::init();

    // Phase: stream registers.
    init_txqs();
    postcode(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);
    receiver::init_channel_streams();
    sender::init_channel_streams();
    init_context_switch_state();
    sync_erisc(Phase::STREAMS_SET);
    postcode(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);
    if constexpr (code_profiling_enabled_timers_bitfield != 0) {
        clear_code_profiling_buffer(code_profiling_buffer_base_addr);
    }

    // Phase: local setup.
    RuntimeArgs args;
    read_runtime_args(args);
    // The receiver's downstream connections (router_downstream.hpp): set up here, then used by its loop.
    DownstreamConnections downstreams{};
    if constexpr (is_sender_role) {
        sender::reset_connections(args.connection_semaphores, args.producer_cursors);
    }
    postcode(tt::tt_fabric::EDMStatus::STARTED);
    *edm_status() = tt::tt_fabric::EDMStatus::STARTED;
    postcode(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
    if constexpr (is_receiver_role) {
        receiver::set_up_channels(downstreams, args);
    }
    postcode(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);
    if constexpr (is_sender_role) {
        sender::set_up_channels(args.connection_semaphores);
    }
    sync_erisc(Phase::SETUP_DONE);

    // Phase: handshakes.
    run_handshakes();
    sync_erisc(Phase::HANDSHAKE_DONE);
    postcode(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);

    // Phase: downstream connections.
    if constexpr (is_receiver_role) {
        receiver::open_connections(args);
    }
    sync_erisc(Phase::DOWNSTREAMS_OPEN);
    postcode(tt::tt_fabric::EDMStatus::VCS_OPENED);
    if constexpr (is_receiver_role) {
        receiver::preset_forward_commands(args);
    }
    sync_erisc(Phase::CMD_BUFS_SET);
    postcode(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);

    // Phase: upstream connections.
    WAYPOINT("FSCW");
    if constexpr (is_sender_role) {
        sender::wait_for_upstream_routers();
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
        single::run(downstreams);
    } else if constexpr (is_sender_role) {
        sender::run();
    } else if constexpr (is_receiver_role) {
        receiver::run(downstreams);
    } else {
        run_idle();
    }
    WAYPOINT("LPDN");
    teardown();
    WAYPOINT("TEDN");
}
