// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric ERISC router: moves fabric packets across one Ethernet link.
//
// A router has sender channels, which take packets from a worker on this chip (a VC's worker channel) or from the
// router facing another direction (forwarding channels) and send them over the link, and receiver channels, which take
// packets off the link and write them to this chip, forward them to the next router, or both. Credits flow back the
// same way.
//
// On a two-ERISC Blackhole router, ERISC 0 runs the sender program (router/router_sender.hpp) and ERISC 1 the receiver
// program (router/router_receiver.hpp); on a single-ERISC router one ERISC runs both (router/router_single.hpp).
// kernel_main is the init and teardown skeleton every ERISC runs (its helpers: router/router_init.hpp): in each phase
// each role does its own part, and the ERISCs meet at the phase's end. The configurations the router supports are
// listed in router/router_common.hpp; others fail to compile there.

// Selects, in the shared ct-args header, this router's transaction-ID count (rx_batch_completion: 16 IDs on a lone
// receiver channel without bubble flow control).
#define FABRIC_ERISC_ROUTER_V4 1

#include "api/dataflow/dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_init.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp"

namespace fabric_router {

// The ERISC's loop. One ERISC runs the single-ERISC loop whatever channel trimming left it (a role without channels
// makes empty passes); on two ERISCs each runs its role's loop, and an ERISC trimming left nothing to service runs
// rounds only. (A template: an ERISC without the receiver's downstream connections names them only here.)
template <typename Downstreams>
FORCE_INLINE void run_loop(Downstreams& downstreams) {
    if constexpr (NUM_ACTIVE_ERISCS == 1) {
        single::run(downstreams);
    } else if constexpr (is_sender_role) {
        sender::run();
    } else if constexpr (is_receiver_role) {
        receiver::run(downstreams);
    } else {
        run_idle();
    }
}

}  // namespace fabric_router

void kernel_main() {
    using namespace fabric_router;
    init_router();

    // Phase: stream registers, before the Ethernet handshake and before any producer connects.
    init_txqs();
    receiver::init_channel_streams();
    sender::init_channel_streams();
    init_context_switch_state();
    finish_phase<Phase::STREAMS_SET>();

    // Phase: local setup.
    RuntimeArgs args;
    read_runtime_args(args);
    if constexpr (is_sender_role) {
        sender::reset_connections(args.connection_semaphores, args.producer_cursors);
    }
    *edm_status() = tt::tt_fabric::EDMStatus::STARTED;
    auto downstreams = receiver::set_up_channels(args);  // the receiver's downstream connections, for its loop
    if constexpr (is_sender_role) {
        sender::set_up_channels(args.connection_semaphores);
    }
    finish_phase<Phase::SETUP_DONE>();

    // Phase: handshakes.
    run_handshakes();
    finish_phase<Phase::HANDSHAKE_DONE>();

    // Phase: downstream connections, then the forwards' command buffers.
    if constexpr (is_receiver_role) {
        receiver::open_connections(args);
    }
    finish_phase<Phase::DOWNSTREAMS_OPEN>();
    if constexpr (is_receiver_role) {
        receiver::preset_forward_commands(args);
    }
    finish_phase<Phase::CMD_BUFS_SET>();

    // Phase: upstream connections.
    WAYPOINT("FSCW");
    if constexpr (is_sender_role) {
        sender::wait_for_upstream_routers();
    }
    WAYPOINT("FSCD");
    finish_phase<Phase::CONNECTED>();
    if constexpr (is_teardown_master) {
        // Again, as main does at this point: the router state reads INITIALIZING until the first round starts running.
        router_state()->state = RouterState::INITIALIZING;
    }

    run_loop(downstreams);
    WAYPOINT("LPDN");
    teardown();
    WAYPOINT("TEDN");
}
