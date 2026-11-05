// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*

The fabric ERISC router builds fabric topologies from k-ary n-cubes (lines, rings, meshes, tori). One router runs on
each Ethernet link. It carries packets only; there are no sockets.

## Router Structure

A router has sender channels and receiver channels. "Sender" and "receiver" are relative to the Ethernet link, not the
chip: a sender channel sends over the link, a receiver channel receives from it.

Sender channels are grouped by VC, and each VC has one receiver channel. Within a VC:
- the worker channel (VC0's sender channel 0) takes packets from a worker on this chip;
- the forwarding channels (VC0's sender channels 1-N) take packets from upstream routers (the receiver channel of
  another router on this chip).

A receiver channel takes packets off the link and does one or both of:
- write the packet to this chip, when this chip is its destination;
- forward it to downstream routers: a unicast for another chip, or a multicast whose range goes on past this chip.

The sender channels of a VC all write into the remote router's receiver channel of that VC.

Below, two routers connected over one Ethernet link in a line topology, each on its own Ethernet core.

 ┌───────────────────────┐           ┌───────────────────────┐
 │    Sender Channel 0   │           │    Receiver Channel   │
 │   ┌────────────────┐  │           │   ┌────────────────┐  │
 │   │                ┼──┼───┬───────┼───►                │  │
 │   │                │  │   │       │   │                │  │
 │   └────────────────┘  │   │       │   └────────────────┘  │
 │    Sender Channel 1   │   │       │    Sender Channel 1   │
 │   ┌────────────────┐  │   │       │   ┌────────────────┐  │
 │   │                ┼──┼───┘       │   │                │  │
 │   │                │  │         ┌─┼───┼                │  │
 │   └────────────────┘  │         │ │   └────────────────┘  │
 │    Receiver Channel   │         │ │    Sender Channel 0   │
 │   ┌────────────────┐  │         │ │   ┌────────────────┐  │
 │   │                │  │         │ │   │                │  │
 │   │                ◄──┼─────────┴─┼───┼                │  │
 │   └────────────────┘  │           │   └────────────────┘  │
 │                       │           │                       │
 │                       │           │                       │
 └───────────────────────┘           └───────────────────────┘


## Building a "Fabric"

The intended connectivity of routers across chips in a 3-chip line. Longer lines repeat the pattern; higher-dimensional
topologies add channels, and more levels of routing (multi-mesh) add VCs.

           CHIP 0                              CHIP 1                             CHIP 2
     ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
     │                 │                │                 │                │                 │
┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐
│  Router  │ │   │  Router  │      │  Router  │ │   │  Router  │      │  Router  │ │   │  Router  │
│ ┌──────┐ │ │   │ ┌──────┐ │      │ ┌──────┐ │ │   │ ┌──────┐ │      │ ┌──────┐ │ │   │ ┌──────┐ │
│ │ Rx   ┼─┼─┴───┼─► S1   ┼─┼─┬────┼─► Rx   ┼─┼─┴───┼─► S1   ┼─┼┬─────┼─► Rx   ┼─┼─┘   | | S1   │ │
│ └──────┘ │     │ └──────┘ │ │    │ └──────┘ │     │ └──────┘ ││     │ └──────┘ │     │ └──────┘ │
│ ┌──────┐ │     │ ┌──────┐ │ │    │ ┌──────┐ │     │ ┌──────┐ ││     │ ┌──────┐ │     │ ┌──────┐ │
│ │ S0   ◄─┼──┬──┼─► S0   ┼─┼─┘   ┌┼─┼ S0   ◄─┼──┬──┼─► S0   ┼─┼┘    ┌┼─┼ S0   ◄─┼──┬──┼─► S0   │ │
│ └──────┘ │  │  │ └──────┘ │     ││ └──────┘ │  │  │ └──────┘ │     ││ └──────┘ │  │  │ └──────┘ │
│ ┌──────┐ │  │  │ ┌──────┐ │     ││ ┌──────┐ │  │  │ ┌──────┐ │     ││ ┌──────┐ │  │  │ ┌──────┐ │
│ │ S1   | |  │ ┌┼─┼ Rx   ◄─┼─────┴┼─┼ S1   ◄─┼─┐│ ┌┼─┼ Rx   ◄─┼─────┴┼─┼ S1   ◄─┼─┐│ ┌┼─┼ Rx   │ │
│ └──────┘ │  | |│ └──────┘ │      │ └──────┘ │ └┼─┤│ └──────┘ │      │ └──────┘ │ └┼─┤│ └──────┘ │
└────┬─────┘  │ │└─────┬────┘      └────┬─────┘  │ │└─────┬────┘      └────┬─────┘  │ │└─────┬────┘
     │          ▼      │                │          ▼      │                │          ▼      │
     └─────────────────┘                └─────────────────┘                └─────────────────┘


## Connecting Workers to Channels

Only one worker can push to a worker channel at a time, and it must connect first. The worker starts the connection;
the router is the subordinate in the protocol.

*NOTE*: If several workers connect to the same worker channel at once, or a worker pushes packets to a channel it is
not connected to, the behaviour is undefined (likely a hang).

## Packet Structure

Workers fill in the packet header before sending to the router. The header is defined in `fabric_edm_packet_header.hpp`.

## Channel structure

A channel is a ring of slots of one size, each holding at most one packet. Consecutive packets take consecutive slots,
wrapping from the last slot to the first, whatever their size: a packet shorter than its slot leaves a gap, which the
router does not send (unless sending it is faster, in some special cases).

 Example channel with 8 slots
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 slot 0  slot 1  slot 2  slot 3  slot 4  slot 5  slot 6  slot 7


A channel of 4 slots holding packets of different sizes: packets 0, 2 and 3 are shorter than a slot, packet 1 fills
its slot.

┌───────────────┬───────────────┬───────────────┬───────────────┐
│H|Payload| / / │H|Payload      │H|Pyld| / / / /│H|Payload  |/ /│
│ |       |/ / /│ |             │ |    |/ / / / │ |         | / │
└───────────────┴───────────────┴───────────────┴───────────────┘
  slot 0          slot 1          slot 2          slot 3


## Sending Packets

1) The worker checks (and waits for) credits for its worker channel: credits count the channel's free slots.
2) The worker writes its packet over the NoC into the worker channel's next slot.

*NOTE*: !!!ALL PACKETS MUST CONTAIN DESTINATION NOC X/Y AS NOC 0 COORDINATES, REGARDLESS OF THE `noc_index` OF THE
SENDER!!!

## Terms

Comments use the code's names:
* remote router, remote sender, remote receiver: across the Ethernet link (RemoteReceiverChannel)
* upstream, downstream router: on this chip; a receiver channel forwards to its downstream routers, and a forwarding
  channel's upstream router writes into it; a sender channel's producer is its worker or its upstream router
* slot: one packet's place in a channel's ring; credit: slots given back by count, to a producer (credit_slots) or to
  the remote sender (completion and first-level-ack credits, one credit frame per pass)
* completion: a slot whose NoC commands have all left the NIU, credited per completion group once the group's
  transaction ID proves it
* crossover: a receiver channel forwarding on VC1's downstream connections (forwards_on_vc); forward buffer pair: the
  data and credit command buffers a VC's forwards use
* serviced: kept by channel trimming; a router runs a VC (runs_vc) when configured for it and its receiver channel is
  serviced
* pass: one visit of each sender channel or one step of each receiver channel; round: the passes between end_round calls

## Router <-> router flow control
Flow control between routers counts whole slots. The counts live in stream registers (packets written into a channel)
and in L1 counters that the remote router updates over Ethernet (slots it has freed). See `router/router_sender.hpp`
and `router/router_receiver.hpp`.

### Sender Channel Flow Control
* free-slots register (one per sender channel):
  * the producer takes one for every packet it writes to the channel
  * the router gives one back for every packet it sends; any value other than the slot count means a packet waits
* remote receiver free slots:
  * all sender channels of a VC write into the same remote receiver channel and share one count of its free slots
  * a send takes a slot; completion credits from the remote receiver give slots back
* producer credits:
  * the router credits a slot to its producer (a count written to the worker, or +1 on the upstream router's copy of the
    channel's free slots) once the remote receiver completes the packet, or acks it on first-level-ack VCs

### Receiver Channel Flow Control
* packets-received register:
  * the remote sender increments it after each packet, on the same TX queue as the data
* first-level acks (VC0 on rings and tori):
  * every packet that has arrived is acked before it is processed, so the remote sender can credit its producer early
* completions:
  * a slot is complete once every NoC command that reads it (writes to this chip, forwards to downstream routers) has
    left the NIU
  * slots complete in aligned completion groups of up to 4; every NoC command that reads a slot carries its group's
    transaction ID, so one transaction ID status read proves a whole group free
* acks and completions are absolute counts per remote sender channel, copied to the remote router's L1 as one credit
  frame per pass when they change

### Bubble Flow Control (rings and tori with deadlock avoidance)
* an injection channel (the worker channel; on a torus also a channel taking traffic that turns in from the other axis)
  sends only while the remote receiver has T = max(2, R - 16) of its R slots free
* a forwarding channel sends with one free slot, so traffic already on the ring can always make progress
*/

// On a two-ERISC Blackhole router, ERISC 0 runs the sender program (router/router_sender.hpp) and ERISC 1 the receiver
// program (router/router_receiver.hpp); on a single-ERISC router one ERISC runs both (router/router_single.hpp).
// kernel_main is the init and teardown skeleton every ERISC runs (helpers: router/router_init.hpp): in each phase each
// role does its own part, and the ERISCs meet at the phase's end. The supported configurations are listed in
// router/router_common.hpp; others fail to compile there.

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
        // Again: the router state reads INITIALIZING until the first round starts running.
        router_state()->state = RouterState::INITIALIZING;
    }

    run_loop(downstreams);
    WAYPOINT("LPDN");
    teardown();
    WAYPOINT("TEDN");
}
