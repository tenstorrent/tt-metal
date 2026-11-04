// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*

The fabric_erisc_router is a component that can be used to implement various fabric topologies - namely graphs of k-ary n-cubes.
One of these EDMs can be instantiated on each ethernet link. It is built from 2 or more "channels", though the definition
of channel here is a little loose since multiple sender channels can merge traffic into shared receiver channels.
This router implements packet based messages only - concepts like sockets are not supported.

## Router Structure

There are one or more sender channels and one or more receiver channel. "Sender" and "receiver" are relative to the Ethernet link,
not the chip. Sender sends over the link and receiver receives from the link.

Each sender channel are grouped by VC. Each VC holds one or more sender channels. Receiver channels are one per VC. Each sender
channel serves a different purpose:
- VC0: forwards traffic originated from within this DeviceMesh
  - Sender channel 0 : Accepts packets from a workers on the local chip
  - Sender channel 1-N: Accepts packets from an upstream routers (i.e. an upstream
    router receiver channel on the same chip but different core)

The receiver channel accepts packets from the Ethernet link and can do one (or both) of:
- Write the packet to local chip if it is the intended destination (unicast)
- Forward the packet to a downstream router if:
  - Unicast and not the target chip
  - Multicast and this chip is in the multicast target range

Sender channels will merge traffic into the remote router's receiver channel.

Below is an example diagram that shows how routers can be connected over an ethernet link. In this case,
a simple line topology is instantiated, and the kernels run on separate, but connected ethernet link cores.

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

Below shows the intended connectivity of routers across chips in a hypothetical 3-chip line fabric. For longer
lines, the pattern would be extended. For higher dimensional topologies, more channels are instantiated. For
more levels of routing (i.e. multi-mesh), more VCs are instantiated.

           CHIP 0                              CHIP 1                             CHIP 2
     ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
     │                 │                │                 │                │                 │
┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐      ┌────┴─────┐ ▲   ┌─────┴────┐
│   EDM    │ │   │   EDM    │      │   EDM    │ │   │   EDM    │      │   EDM    │ │   │   EDM    │
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

As mentioned, only one worker can push to a given EDM sender channel at a time. In order to send to an EDM
sender channel, the worker must establish a connection. The connection protocol is as follows and is started
by the worker (the EDM is a subordinate in this protocol).

*NOTE*: If multiple workers try to connect to the same EDM sender channel at the same time, the behavior is undefined.
*NOTE*: Additionally, if a worker pushes packets to a channel it isn't connected to, behaviour is undefined.
*NOTE*: Undefined == likely hang

## Packet Structure

Workers are responsible for populating packet headers before sending to the EDM. The packet header structure is defined
in `fabric_edm_packet_header.hpp`.

## Channel structure

Each EDM channel is built from one or more buffers. Each buffer is the same size and can hold at most one packet.
Neighbouring packets occupy nehighouring buffers - with the exception of the last buffer index. The next packet after a
write into the last buffer index will wrap around to the first buffer index. Even if packets do not occupy the full
buffer, subsequent packets will always be written into the next logical buffer. A gap will exist in memory but the EDM
will not send that padded data (unless it is more performant - which is possible in some special cases)

 Example channel with 8 buffers
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 buf 0   buf 1   buf 2   buf 3   buf 4   buf 5   buf 6   buf 7


Here we have an example of a channel with 4 buffers, filled with some number of packets. Each packet is a different
size. Packets 0, 2, and 3 are smaller than the full buffer size, while packet 1 is the full buffer size.

┌───────────────┬───────────────┬───────────────┬───────────────┐
│H|Payload| / / │H|Payload      │H|Pyld| / / / /│H|Payload  |/ /│
│ |       |/ / /│ |             │ |    |/ / / / │ |         | / │
└───────────────┴───────────────┴───────────────┴───────────────┘
  buf 0           buf 1           buf 2           buf 3


## Sending Packets
Sending a packet is done as follows:

1) Worker checks (and conditionally waits for) credits to send to the router's sender channel
  - credits are used to notify of space in the sender channel
2) Worker performs a noc write of its packet to the router's sender channel 0 at the buffer index

*NOTE*: !!!ALL PACKETS MUST CONTAIN DESTINATION NOC X/Y AS NOC 0 COORDINATES, REGARDLESS OF THE `noc_index` OF THE
SENDER!!!

## router <-> router Channel Flow Control
Flow control between routers counts whole buffer slots. The counts live in stream registers (packets written into a
channel) and in L1 counters that the far router updates over Ethernet (slots it has freed). See
`router/router_sender.hpp` and `router/router_receiver.hpp` for the implementation.

### Sender Channel Flow Control
* free-slots register (one per sender channel):
  * the producer (a worker, or the upstream router on this chip) takes one for every packet it writes to the channel
  * the router gives one back for every packet it sends; any value other than the slot count means a packet waits
* far receiver free slots:
  * all sender channels of a VC write into the same far receiver channel and share one count of its free slots
  * a send takes a slot; completion credits from the far receiver give slots back
* producer credits:
  * the router returns a slot to its producer (a count written to the worker, or to the upstream router's copy of
    the channel's free slots) once the far receiver completes the packet, or acks it on channels with first-level acks

### Receiver Channel Flow Control
* packets-received register:
  * the far sender increments it after each packet, on the same Ethernet queue as the data
* first-level acks (VC0 on rings and tori):
  * every packet that has arrived is acked before it is processed, so the far sender can credit its producer early
* completions:
  * a slot is complete once every NoC command that reads it (writes to this chip, forwards to downstream routers) has
    left the NIU
  * slots complete in aligned groups of up to 4; every NoC command that reads a slot carries its group's transaction
    ID, so one transaction ID status read proves a whole group free
* acks and completions are absolute counts per far sender channel, copied to the far router's L1 as one Ethernet
  frame per pass when they change

### Bubble Flow Control (rings and tori with deadlock avoidance)
* a channel that injects traffic into the ring (the worker channel; on a torus also a channel taking traffic that
  turns in from the other axis) sends only while the far receiver has T = max(2, R - 16) of its R slots free
* a forwarding channel sends with one free slot, so traffic already on the ring can always make progress
*/

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
