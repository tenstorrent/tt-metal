// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_link_transport.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_tmp_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_packet_recorder.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_erisc_microbench.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_channel_traits.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"

#include "noc_overlay_parameters.h"
#include "api/alignment.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
#include "hostdevcommon/fabric_telemetry_msgs.h"
#ifdef FABRIC_2D
// The router re-encodes an intermesh packet's route when it lands, via
// fabric_set_2d_intermesh_landing_route(). Include its declaration directly and only for the 2D call site.
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(PROFILE_STREAMING) && defined(ARCH_BLACKHOLE)
#include "tt_metal/impl/streaming_profiler/kernels/link_sync.hpp"
#endif

using namespace tt::tt_fabric;

// Type alias for the 1D low-latency hop fields.
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

/*

The fabric Erisc Data Mover (EDM) is a component that can be used to build *very* simple linear topology fabrics.
One of these EDMs can be instantiated on each ethernet link. It is built from 3 "channels" (though the definition
of channel here is a little loose since two of the 3 will merge traffic, so this setup could be interpreted as a
two channel setup.). This EDM implements packet based packets only - concepts like sockets are not supported.

## EDM Structure

There are two sender channels and one receiver channel. "Sender" and "receiver" are relative to the Ethernet link,
not the chip. Sender sends over the link and receiver receives from the link.

Each sender channel serves a different purpose:
- Sender channel 0 : Accepts packets from a workers on the local chip
- Sender channel 1: accepts packets from an upstream EDM (i.e. an upstream
  EDM receiver channel on the same chip but different core)

The receiver channel accepts packets from the Ethernet link and can do one (or both) of:
- Write the packet to local chip if it is the intended destination (unicast or mcast)
- Forward the packet to the next chip in the line if:
  - Unicast and not the target chip
  - Multicast and this chip is in the multicast target range

Sender channels will merge traffic into the remote EDM's receiver channel.

Below is a diagram that shows how EDMs can be connected over an ethernet link. In this case, the two
EDM kernels are run on separate, but connected ethernet link cores.

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

At present, only linear topologies are supported, and one per ethernet link along that given line.
Below shows the intended connectivity of EDMs across chips in a hypothetical 3-chip fabric. For longer
lines, the pattern would be extended.

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

The `EdmToEdmSender` from `edm_fabric_worker_adapters.hpp`
provides an implementation of the connection protocol. `EdmToEdmSender` also acts as a wrapper around that
protocol so workers can simply call `open()` to execute the connection protocol without having to manually reimplement
for each kernel.

### Protocol
Worker:
- Read from EDM sender channel buffer_index address
  - Required so that the worker knows where to write its first packet (since the channel may already contain packets
from a previous connection)
- Write worker core X/Y (NOC 0 based)
- Write worker flow control semaphore L1 address

EDM Sender Channel:
- Check local connection valid semaphore for new established connection
  - When the connection semaphore indicates an active connection, the channel assumes all other relevant fields were
    correctly populated by the worker:
    - Worker core_x (on NOC 0)
    - Worker core_y (on NOC 0)
    - Worker flow control semaphore L1 address


## Tearing Down Connections

Every worker is required to explicitly teardown its connection with the EDM before terminating. To do this, the worker
must simply write a `0` to the EDM sender channel's connection semaphore address. As long as the worker has sent all
of its packets to the EDM before this, then the EDM will guarantee to forward the messages correctly.

At this point, it is safe for another kernel to establish a connection.

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

1) Worker waits for flow control semaphore increment from EDM sender channel
  - Indicates there is space at the next buffer index for a packet
2) Worker performs a noc write of its packet to the EDM sender channel at the buffer index

*NOTE*: !!!ALL PACKETS MUST CONTAIN DESTINATION NOC X/Y AS NOC 0 COORDINATES, REGARDLESS OF THE `noc_index` OF THE
SENDER!!!


## EDM <-> EDM Channel Flow Control
The flow control protocol between EDM channels is built on a rd/wr ptr based protocol where pointers are
to buffer slots within the channel (as opposed so something else like byte or word offset). Ptrs are
free to advance independently from each other as long as there is no overflow or underflow.

The flow control is implemented through the use of several stream registers: one per conceptual pointer being tracked.
In total there are 5 such counters:
1) to receiver channel packets sent
  - Incremented by sender (via eth_reg_write) by the number of buffer slots written. In practice, this means it is
    incremented once per packet
2) to sender 0 packets acked
  - Incremented by receiver for every new packet from channel 0 that it sees
3) to sender 1 packets acked
  - Incremented by receiver for every new packet from channel 1 that it sees
4) to sender 0 packets completed
  - Incremented by receiver for every packet from channel 0 that it completes processing for
5) to sender 1 packets completed
  - Incremented by receiver for every packet from channel 1 that it completes processing for

See calls to `increment_local_update_ptr_val`, `remote_update_ptr_val`, `init_ptr_val` for more on implementation.

### Sender Channel Flow Control
Both sender channels share the same flow control view into the receiver channel. This is because both channels
write to the same receiver channel.
* wrptr:
  * points to next buffer slot to write to into the remote (over Ethernet) receiver channel.
  * leads other pointers
  * writer updates for every new packet
  * `has_data_to_send(): local_wrptr != remote_sender_wrptr`
* ackptr
  * trails `wrptr`
  * advances as the channel receives acknowledgements from the receiver
    * as this advances, the sender channel can notify the upstream worker of additional space in sender channel buffer
* completion_ptr:
  * trails `local_wrptr`
  * "rdptr" from remote sender's perspective
  * advances as packets completed by receiver
    * as this advances, the sender channel can write additional packets to the receiver at this slot

### Receiver Channel Flow Control
* ackptr/rdptr:
  * leads all pointers
  * indicates the next buffer slot we expect data to arrive (from remote sender) at
    * advances as packets are received (and acked)
  * make sure not to overlap completion pointer
* wr_sent_ptr:
  * trails `ackptr`
  * indicates the buffer slot currently being processed, written out
    * advances after all forwding writes (to noc or downstream EDM) are initiated
* wr_flush_ptr:
  * trails `wr_sent_ptr`
  * advances as writes are flushed
* completion_ptr:
  * trails `wr_flush_ptr`
  * indicates the next receiver buffer slot in the receiver channel to send completion acks for
*/

////////////////////////////////////////////////
// Data structures, types, enums, and constants
////////////////////////////////////////////////

// read and write stream scratch register store values as uint32_t
enum class CoordinatedEriscContextSwitchState : uint32_t {
    // Initially set by the master (erisc0) in entrance of kernel_main() and is the default state. erisc1 polls for this
    // state at the end of the handshake
    NORMAL_EXECUTION = 0,
    // Set by master to signal intent on beginning handshake. Checked by erisc1 before it begins handshake
    RETRAIN_INTENT = 1,
    // Set by erisc1 to indicate it's cooperation. Polled by master before it runs the retrain to ensure that it has
    // erisc1's cooperation
    INTENT_ACK = 2,
    // Set by erisc0 to indicate completion of retrain. Polled by erisc1
    RETRAIN_COMPLETE = 3,
    // Set by erisc1 to indicate it has seen the completion of retrain. Polled by erisc0 before it sets register back to
    // NORMAL_EXECUTION
    COMPLETE_ACK = 4,
};

// In case underlying type for the enum class is changed later
using CoordinatedEriscCtxType = std::underlying_type_t<CoordinatedEriscContextSwitchState> ;

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using SenderEthChannel = StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>;

static constexpr bool PERF_TELEMETRY_DISABLED = perf_telemetry_mode == PerfTelemetryRecorderType::NONE;
static constexpr bool PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH =
    perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH;
using PerfTelemetryRecorder = std::conditional_t<
    PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH,
    LowResolutionBandwidthTelemetry,
    std::conditional_t<PERF_TELEMETRY_DISABLED, bool, std::nullptr_t>>;

template <uint8_t SENDER_NUM_BUFFERS>
using RouterToRouterSender = tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>;

static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_free_slots_stream_ids = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id,
    sender_channel_5_free_slots_stream_id,
    sender_channel_6_free_slots_stream_id,
    sender_channel_7_free_slots_stream_id,
    sender_channel_8_free_slots_stream_id,
    sender_channel_9_free_slots_stream_id};
// Each VC's sender free-slots registers form one contiguous group. Boundaries come from the VC-start
// CT args rather than pinned literals, so a future repack still holds.
constexpr bool is_contiguous_stream_span(const uint32_t* ids, size_t count) {
    for (size_t i = 1; i < count; ++i) {
        if (ids[i] != ids[i - 1] + 1) {
            return false;
        }
    }
    return true;
}
static_assert(is_contiguous_stream_span(sender_channel_free_slots_stream_ids.data(), VC1_FABRIC_POSITION_START));
static_assert(is_contiguous_stream_span(
    sender_channel_free_slots_stream_ids.data() + VC1_FABRIC_POSITION_START,
    VC2_FABRIC_POSITION_START - VC1_FABRIC_POSITION_START));
// VC2's sender rides the pinned register 30 when present, the out-of-range sentinel otherwise.
static_assert(
    sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START] ==
    (ACTUAL_VC2_SENDER_CHANNELS > 0 ? tt::tt_fabric::connection_interface::vc2_sender_free_slots_stream_id
                                    : tt::tt_fabric::k_unused_stream_id));

// The sentinel marks an inactive consumer, e.g. a VC whose credits travel through L1 counters and
// reads no register at all. It is out of register range, so an inactive entry can clobber nothing.
template <uint32_t STREAM_ID>
FORCE_INLINE void init_stream_reg_if_used(uint32_t value) {
    if constexpr (STREAM_ID != tt::tt_fabric::k_unused_stream_id) {
        init_ptr_val<STREAM_ID>(value);
    }
}

// For 2D fabric: maps compact index to downstream direction for each my_direction
// For 1D fabric: only 1 downstream direction per router (EAST forwards to WEST in 1D linear topology)
constexpr uint32_t MAX_DOWNSTREAM_EDM_COUNT = 4;
#if defined(FABRIC_2D)
constexpr uint32_t edm_index_to_edm_direction[eth_chan_directions::COUNT][MAX_DOWNSTREAM_EDM_COUNT] = {
    {eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // EAST router
    {eth_chan_directions::EAST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // WEST router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // NORTH router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::Z},  // SOUTH router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH},  // Z router
};

// sender_channel_free_slots_stream_ids[] mapping:
//   [0] → Local worker (always uses sender channel 0 on the outgoing router).
//   [1–3] → Sender channels 1–3 on the outgoing router, corresponding to
//           inbound traffic from neighboring routers.
//
// The mapping is relative to the outgoing router's direction:
//
//   • East-outbound router:
//         sender channel 1 (idx 0) ← West inbound
//         sender channel 2 (idx 1) ← North inbound
//         sender channel 3 (idx 2) ← South inbound
//
//   • West-outbound router:
//         sender channel 1 (idx 0) ← East inbound
//         sender channel 2 (idx 1) ← North inbound
//         sender channel 3 (idx 2) ← South inbound
//
//   • North-outbound router:
//         sender channel 1 (idx 0) ← East inbound
//         sender channel 2 (idx 1) ← West inbound
//         sender channel 3 (idx 2) ← South inbound
//
//   • South-outbound router:
//         sender channel 1 (idx 0) ← East inbound
//         sender channel 2 (idx 1) ← West inbound
//         sender channel 3 (idx 2) ← North inbound
constexpr uint32_t get_vc0_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        // downstream sender channel = my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[my_direction];
    } else {
        // downstream sender channel = my_direction + 1
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[(1 + my_direction)];
    }
}

// VC1 downstream sender channel mapping (for inter-mesh routing)
// Same direction rules as VC0, offset by VC1's fabric base to skip VC0's channels. The base is a CT
// arg because VC0's width varies: 4 on legacy fabrics, 5 on one with a boundary router.
constexpr uint32_t get_vc1_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        // downstream sender channel = (VC1 fabric base - 1) + my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[VC1_FABRIC_POSITION_START - 1 + my_direction];
    } else {
        // downstream sender channel = (VC1 fabric base) + my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[VC1_FABRIC_POSITION_START + my_direction];
    }
}
#endif

FORCE_INLINE constexpr eth_chan_directions map_compact_index_to_direction(size_t compact_index) {
#if defined(FABRIC_2D)
    return static_cast<eth_chan_directions>(edm_index_to_edm_direction[my_direction][compact_index]);
#else
    return static_cast<eth_chan_directions>(compact_index);
#endif
}

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_0_free_slots_stream_ids = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    vc_0_free_slots_from_downstream_edge_4_stream_id};

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_1_free_slots_stream_ids = {
    vc_1_free_slots_from_downstream_edge_1_stream_id,
    vc_1_free_slots_from_downstream_edge_2_stream_id,
    vc_1_free_slots_from_downstream_edge_3_stream_id,
    vc_1_free_slots_from_downstream_edge_4_stream_id};

// Compact-keyed local copy of downstream free-slots. Both overloads read the same array so init
// (runtime compact_index) and the indexed space check (NTTP compact / LIVE slot) cannot diverge.
template <size_t Compact>
constexpr uint32_t vc0_local_free_slots_stream_id() {
    return vc_0_free_slots_stream_ids[Compact];
}
constexpr uint32_t vc0_local_free_slots_stream_id(size_t compact) { return vc_0_free_slots_stream_ids[compact]; }

template <size_t Compact>
constexpr uint32_t vc1_local_free_slots_stream_id() {
    return vc_1_free_slots_stream_ids[Compact];
}
constexpr uint32_t vc1_local_free_slots_stream_id(size_t compact) { return vc_1_free_slots_stream_ids[compact]; }

enum PacketLocalForwardType : uint8_t {
    PACKET_FORWARD_INVALID = 0x0,
    PACKET_FORWARD_LOCAL_ONLY = 0x1,
    PACKET_FORWARD_REMOTE_ONLY = 0x2,
    PACKET_FORWARD_LOCAL_AND_REMOTE = 0x3
};

// tracks if the main loop made any progress. If many loop iterations were completed without
// did_something=true (i.e. no progress was made), then we allow for context switch in case
// the link is down
bool did_something;

// Experimental per-ERISC phase accounting (TT_FABRIC_PHASE_PROFILE); compiles away when disabled.
FabricPhaseProfiler<
    phase_profile_mode,
    phase_profile_buffer_addr,
    MY_ERISC_ID,
    phase_profile_mask,
    phase_profile_iter_timing,
    phase_profile_txq_mask>
    phase_profiler;

template <size_t N>
constexpr uint32_t count_serviced(const std::array<bool, N>& serviced) {
    uint32_t count = 0;
    for (bool s : serviced) {
        count += s ? 1 : 0;
    }
    return count;
}

// State for the EXP_TX_AMORTIZE_WORKER_CREDITS / EXP_TX_LAZY_CONN_CHECK experiments, per sender channel.
std::array<int32_t, MAX_NUM_SENDER_CHANNELS> exp_pending_upstream_credits;
std::array<uint8_t, MAX_NUM_SENDER_CHANNELS> exp_conn_poll_counter;

/////////////////////////////////////////////
//   SENDER SIDE HELPERS
/////////////////////////////////////////////

FORCE_INLINE void record_sender_channel_usage(size_t sender_channel_index, size_t packet_size_bytes) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(sender_channel_index);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(
            sender_channel_index, static_cast<uint16_t>(packet_size_bytes));
    }
}

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    bool SKIP_TAIL = false,
    uint32_t REMOTE_RECEIVER_BASE_ADDR = 0,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT>
FORCE_INLINE void send_next_data(
    SenderChannelT& sender_buffer_channel,
    WorkerInterfaceT& sender_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    uint32_t exp_payload_size_bytes = 0) {
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
    size_t payload_size_bytes;
    if constexpr (fabric_exp_tx_send_early_size_read) {
        payload_size_bytes = exp_payload_size_bytes;  // read by the caller before its TX-queue check
    } else {
        payload_size_bytes = pkt_header->get_payload_size_including_header();
    }

    record_sender_channel_usage(sender_channel_index, payload_size_bytes);

    auto const dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;

    if constexpr (!skip_src_ch_id_update) {
        pkt_header->src_ch_id = sender_channel_index;
    }

    if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        const uint32_t prespin_start = phase_profiler.open<PHASE_TX_TXQ_PRESPIN>();
        while (SenderLinkTransport::is_busy()) {
        };
        phase_profiler.close<PHASE_TX_TXQ_PRESPIN>(prespin_start);
    }
    SenderLinkTransport::send_packet(src_addr, dest_addr, payload_size_bytes);

    // Note: We can only advance to the next buffer index if we have fully completed the send (both the payload and sync
    // messages)
    if constexpr (!fabric_exp_tx_send_no_write_counter) {
        sender_worker_interface.template update_write_counter_for_send<SKIP_CONNECTION_LIVENESS_CHECK>();
    }

    // Advance receiver buffer pointers
    if constexpr (fabric_exp_tx_lean_vc0_loop && REMOTE_RECEIVER_BASE_ADDR != 0) {
        outbound_to_receiver_channel_pointers
            .template advance_remote_receiver_buffer_pointer_ct<REMOTE_RECEIVER_BASE_ADDR, channel_buffer_size>();
    } else {
        outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
    }
    // EXP_TX_SEND_EARLY_TXQ_READ: issue the busy read here and test it after the bookkeeping, as the speedy path does.
    // eth_txq_is_busy still makes the BH-55 dummy register read before reading STATUS.
    bool txq_busy = false;
    if constexpr (fabric_exp_tx_send_early_txq_read && !SKIP_TAIL) {
        txq_busy = SenderLinkTransport::is_busy();
    }
    if constexpr (fabric_exp_tx_send_stride_slot) {
        if constexpr (fabric_exp_tx_lean_vc0_loop) {
            constexpr size_t sender_entry_idx = SENDER_TO_ENTRY_IDX[sender_channel_index];
            constexpr size_t sender_base_addr = channel_allocs::template Entry<sender_entry_idx>::base_address;
            sender_buffer_channel
                .template advance_to_next_cached_buffer_slot_addr_by_stride_ct<sender_base_addr, channel_buffer_size>();
        } else {
            sender_buffer_channel.advance_to_next_cached_buffer_slot_addr_by_stride();
        }
    } else {
        sender_buffer_channel.advance_to_next_cached_buffer_slot_addr();
    }
    record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);
    if constexpr (SKIP_TAIL) {
        return;
    }
    remote_receiver_num_free_slots--;
    // update the remote reg

    const uint32_t spin_start = phase_profiler.open<PHASE_TX_TXQ_SPIN>();
    if constexpr (fabric_exp_tx_send_early_txq_read) {
        while (txq_busy) {
            txq_busy = SenderLinkTransport::is_busy();
        }
    } else {
        while (SenderLinkTransport::is_busy()) {
        };
    }
    phase_profiler.close<PHASE_TX_TXQ_SPIN>(spin_start);
    SenderLinkTransport::remote_update_ptr_val<to_receiver_pkts_sent_id>(1U);
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////

// True when local delivery of the packet is pure data movement, with no atomic or semaphore side effect that
// other cores (e.g. test sync) depend on. Only such packets may be dropped by EXP_RX_SKIP_LOCAL_DELIVERY.
FORCE_INLINE bool exp_local_delivery_is_data_only(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start, tt::tt_fabric::NocSendType noc_send_type) {
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE) {
        return true;
    }
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        const auto& scatter = packet_start->command_fields.unicast_scatter_write;
        const uint8_t last_encoding = (scatter.chunk_encoding >> (2 * (scatter.chunk_count - 1))) & 0b11;
        return last_encoding == tt::tt_fabric::NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE;
    }
    return false;
}

// EXP_RX_LOCAL_AS_PUSH: the local-write command buffer keeps its state pointed at this core's L1 scratch.
static_assert(
    !fabric_exp_rx_local_as_push ||
        (receiver_channel_forwarding_data_cmd_buf_ids[0] != tt::tt_fabric::local_chip_data_cmd_buf &&
         receiver_channel_forwarding_sync_cmd_buf_ids[0] != tt::tt_fabric::local_chip_data_cmd_buf),
    "EXP_RX_LOCAL_AS_PUSH needs a local-write command buffer that forwarding does not share");
// EXP_RX_PUSH_LEAN: the push's doorbell goes to a stream register on this tile that the router does not use.
constexpr uint32_t exp_push_doorbell_stream_id = 26;
template <size_t N>
constexpr bool exp_stream_unused_in(const std::array<uint32_t, N>& ids, uint32_t id) {
    for (size_t i = 0; i < N; i++) {
        if (ids[i] == id) {
            return false;
        }
    }
    return true;
}
static_assert(
    !fabric_exp_rx_push_lean ||
        (exp_stream_unused_in(sender_channel_free_slots_stream_ids, exp_push_doorbell_stream_id) &&
         exp_stream_unused_in(to_receiver_packets_sent_streams, exp_push_doorbell_stream_id) &&
         exp_stream_unused_in(to_sender_packets_acked_streams, exp_push_doorbell_stream_id) &&
         exp_stream_unused_in(to_sender_packets_completed_streams, exp_push_doorbell_stream_id) &&
         vc_0_free_slots_from_downstream_edge_1_stream_id != exp_push_doorbell_stream_id &&
         vc_0_free_slots_from_downstream_edge_2_stream_id != exp_push_doorbell_stream_id &&
         vc_0_free_slots_from_downstream_edge_3_stream_id != exp_push_doorbell_stream_id &&
         vc_0_free_slots_from_downstream_edge_4_stream_id != exp_push_doorbell_stream_id),
    "EXP_RX_PUSH_LEAN's doorbell stream register is in use");
uint64_t exp_push_doorbell_noc_addr;

// EXP_RX_PUSH_DRISC: payload slots in the DRISC's free L1 (one channel buffer each), and the DRISC stream register the
// doorbell adds to (one the DRISC microbenchmark used as its doorbell under the DRISC firmware).
// In 2D two-ERISC mode, forwarding uses stateless noc_inline_dw_write<REG> for downstream credits, which hardcodes
// write_at_cmd_buf (3) rather than receiver_channel_forwarding_sync_cmd_buf_ids (read_cmd_buf = 1). Putting the
// stateful doorbell on write_at_cmd_buf therefore collides with forwarding on packets that both forward and deliver
// locally (forwarding overwrites buffer 3 with the downstream router's credit register and -1, and the stateful
// doorbell then sends a second -1 to the downstream router). read_cmd_buf (1) is unused on the receiving ERISC in 2D
// two-ERISC mode once setup_edm_noc_cmd_buf has run before the main loop. In 1D, stateful forwarding uses read_cmd_buf
// (1) and leaves write_at_cmd_buf (3) to local atomics.
constexpr uint32_t exp_push_drisc_doorbell_stream_id = 13;
constexpr uint32_t exp_push_drisc_slot_bytes = channel_buffer_size;
constexpr uint32_t exp_push_drisc_slots = !fabric_exp_rx_push_drisc ? 0
                                          : exp_push_drisc_l1_size / exp_push_drisc_slot_bytes >= 8
                                              ? 8
                                              : exp_push_drisc_l1_size / exp_push_drisc_slot_bytes;
constexpr uint32_t exp_push_drisc_ring_end = exp_push_drisc_l1_base + exp_push_drisc_slots * exp_push_drisc_slot_bytes;
#if defined(FABRIC_2D)
constexpr uint8_t exp_push_doorbell_cmd_buf = read_cmd_buf;
static_assert(
    !fabric_exp_rx_push_drisc || NUM_ACTIVE_ERISCS > 1,
    "EXP_RX_PUSH_DRISC in 2D uses read_cmd_buf, which single-ERISC stateful 2D forwarding also uses");
#else
constexpr uint8_t exp_push_doorbell_cmd_buf = write_at_cmd_buf;
#endif
static_assert(
    !fabric_exp_rx_push_drisc || !any_receiver_channel_serviced ||
        (exp_push_drisc_slots > 0 && exp_push_drisc_l1_base % 16 == 0 && exp_push_drisc_slot_bytes % 16 == 0),
    "EXP_RX_PUSH_DRISC: the builder found no DRISC with free L1 for this router");
static_assert(
    !fabric_exp_rx_push_drisc || (exp_push_doorbell_cmd_buf != tt::tt_fabric::local_chip_data_cmd_buf &&
                                  exp_push_doorbell_cmd_buf != receiver_channel_forwarding_data_cmd_buf_ids[0]),
    "EXP_RX_PUSH_DRISC's doorbell command buffer is used by the push or by forwarding");
uint32_t exp_push_drisc_next_slot;

FORCE_INLINE void exp_push_set_state() {
    if constexpr (fabric_exp_rx_push_drisc) {
        noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
            get_noc_addr(
                exp_push_drisc_x,
                exp_push_drisc_y,
                STREAM_REG_ADDR(
                    exp_push_drisc_doorbell_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
                tt::tt_fabric::edm_to_local_chip_noc),
            1u << REMOTE_DEST_BUF_WORDS_FREE_INC,
            0xF,
            exp_push_doorbell_cmd_buf,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
        noc_async_write_one_packet_with_trid_set_state<false>(
            get_noc_addr(
                exp_push_drisc_x, exp_push_drisc_y, exp_push_drisc_l1_base, tt::tt_fabric::edm_to_local_chip_noc),
            tt::tt_fabric::local_chip_data_cmd_buf,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
        NOC_CMD_BUF_WRITE_REG(
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::local_chip_data_cmd_buf,
            NOC_AT_LEN_BE,
            exp_push_drisc_slot_bytes);
        return;
    }
    if constexpr (fabric_exp_rx_push_lean) {
        exp_push_doorbell_noc_addr = get_noc_addr(
            my_x[tt::tt_fabric::edm_to_local_chip_noc],
            my_y[tt::tt_fabric::edm_to_local_chip_noc],
            STREAM_REG_ADDR(exp_push_doorbell_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
            tt::tt_fabric::edm_to_local_chip_noc);
    }
    if constexpr (fabric_exp_rx_local_as_push) {
        noc_async_write_one_packet_with_trid_set_state<false>(
            get_noc_addr(
                my_x[tt::tt_fabric::edm_to_local_chip_noc],
                my_y[tt::tt_fabric::edm_to_local_chip_noc],
                exp_push_loopback_scratch_addr,
                tt::tt_fabric::edm_to_local_chip_noc),
            tt::tt_fabric::local_chip_data_cmd_buf,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
    }
}

#if defined(FABRIC_DRISC_OFFLOAD)
static_assert(
    drisc_ring_base_addr + drisc_ring_slots * channel_buffer_size <= exp_push_drisc_l1_base + exp_push_drisc_l1_size,
    "DRISC 16-slot staging ring exceeds DRISC UNRESERVED L1");
static_assert(
    exp_stream_unused_in(sender_channel_free_slots_stream_ids, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_receiver_packets_sent_streams, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_sender_packets_acked_streams, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_sender_packets_completed_streams, drisc_free_slots_stream_id),
    "FABRIC_DRISC_OFFLOAD free-slots stream register is in use on ERISC");

uint32_t drisc_offload_next_slot_addr;

FORCE_INLINE void drisc_offload_init_noc_state() {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        get_noc_addr(
            exp_push_drisc_x,
            exp_push_drisc_y,
            STREAM_REG_ADDR(drisc_doorbell_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
            noc),
        1u << REMOTE_DEST_BUF_WORDS_FREE_INC,
        0xF,
        read_cmd_buf,
        noc,
        tt::tt_fabric::forward_and_local_write_noc_vc);
    noc_async_write_one_packet_with_trid_set_state</*posted=*/false>(
        get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_ring_base_addr, noc),
        tt::tt_fabric::local_chip_data_cmd_buf,
        noc,
        tt::tt_fabric::forward_and_local_write_noc_vc);
}

FORCE_INLINE void drisc_offload_push_packet(uint32_t pkt_addr, uint32_t total_bytes, uint8_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    constexpr uint8_t data_cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    const uint32_t dst_addr = drisc_offload_next_slot_addr;
    const uint32_t next_addr = dst_addr + channel_buffer_size;
    drisc_offload_next_slot_addr = next_addr == drisc_ring_end_addr ? drisc_ring_base_addr : next_addr;

    while (!noc_cmd_buf_ready(noc, data_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, data_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    ncrisc_noc_write_with_state<
        DM_DEDICATED_NOC,
        /*posted=*/false,
        /*update_counter=*/true,
        /*one_packet=*/false,
        /*skip_cmdbuf_chk=*/true>(noc, data_cmd_buf, pkt_addr, dst_addr, total_bytes);
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        /*update_counter=*/true,
        /*posted=*/true,
        /*update_addr_hi=*/false,
        /*update_val=*/false,
        InlineWriteDst::REG>(0, 0, read_cmd_buf, noc);
}
#endif

// Local-chip delivery for the generic receiver path, with the throughput-experiment overrides applied
// (see FabricExperimentFlags). The speedy paths call execute_chip_unicast_to_local_chip_impl directly.
FORCE_INLINE void generic_rx_local_delivery_impl(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    const uint32_t local_start = phase_profiler.open<PHASE_RX_LOCAL>();
    if constexpr (fabric_exp_rx_skip_local_delivery) {
        // Data writes are dropped on purpose: measures the RX interval without their local-delivery work.
        if (!exp_local_delivery_is_data_only(packet_start, noc_send_type)) {
            execute_chip_unicast_to_local_chip_impl(
                packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id);
        }
    } else if constexpr (fabric_exp_rx_local_as_push) {
        if (exp_local_delivery_is_data_only(packet_start, noc_send_type)) {
            uint32_t push_dst = exp_push_loopback_scratch_addr;
            if constexpr (fabric_exp_rx_push_drisc) {
                push_dst = exp_push_drisc_next_slot;
                const uint32_t next = push_dst + exp_push_drisc_slot_bytes;
                exp_push_drisc_next_slot = next == exp_push_drisc_ring_end ? exp_push_drisc_l1_base : next;
            }
            if constexpr (fabric_exp_rx_push_lean) {
                // One command-buffer poll (the stock call polls twice), then the id, length, addresses and go.
                constexpr uint32_t noc = tt::tt_fabric::edm_to_local_chip_noc;
                constexpr uint32_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
                while (!noc_cmd_buf_ready(noc, cmd_buf));
                NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(transaction_id));
                ncrisc_noc_write_with_state<
                    DM_DEDICATED_NOC,
                    /*posted=*/false,
                    /*update_counter=*/true,
                    /*one_packet=*/false,
                    /*skip_cmdbuf_chk=*/true>(
                    noc,
                    cmd_buf,
                    reinterpret_cast<uint32_t>(packet_start),
                    push_dst,
                    payload_size_bytes + sizeof(PACKET_HEADER_TYPE));
                if constexpr (!fabric_exp_rx_push_drisc) {
                    noc_inline_dw_write<InlineWriteDst::REG>(
                        exp_push_doorbell_noc_addr, 1u << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xF, noc);
                }
            } else {
                noc_async_write_one_packet_with_trid_with_state<false>(
                    reinterpret_cast<size_t>(packet_start),
                    push_dst,
                    payload_size_bytes + sizeof(PACKET_HEADER_TYPE),
                    transaction_id,
                    tt::tt_fabric::local_chip_data_cmd_buf,
                    tt::tt_fabric::edm_to_local_chip_noc);
            }
            if constexpr (fabric_exp_rx_push_drisc) {
                // Stateful doorbell: destination and value were set once, so this is a command-buffer poll and a go.
                noc_inline_dw_write_with_state<
                    /*update_addr_lo=*/false,
                    /*update_counter=*/true,
                    /*posted=*/false,
                    /*update_addr_hi=*/false,
                    /*update_val=*/false,
                    InlineWriteDst::REG>(0, 0, exp_push_doorbell_cmd_buf, tt::tt_fabric::edm_to_local_chip_noc);
            }
        } else {
            execute_chip_unicast_to_local_chip_impl(
                packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id);
            exp_push_set_state();
        }
    } else if constexpr (fabric_exp_rx_local_as_single_write) {
        if (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE) {
            noc_async_write_one_packet_with_trid<false, false>(
                reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE),
                packet_start->command_fields.unicast_write.noc_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } else {
            execute_chip_unicast_to_local_chip_impl(
                packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id);
        }
    } else {
        execute_chip_unicast_to_local_chip_impl(
            packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id);
    }
    phase_profiler.close<PHASE_RX_LOCAL>(local_start);
}

FORCE_INLINE void generic_rx_local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    generic_rx_local_delivery_impl(
        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
}

template <typename DownstreamSenderT>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields, DownstreamSenderT& downstream_edm_interface) {
    // We always check if it is the terminal mcast packet value. We can do this because all unicast packets have the
    // mcast terminal value masked in to the routing field. This simplifies the check here to a single compare.
    bool deliver_locally_only;
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        deliver_locally_only = cached_routing_fields.value == tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        deliver_locally_only =
            (cached_routing_fields.value & LowLatencyFields::FIELD_MASK) == LowLatencyFields::WRITE_ONLY;
    }
    return deliver_locally_only || downstream_edm_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
}

// Map downstream direction to compact array index [0-3], excluding my_direction.
// Routers never forward back out their own facing, so that slot is unused and the remaining four
// directions pack densely.
// - EAST router  (my_direction=0): WEST(1)->0, NORTH(2)->1, SOUTH(3)->2, Z(4)->3
// - WEST router  (my_direction=1): EAST(0)->0, NORTH(2)->1, SOUTH(3)->2, Z(4)->3
// - NORTH router (my_direction=2): EAST(0)->0, WEST(1)->1,  SOUTH(3)->2, Z(4)->3
// - SOUTH router (my_direction=3): EAST(0)->0, WEST(1)->1,  NORTH(2)->2, Z(4)->3
// - Z router     (my_direction=4): EAST(0)->0, WEST(1)->1,  NORTH(2)->2, SOUTH(3)->3
constexpr size_t direction_to_compact_index_map[eth_chan_directions::COUNT][eth_chan_directions::COUNT] = {
    {0, 0, 1, 2, 3},  // EAST router -> WEST, NORTH, SOUTH, Z
    {0, 0, 1, 2, 3},  // WEST router -> EAST, NORTH, SOUTH, Z
    {0, 1, 0, 2, 3},  // NORTH router -> EAST, WEST, SOUTH, Z
    {0, 1, 2, 0, 3},  // SOUTH router -> EAST, WEST, NORTH, Z
    {0, 1, 2, 3, 0},  // Z router -> EAST, WEST, NORTH, SOUTH
};

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index() {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index(eth_chan_directions downstream_direction) {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t get_downstream_edm_interface_index() {
    // Map downstream direction to compact array index (excluding router's own direction)
    size_t downstream_edm_interface_index = map_downstream_direction_to_compact_index<downstream_direction>();

    return downstream_edm_interface_index;
}

FORCE_INLINE constexpr size_t get_downstream_edm_interface_index(eth_chan_directions downstream_direction) {
    // Map downstream direction to compact array index (excluding router's own direction)
    return map_downstream_direction_to_compact_index(downstream_direction);
}

// Whether this router actually instantiates the downstream slot a direction maps to. A compact index
// is fixed by the direction pair alone, so a narrow router simply leaves the higher slots absent --
// X_RING_ONLY gives an E/W-facing router only its opposite, so slots 1 and 2 do not exist there.
// NOTE (Z-arm invariant): a router with no Z downstream must never index past its downstream array.
// fwd_dirs<MY_DIR>() always lists Z, but the linear LIVE scan only emits a runtime ACTION_Z test when
// that slot's compact index is < DOWNSTREAM_EDM_SIZE. Wormhole has no Z sender, so the test is
// if-constexpr gone. num_z_ports is not visible to the kernel to cross-check against.
template <size_t DOWNSTREAM_EDM_SIZE, eth_chan_directions DIRECTION>
constexpr bool dispatch_arm_is_realizable() {
    return get_downstream_edm_interface_index<DIRECTION>() < DOWNSTREAM_EDM_SIZE;
}

// The selected downstream array determines the carrier VC. Query its adapter so crossover traffic
// reads VC1 credits rather than the compact-indexed VC0 counter.
template <typename DownstreamSenderT, eth_chan_directions DIRECTION, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE bool check_downstream_has_space(
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces) {
    if constexpr (DIRECTION == my_direction) {
        return true;
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index<DIRECTION>();
        return downstream_edm_interfaces[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    eth_chan_directions DIRECTION,
    size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE bool check_downstream_has_space(
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    if constexpr (DIRECTION == my_direction) {
        if constexpr (udm_mode) {
            return local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
        } else {
            return true;
        }
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index<DIRECTION>();
        return downstream_edm_interfaces[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    size_t DOWNSTREAM_EDM_SIZE,
    eth_chan_directions... DIRECTIONS>
FORCE_INLINE bool downstreams_have_space(
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    return (
        ... && check_downstream_has_space<DownstreamSenderT, LocalRelayInterfaceT, DIRECTIONS, DOWNSTREAM_EDM_SIZE>(
                   downstream_edm_interfaces, local_relay_interface));
}

#ifndef FABRIC_2D

// !!!WARNING!!! - MAKE SURE CONSUMER HAS SPACE BEFORE CALLING
template <uint8_t rx_channel_id, typename DownstreamSenderT>
FORCE_INLINE void receiver_forward_packet(
    // TODO: have a separate cached copy of the packet header to save some additional L1 loads
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    DownstreamSenderT& downstream_edm_interface,
    uint8_t transaction_id) {
    constexpr bool ENABLE_STATEFUL_NOC_APIS =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
        !FORCE_ALL_PATHS_TO_USE_SAME_NOC && true;
#else
        false;
#endif
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();  // Make sure we have the latest packet header in L1
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        // If the packet is a terminal packet, then we can just deliver it locally
        bool start_distance_is_terminal_value =
            (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
            tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        bool not_last_destination_device = cached_routing_fields.value != tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
        // disable when dprint enabled due to noc cmd buf usage of DPRINT
        if (not_last_destination_device) {
            const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
            channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
        }
        if (start_distance_is_terminal_value) {
            generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
        }
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        switch (routing) {
            case LowLatencyFields::WRITE_ONLY:
                generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            case LowLatencyFields::FORWARD_ONLY: {
                const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                phase_profiler.close<PHASE_RX_FORWARD>(forward_start);

                // In 1D, the forwarding sender channel is always sender channel 1
                // (channel 0 is worker, channel 1 receives forwarded traffic from upstream)
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            } break;
            case LowLatencyFields::WRITE_AND_FORWARD: {
                // Resolve noc_send_type via the same packed 4B load the local-write path uses, then
                // reuse it through the _impl entry point so the standard path does no extra L1 read
                // versus the non-sparse build. The sparse case is unlikely, keeping the common path hot.
                const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
                if (packed.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
                    // This chip writes its own page group (counts[chip_idx] pages from write_idx), then
                    // write_idx/chip_idx advance so the downstream chip picks the next group. The advance
                    // must precede the forward because forwarding sends the (mutated) header downstream,
                    // and must follow the write because the write reads the pre-advance chip_idx.
                    generic_rx_local_delivery_impl(
                        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                    auto& sparse = packet_start->command_fields.sparse_mcast_write;
                    sparse.write_idx += sparse.counts[sparse.chip_idx];
                    sparse.chip_idx++;
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                } else {
                    const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                    phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                    generic_rx_local_delivery_impl(
                        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                }
                // In 1D, the forwarding sender channel is always sender channel 1
                // (channel 0 is worker, channel 1 receives forwarded traffic from upstream)
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            } break;
            default: {
                ASSERT(false);
            }
        }
    }
}

#endif  // !FABRIC_2D

#if defined(FABRIC_2D)

// Helper to forward packet to local destination
// (relay in UDM mode, or local chip directly in non-UDM mode)
template <uint8_t rx_channel_id, typename LocalRelayInterfaceT>
FORCE_INLINE void forward_to_local_destination(
    LocalRelayInterfaceT& local_relay_interface,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    uint16_t payload_size_bytes,
    uint8_t transaction_id) {
    if constexpr (udm_mode) {
        execute_chip_unicast_to_relay(
            local_relay_interface, packet_start, payload_size_bytes, transaction_id, rx_channel_id);
    } else {
        generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
    }
}

// ============================================================================
// 2D action-map routing — admit/forward dispatch
// ============================================================================
// Test each action bit directly against fwd_dirs<MY_DIR>()'s LIVE slots. Admission must succeed for
// every selected output before any copy is committed; each receives an identical packet image, with
// local delivery last.
//
// A compile-time linear scan avoids GCC jump tables in ERISC_APP_KERNEL_DATA (ETH L1, ~8-cycle load
// then jump). Unrealizable slots are removed with if constexpr. Extra action bits outside LIVE are
// ignored because refusing them would hang the slot without a software retry path.
//
// 2D action-map transit consumes no hop program, so no header update runs here.

// Packed-slot bitset of eth outputs this ELF actually has. Bit i is fwd_dirs<MY_DIR>()[i].
// Intramesh E/W also drop N/S (packed bits 1 and 2): Y-before-X forbids those turns, and
// decode_action for E/W never reads the Y byte. INTERMESH E/W landings keep them.
template <size_t DOWNSTREAM_EDM_SIZE>
constexpr uint8_t live_eth_mask_2d() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr auto dirs = Routing2DCodec::fwd_dirs<my_dir>();
    uint8_t live = 0;
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[0]>()) {
        live |= static_cast<uint8_t>(1u << 0);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[1]>()) {
        live |= static_cast<uint8_t>(1u << 1);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[2]>()) {
        live |= static_cast<uint8_t>(1u << 2);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[3]>()) {
        live |= static_cast<uint8_t>(1u << 3);
    }
    constexpr bool trim_ns_for_intramesh_ew =
        (my_dir == eth_chan_directions::EAST || my_dir == eth_chan_directions::WEST) && !is_intermesh_router;
    if constexpr (trim_ns_for_intramesh_ew) {
        live = static_cast<uint8_t>(live & ~0b0110);
    }
    return live;
}

// LIVE slot i is compact index i: fwd_dirs is enum order with self removed, matching
// direction_to_compact_index_map and host direction_compact_index. The credit-read helper
// is compact-keyed, so this identity is what keeps admit's stream id on the same slot init wrote.
constexpr bool live_slot_is_compact_index() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr auto dirs = Routing2DCodec::fwd_dirs<my_dir>();
    return get_downstream_edm_interface_index<dirs[0]>() == 0 && get_downstream_edm_interface_index<dirs[1]>() == 1 &&
           get_downstream_edm_interface_index<dirs[2]>() == 2 && get_downstream_edm_interface_index<dirs[3]>() == 3;
}
static_assert(live_slot_is_compact_index());

// A pure-mesh E/W router has exactly one LIVE output (its opposite direction, compact slot 0).
// Its existing data/sync command buffers can therefore retain that adapter's state across packets.
// Any configuration with another runtime user of those command buffers stays on the non-stateful path.
template <size_t DOWNSTREAM_EDM_SIZE>
constexpr bool use_2d_single_live_stateful_noc() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr bool is_ew = my_dir == eth_chan_directions::EAST || my_dir == eth_chan_directions::WEST;
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED) && !defined(FABRIC_2D_VC1_ACTIVE)
    return is_ew && LIVE == 0b0001 && NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode &&
           !is_intermesh_router && !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    return false;
#endif
}

// Checks local relay capacity when ld, plus the downstream queue for every LIVE direction whose
// action bit is set. Only UDM mode queues local delivery through a relay interface.
template <typename DownstreamSenderT, typename LocalRelayInterfaceT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) bool admit_2d_dispatch(
    uint8_t action,
    bool ld,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    bool ok = true;
    if constexpr (udm_mode) {
        ok = !ld || local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
    if constexpr ((LIVE >> 0) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[0])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[0]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 1) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[1])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[1]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 2) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[2])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[2]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 3) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[3])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[3]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    return ok;
}

// One unmodified full-packet copy per LIVE direction whose action bit is set. Remote receiver
// credit is left to the sender step's bubble flow control, and local delivery to the caller.
template <typename DownstreamSenderT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) uint8_t forward_2d_dispatch(
    uint8_t action,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    uint8_t transaction_id) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    constexpr bool stateful_api = use_2d_single_live_stateful_noc<DOWNSTREAM_EDM_SIZE>();
    const uint16_t payload_size_bytes = packet_start->payload_size_bytes;
    uint8_t forwarded_slot_mask = 0;
    if constexpr ((LIVE >> 0) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[0])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[0]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 0;
        }
    }
    if constexpr ((LIVE >> 1) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[1])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[1]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 1;
        }
    }
    if constexpr ((LIVE >> 2) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[2])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[2]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 2;
        }
    }
    if constexpr ((LIVE >> 3) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[3])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[3]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 3;
        }
    }
    return forwarded_slot_mask;
}

// The host-provided slot map translates this receiver's compact downstream indices into each
// destination router's absolute sender channels. The row remains the receiving VC, including
// the VC0-to-VC1 crossover case.
template <uint8_t rx_channel_id>
FORCE_INLINE void record_2d_forwarded_slots(uint8_t forwarded_slot_mask) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
        constexpr uint32_t packed_downstream_sender_channel_ids = PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
#else
        constexpr uint32_t packed_downstream_sender_channel_ids =
            rx_channel_id == 0 ? PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS : PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
#endif
        const uint16_t sender_mask =
            tt::tt_fabric::forwarded_slots_to_sender_mask(forwarded_slot_mask, packed_downstream_sender_channel_ids);
        channel_trimming_usage_recorder.merge_sender_channel_forwarded_to(rx_channel_id, sender_mask);
    }
}

#endif

template <typename EdmChannelWorkerIFs>
FORCE_INLINE void establish_edm_connection(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface, uint32_t stream_id) {
    local_sender_channel_worker_interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
}

bool any_sender_channels_active(
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        // The table is fabric-flat: this router's compact channel i sits at its fabric position.
        if (get_ptr_val(local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(i)]) !=
            static_cast<int32_t>(SENDER_NUM_BUFFERS_ARRAY[i])) {
            return true;
        }
    }
    return false;
}

/*
 * In Blackhole, there are typically 2-eriscs running cooperatively to implement the fabric router. They
 * execute the sender and receiver traffic flows, respectively (hence they run completely independently).
 * However, in Blackhole, there is special behaviour that must be accommodated,  specifically around link
 * training. During link training, the erisc does not have direct access to a part of the Ethernet subsystem
 * that needs to be programmed; it is only accessible over the noc. Therefore, to avoid potential noc
 * resource conflicts between the non-link-training erisc and the link training erisc (because we don't
 * know exactly which nocs they will use), the router implements a coordinated context switch where
 * the non-link-training erisc will spin idly while the retrain process takes place.
 */
constexpr bool IS_RETRAIN_SYNC_MASTER() { return enable_context_switch; }

// Signals intent to enter coordinated context switch and waits for erisc1 acknowledgement.
// After this returns, erisc1 is in a wait loop and will not execute routing logic.
FORCE_INLINE void coordinated_context_switch_start_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_INTENT));
        // Wait for erisc1 to ack
        while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
               static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::INTENT_ACK)) {
            if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                return;
            }
        }
    }
}

// Signals completion of coordinated context switch and waits for erisc1 acknowledgement.
// After this returns, erisc1 is free to resume normal routing execution.
FORCE_INLINE void coordinated_context_switch_finish_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_COMPLETE));
        // Wait for erisc1 to ack
        while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
               static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::COMPLETE_ACK)) {
            if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                return;
            }
        }
        // Resume normal operation
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION));
    }
}

void run_routing_without_noc_sync_coordinated_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        coordinated_context_switch_start_as_master(termination_signal_ptr);
        run_routing_without_noc_sync();
        coordinated_context_switch_finish_as_master(termination_signal_ptr);
    }
}
FORCE_INLINE void run_routing_without_noc_sync_coordinated_as_non_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (!IS_RETRAIN_SYNC_MASTER()) {
        if (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() ==
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_INTENT)) {
            write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
                static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::INTENT_ACK));
            while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
                   static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_COMPLETE)) {
                if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                    return;
                }
            }
            write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
                static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::COMPLETE_ACK));
            while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
                   static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION)) {
                if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                    return;
                }
            }
        }
    }
}
void run_coordinated_context_switch_to_base_firmware(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (NUM_ACTIVE_ERISCS == 1) {
        run_routing_without_noc_sync();
    } else {
        if constexpr (IS_RETRAIN_SYNC_MASTER()) {
            run_routing_without_noc_sync_coordinated_as_master(termination_signal_ptr);
        }
    }
}
template <typename LocalTelemetryT>
FORCE_INLINE void update_telemetry(
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids_ordered,
    bool tx_progress,
    bool rx_progress,
    LocalTelemetryT& local_fabric_telemetry,
    volatile tt_l1_ptr LocalTelemetryT* fabric_telemetry) {
    if constexpr (FABRIC_TELEMETRY_ROUTER_STATE) {
        const auto* routing_table_l1 =
            reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
        auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
        fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = state_manager_l1->state;
    }
    if constexpr (FABRIC_TELEMETRY_HEARTBEAT_TX) {
        bool sender_idle = false;
        if (!tx_progress) {
            sender_idle = !any_sender_channels_active(local_sender_channel_free_slots_stream_ids_ordered);
        }
        if (tx_progress || sender_idle) {
            volatile RiscTimestampV2* tx_heartbeat_addr =
                &fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat;
            local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat.full++;
            tx_heartbeat_addr->full = local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat.full;
        }
    }
    if constexpr (FABRIC_TELEMETRY_HEARTBEAT_RX) {
        bool receiver_idle = false;
        if (!rx_progress) {
            receiver_idle = (get_ptr_val<to_receiver_packets_sent_streams[0]>() == 0);
        }
        if (rx_progress || receiver_idle) {
            volatile RiscTimestampV2* rx_heartbeat_addr =
                &fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat;
            local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat.full++;
            rx_heartbeat_addr->full = local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat.full;
        }
    }

    if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
        // Helper to safely write to volatile BandwidthTelemetry destinations without discarding qualifiers
        auto store_bandwidth_telemetry = [](volatile BandwidthTelemetry* dst, const BandwidthTelemetry& src) {
            dst->elapsed_active_cycles.full = src.elapsed_active_cycles.full;
            dst->elapsed_cycles.full = src.elapsed_cycles.full;
            dst->num_words_sent = src.num_words_sent;
            dst->num_packets_sent = src.num_packets_sent;
        };

        if constexpr (NUM_ACTIVE_ERISCS == 1) {
            store_bandwidth_telemetry(
                &fabric_telemetry->dynamic_info.tx_bandwidth, local_fabric_telemetry.dynamic_info.tx_bandwidth);
            store_bandwidth_telemetry(
                &fabric_telemetry->dynamic_info.rx_bandwidth, local_fabric_telemetry.dynamic_info.rx_bandwidth);
        } else {
            if constexpr (MY_ERISC_ID == 0) {
                store_bandwidth_telemetry(
                    &fabric_telemetry->dynamic_info.tx_bandwidth, local_fabric_telemetry.dynamic_info.tx_bandwidth);
            } else {
                store_bandwidth_telemetry(
                    &fabric_telemetry->dynamic_info.rx_bandwidth, local_fabric_telemetry.dynamic_info.rx_bandwidth);
            }
        }
    }
}

template <bool enable_deadlock_avoidance, bool SKIP_CONNECTION_LIVENESS_CHECK, typename EdmChannelWorkerIFs>
FORCE_INLINE void send_credits_to_upstream_workers(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface,
    int32_t num_credits,
    bool channel_connection_established) {
    if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
        local_sender_channel_worker_interface
            .template notify_persistent_connection_of_free_space<enable_deadlock_avoidance>(num_credits);
    } else {
        // Connection liveness checks are only done for connections that are not persistent
        // For those connections, it's unsafe to use free-slots counters held in stream registers
        // due to the lack of race avoidant connection protocol. Therefore, we update our read counter
        // instead because these connections will be read/write counter based instead
        local_sender_channel_worker_interface.increment_local_read_counter(num_credits);
        if (channel_connection_established) {
            local_sender_channel_worker_interface
                .template notify_worker_of_read_counter_update<enable_read_counter_update_noc_flush>();
        } else {
            local_sender_channel_worker_interface.copy_read_counter_to_worker_location_info();
            // If not connected, we update the read counter in L1 as well so the next connecting worker
            // is more likely to see space available as soon as it tries connecting
        }
    }
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_bw_counters(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, LocalTelemetryT& local_fabric_telemetry) {
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 0)) {
        size_t packet_bytes = packet_header->get_payload_size_including_header();
        local_fabric_telemetry.dynamic_info.tx_bandwidth.num_packets_sent++;
        local_fabric_telemetry.dynamic_info.tx_bandwidth.num_words_sent += (packet_bytes + 3) >> 2;
    }
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 1)) {
        size_t packet_bytes = packet_header->get_payload_size_including_header();
        local_fabric_telemetry.dynamic_info.rx_bandwidth.num_packets_sent++;
        local_fabric_telemetry.dynamic_info.rx_bandwidth.num_words_sent += (packet_bytes + 3) >> 2;
    }
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_bw_cycles(
    uint64_t loop_delta_cycles, bool tx_progress, bool rx_progress, LocalTelemetryT& local_fabric_telemetry) {
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 0)) {
        local_fabric_telemetry.dynamic_info.tx_bandwidth.elapsed_cycles.full += loop_delta_cycles;
        if (tx_progress) {
            local_fabric_telemetry.dynamic_info.tx_bandwidth.elapsed_active_cycles.full += loop_delta_cycles;
        }
    }
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 1)) {
        local_fabric_telemetry.dynamic_info.rx_bandwidth.elapsed_cycles.full += loop_delta_cycles;
        if (rx_progress) {
            local_fabric_telemetry.dynamic_info.rx_bandwidth.elapsed_active_cycles.full += loop_delta_cycles;
        }
    }
}

// Experiment (TT_FABRIC_EXP_STACK_PAINT): stack depth. kernel_main fills the free stack, from its stack pointer down
// to the end of this kernel's local-data-memory sections, with a pattern; the outer loop rescans it every 64
// iterations (and teardown once more) and records the lowest word written in this ERISC's phase-profile block. On
// ERISC0 this includes the base firmware routines the router calls on its own stack (context switch, link status).
constexpr uint32_t exp_stack_pattern = 0x57AC57AC;
uint32_t exp_stack_main_sp;
uint32_t exp_stack_scan_counter;

FORCE_INLINE uint32_t exp_stack_limit() {
    extern uint32_t __ldm_bss_end[];
    return (reinterpret_cast<uint32_t>(__ldm_bss_end) + 15) & ~15u;
}

FORCE_INLINE void exp_stack_report(uint32_t lowest_used) {
    auto* block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        phase_profile_buffer_addr + MY_ERISC_ID * sizeof(FabricPhaseProfile));
    block[offsetof(FabricPhaseProfile, stack_main_sp) / sizeof(uint32_t)] = exp_stack_main_sp;
    block[offsetof(FabricPhaseProfile, stack_lowest_used) / sizeof(uint32_t)] = lowest_used;
    block[offsetof(FabricPhaseProfile, stack_limit) / sizeof(uint32_t)] = exp_stack_limit();
    block[offsetof(FabricPhaseProfile, stack_painted) / sizeof(uint32_t)] = 1;
}

// Inlined into kernel_main, so the fill starts right below kernel_main's frame and uses no stack itself.
FORCE_INLINE void exp_paint_stack() {
    if constexpr (fabric_exp_stack_paint) {
        uint32_t sp;
        asm volatile("mv %0, sp" : "=r"(sp));
        exp_stack_main_sp = sp;
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(sp & ~15u);
        volatile uint32_t* const end = reinterpret_cast<volatile uint32_t*>(exp_stack_limit());
        while (p > end) {
            *--p = exp_stack_pattern;
        }
        exp_stack_report(sp);
    }
}

FORCE_INLINE void exp_scan_stack(bool force = false) {
    if constexpr (fabric_exp_stack_paint) {
        if (force || (++exp_stack_scan_counter & 63) == 0) {
            volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(exp_stack_limit());
            while (*p == exp_stack_pattern) {
                p++;
            }
            exp_stack_report(reinterpret_cast<uint32_t>(p));
        }
    }
}

// Experiment (EXP_TX_UPSTREAM_ON_E1), run by E1: the upstream side of the sender channels E0 services. Completions come
// from the same L1 counters E0 reads (each core keeps its own processed count); the first four channels' counters are
// contiguous and read in one group, so their latencies overlap. E1's main loop only checks the in-flight register
// (exp_upstream_check) and calls the work out of line, when packets are in flight or every 16th iteration for the
// worker connection check, so the upstream code does not take registers from the receive path.
uint32_t exp_upstream_iteration;
// Packets E0 has sent that E1 has not yet credited upstream: E0 adds one per send, E1 subtracts the completions it
// processes. E1 skips the counter reads while it reads zero, so an ERISC whose E0 sends nothing pays one register
// read per iteration. Stream 24 is unused by the router's 2D VC0 assignments (0-8, 30, 31).
constexpr uint32_t exp_upstream_in_flight_stream_id = 24;
static_assert(
    !fabric_exp_tx_upstream_on_e1 ||
        [] {
            for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
                if (sender_channel_free_slots_stream_ids[i] == exp_upstream_in_flight_stream_id ||
                    to_sender_packets_acked_streams[i] == exp_upstream_in_flight_stream_id ||
                    to_sender_packets_completed_streams[i] == exp_upstream_in_flight_stream_id) {
                    return false;
                }
            }
            for (size_t i = 0; i < MAX_NUM_RECEIVER_CHANNELS; i++) {
                if (to_receiver_packets_sent_streams[i] == exp_upstream_in_flight_stream_id) {
                    return false;
                }
            }
            return true;
        }(),
    "EXP_TX_UPSTREAM_ON_E1 in-flight stream register collides with a router stream");

template <size_t I, typename WorkerIFs, typename Credits, size_t N>
FORCE_INLINE void exp_upstream_channel(
    WorkerIFs& worker_interfaces,
    std::array<bool, N>& connection_established,
    Credits& credits,
    uint32_t completions_received,
    bool check_connections,
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& free_slots_stream_ids) {
    if constexpr (I < N && peer_sender_channel_serviced[I]) {
        auto& channel_credits = credits.template get<I>();
        auto& worker_interface = worker_interfaces.template get<I>();
        const uint32_t completions = completions_received - channel_credits.completions_received_and_processed;
        if (completions != 0) {
            channel_credits.increment_num_processed_completions(completions);
            increment_local_update_ptr_val<exp_upstream_in_flight_stream_id>(-static_cast<int32_t>(completions));
            send_credits_to_upstream_workers<enable_deadlock_avoidance, sender_ch_live_check_skip[I]>(
                worker_interface, completions, connection_established[I]);
        }
        if constexpr (!sender_ch_live_check_skip[I]) {
            if (check_connections && (!connection_established[I] || worker_interface.has_worker_teardown_request())) {
                check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                    worker_interface,
                    connection_established[I],
                    free_slots_stream_ids[fabric_position_for_compact_sender(I)]);
            }
        }
    }
}

template <typename WorkerIFs, typename Credits, size_t N>
__attribute__((noinline)) void exp_run_upstream_steps(
    WorkerIFs& worker_interfaces,
    std::array<bool, N>& connection_established,
    Credits& credits,
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& free_slots_stream_ids,
    bool check_connections) {
    uint32_t received[4];
    asm volatile("lw %0, 0(%4)\n lw %1, 4(%4)\n lw %2, 8(%4)\n lw %3, 12(%4)\n"
                 : "=&r"(received[0]), "=&r"(received[1]), "=&r"(received[2]), "=&r"(received[3])
                 : "r"(to_sender_remote_completion_counters_base_address)
                 : "memory");
    exp_upstream_channel<0>(
        worker_interfaces, connection_established, credits, received[0], check_connections, free_slots_stream_ids);
    exp_upstream_channel<1>(
        worker_interfaces, connection_established, credits, received[1], check_connections, free_slots_stream_ids);
    exp_upstream_channel<2>(
        worker_interfaces, connection_established, credits, received[2], check_connections, free_slots_stream_ids);
    exp_upstream_channel<3>(
        worker_interfaces, connection_established, credits, received[3], check_connections, free_slots_stream_ids);
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (exp_upstream_channel<Is + 4>(
             worker_interfaces,
             connection_established,
             credits,
             reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address)[Is + 4],
             check_connections,
             free_slots_stream_ids),
         ...);
    }(std::make_index_sequence<(N > 4 ? N - 4 : 0)>{});
}

// Experiment (EXP_TX_GROUPED_PACKET_CHECK): the free-slot stream registers of the first (up to) four VC0 sender
// channels are read back to back in one asm block at the top of each loop iteration. Reads issued together overlap
// (RESULTS §17: four cost ~4 cycles, against ~7 each when used at once); each channel masks its raw value where it
// uses it, so nothing consumes a read right after it is issued.
constexpr size_t exp_grouped_packet_check_channels = ACTUAL_VC0_SENDER_CHANNELS < 4 ? ACTUAL_VC0_SENDER_CHANNELS : 4;

template <size_t CH>
constexpr uint32_t exp_free_slots_reg_addr() {
    return STREAM_REG_ADDR(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <size_t N = exp_grouped_packet_check_channels>
FORCE_INLINE void exp_read_grouped_free_slots(uint32_t (&raw)[4]) {
    if constexpr (N == 4) {
        asm volatile("lw %0, 0(%4)\n lw %1, 0(%5)\n lw %2, 0(%6)\n lw %3, 0(%7)\n"
                     : "=&r"(raw[0]), "=&r"(raw[1]), "=&r"(raw[2]), "=&r"(raw[3])
                     : "r"(exp_free_slots_reg_addr<0>()),
                       "r"(exp_free_slots_reg_addr<1>()),
                       "r"(exp_free_slots_reg_addr<2>()),
                       "r"(exp_free_slots_reg_addr<3>())
                     : "memory");
    } else if constexpr (N == 3) {
        asm volatile(
            "lw %0, 0(%3)\n lw %1, 0(%4)\n lw %2, 0(%5)\n"
            : "=&r"(raw[0]), "=&r"(raw[1]), "=&r"(raw[2])
            : "r"(exp_free_slots_reg_addr<0>()), "r"(exp_free_slots_reg_addr<1>()), "r"(exp_free_slots_reg_addr<2>())
            : "memory");
    } else if constexpr (N == 2) {
        asm volatile("lw %0, 0(%2)\n lw %1, 0(%3)\n"
                     : "=&r"(raw[0]), "=&r"(raw[1])
                     : "r"(exp_free_slots_reg_addr<0>()), "r"(exp_free_slots_reg_addr<1>())
                     : "memory");
    } else if constexpr (N == 1) {
        // A read issued alone does not overlap (RESULTS §17); kept for completeness.
        asm volatile("lw %0, 0(%1)\n" : "=&r"(raw[0]) : "r"(exp_free_slots_reg_addr<0>()) : "memory");
    }
}

// Experiment (EXP_TX_SKIP_IDLE_CHANNELS / EXP_TX_SKIP_IDLE_PER_CHANNEL): the forwarding channels in the grouped read
// (compact 1..3) are skipped while they have no packet and no packets awaiting completion. A completion only follows
// a send (EXP_TX_LEAN_COMPLETIONS), so a skipped visit misses no event. Their visits must have nothing else to do:
// no connection liveness check, no batched upstream credits, no first-level acks, upstream credits on this ERISC.
constexpr bool exp_skip_idle_channel(size_t ch) {
    return ch >= 1 && ch < exp_grouped_packet_check_channels && is_sender_channel_serviced[ch];
}
constexpr bool exp_skip_idle_supported = [] {
    // The main loop wraps the visits to channels 1 (and 2, 3 in 2D) in the skip: each serviced one must be covered.
    for (size_t ch = 1; ch <= (is_2d_fabric ? 3 : 1) && ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        if (is_sender_channel_serviced[ch] && !exp_skip_idle_channel(ch)) {
            return false;
        }
    }
    bool any = false;
    for (size_t ch = 1; ch < exp_grouped_packet_check_channels; ch++) {
        if (is_sender_channel_serviced[ch]) {
            if (!sender_ch_live_check_skip[ch]) {
                return false;
            }
            any = true;
        }
    }
    return any && fabric_exp_tx_lean_completions && fabric_exp_tx_grouped_packet_check && !ENABLE_FIRST_LEVEL_ACK_VC0 &&
           !fabric_exp_tx_amortize_worker_credits && !fabric_exp_tx_upstream_on_e1;
}();
constexpr bool fabric_exp_tx_skip_idle_channels =
    (fabric_exp_flags & EXP_TX_SKIP_IDLE_CHANNELS) != 0 && exp_skip_idle_supported;
constexpr bool fabric_exp_tx_skip_idle_per_channel =
    (fabric_exp_flags & EXP_TX_SKIP_IDLE_PER_CHANNEL) != 0 && exp_skip_idle_supported;

// Packets the skipped channels have sent and not yet had completed.
uint32_t exp_fwd_in_flight;

// A free-slot count runs from 0 to the channel's slot count; with a power-of-two slot count N, bit N is set only when
// all N slots are free, so one AND across the raw values tests every channel (the stream register's upper field sits
// above bit 17).
constexpr bool exp_skip_idle_bit_test = [] {
    size_t n = 0;
    for (size_t ch = 1; ch < exp_grouped_packet_check_channels; ch++) {
        if (exp_skip_idle_channel(ch)) {
            const size_t slots = SENDER_NUM_BUFFERS_ARRAY[ch];
            if ((slots & (slots - 1)) != 0 || slots >= (1u << REMOTE_DEST_WORDS_FREE_WIDTH) || (n != 0 && slots != n)) {
                return false;
            }
            n = slots;
        }
    }
    return n != 0;
}();

template <size_t CH>
FORCE_INLINE bool exp_channel_all_slots_free(uint32_t raw) {
    if constexpr (exp_skip_idle_bit_test) {
        return (raw & SENDER_NUM_BUFFERS_ARRAY[CH]) != 0;
    } else {
        return (raw & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1)) == SENDER_NUM_BUFFERS_ARRAY[CH];
    }
}

FORCE_INLINE bool exp_forwarding_channels_idle(const uint32_t (&raw)[4]) {
    if constexpr (exp_skip_idle_bit_test) {
        uint32_t all = ~0u;
        size_t slots = 0;
        if constexpr (exp_skip_idle_channel(1)) {
            all &= raw[1];
            slots = SENDER_NUM_BUFFERS_ARRAY[1];
        }
        if constexpr (exp_skip_idle_channel(2)) {
            all &= raw[2];
            slots = SENDER_NUM_BUFFERS_ARRAY[2];
        }
        if constexpr (exp_skip_idle_channel(3)) {
            all &= raw[3];
            slots = SENDER_NUM_BUFFERS_ARRAY[3];
        }
        return (all & slots) != 0 && exp_fwd_in_flight == 0;
    } else {
        bool all_free = true;
        if constexpr (exp_skip_idle_channel(1)) {
            all_free &= exp_channel_all_slots_free<1>(raw[1]);
        }
        if constexpr (exp_skip_idle_channel(2)) {
            all_free &= exp_channel_all_slots_free<2>(raw[2]);
        }
        if constexpr (exp_skip_idle_channel(3)) {
            all_free &= exp_channel_all_slots_free<3>(raw[3]);
        }
        return all_free && exp_fwd_in_flight == 0;
    }
}

////////////////////////////////////
////////////////////////////////////
//  Main Control Loop
////////////////////////////////////
////////////////////////////////////
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    bool enable_first_level_ack,
    uint8_t VC_RECEIVER_CHANNEL = 0,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename ReceiverChannelT,
    typename LocalTelemetryT,
    typename SenderChannelFromReceiverCreditsT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_sender_channel_step_impl(
        SenderChannelT& local_sender_channel,
        WorkerInterfaceT& local_sender_channel_worker_interface,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        ReceiverChannelT& remote_receiver_channel,
        bool& channel_connection_established,
        uint32_t sender_channel_free_slots_stream_id,
        SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t exp_grouped_free_slots_raw = 0) {
    bool progress = false;
    // If the receiver has space, and we have one or more packets unsent from producer, then send one
    // TODO: convert to loop to send multiple packets back to back (or support sending multiple packets in one shot)
    //       when moving to stream regs to manage rd/wr ptrs
    // TODO: update to be stream reg based. Initialize to space available and simply check for non-zero

    constexpr bool use_bubble_flow_control =
        sender_channel_is_traffic_injection_channel[sender_channel_index] && enable_deadlock_avoidance;
    static_assert(
        !use_bubble_flow_control || enable_first_level_ack,
        "Bubble flow control and first level ack must be set to the same values");

    if constexpr (fabric_exp_tx_skip_idle_per_channel && exp_skip_idle_channel(sender_channel_index)) {
        if constexpr (fabric_exp_tx_idle_likely) {
            if (exp_channel_all_slots_free<sender_channel_index>(exp_grouped_free_slots_raw) &&
                !sender_channel_from_receiver_credits.has_completions_pending()) [[likely]] {
                return false;
            }
        } else {
            if (exp_channel_all_slots_free<sender_channel_index>(exp_grouped_free_slots_raw) &&
                !sender_channel_from_receiver_credits.has_completions_pending()) {
                return false;
            }
        }
    }

    bool receiver_has_space_for_packet;
    if constexpr (use_bubble_flow_control) {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.num_free_slots >=
                                        BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    }
    uint32_t free_slots;
    if constexpr (fabric_exp_tx_grouped_packet_check && sender_channel_index < exp_grouped_packet_check_channels) {
        // Read at the top of the iteration with the other channels' registers (exp_read_grouped_free_slots).
        free_slots = exp_grouped_free_slots_raw & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
    } else {
        free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    }
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    // EXP_TX_SEND_EARLY_SIZE_READ: when a packet is ready, read its size just before the TX-queue reads so the L1
    // read is in flight with them. Only then: an unused read left in flight delays the next read used at once.
    uint32_t exp_payload_size_bytes = 0;
    if constexpr (fabric_exp_tx_send_early_size_read) {
        if (can_send) {
            exp_payload_size_bytes = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                                         local_sender_channel.get_cached_next_buffer_slot_addr())
                                         ->get_payload_size_including_header();
            if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_no_send_txq_check) {
                can_send = !SenderLinkTransport::is_busy();
            }
        }
    } else if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_no_send_txq_check) {
        can_send = can_send && !SenderLinkTransport::is_busy();
    }
    constexpr bool exp_overlap_send_completion =
#if defined(ARCH_BLACKHOLE)
        fabric_exp_tx_send_overlap_completion && sender_channel_uses_counter_credits(sender_channel_index) &&
        !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_send_early_txq_read && !ENABLE_RISC_CPU_DATA_CACHE;
#else
        false;
#endif
    int32_t completions_since_last_check = 0;
    if (can_send) {
        did_something = true;
        progress = true;
        phase_profiler.mark_progress();
        const uint32_t send_start = phase_profiler.open<PHASE_TX_SEND>();

        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        constexpr size_t rem_rx_entry_idx =
            eth_remote_channel_allocs::receiver_channel_to_entry_index[VC_RECEIVER_CHANNEL];
        constexpr uint32_t rem_rx_base_addr =
            static_cast<uint32_t>(eth_remote_channel_allocs::template Entry<rem_rx_entry_idx>::base_address);
        send_next_data<
            sender_channel_index,
            to_receiver_pkts_sent_id,
            SKIP_CONNECTION_LIVENESS_CHECK,
            exp_overlap_send_completion,
            rem_rx_base_addr>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            perf_telemetry_recorder,
            exp_payload_size_bytes);
        if constexpr (exp_overlap_send_completion) {
            constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + sender_txq_id * ETH_TXQ_REGS_SIZE;
            constexpr uint32_t rem_stream_addr =
                STREAM_REG_ADDR(to_receiver_pkts_sent_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
            const uint32_t cptr =
                reinterpret_cast<uint32_t>(sender_channel_from_receiver_credits.completions_received_counter_ptr);
            uint32_t sent = sender_channel_from_receiver_credits.packets_sent;
            const uint32_t proc = sender_channel_from_receiver_credits.completions_received_and_processed;
            uint32_t free_rem = outbound_to_receiver_channel_pointers.num_free_slots;
            uint32_t txq_st, comp_delta, cmd_wr;
            if constexpr (fabric_exp_tx_lean_vc0_loop || fabric_exp_tx_send_ct_free_slot_stream) {
                constexpr uint32_t ct_free_slot_stream_id =
                    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)];
                constexpr uint32_t ct_lreg_addr =
                    STREAM_REG_ADDR(ct_free_slot_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
                uint32_t lreg_addr;
                asm volatile(
                    "lw %[txq_st], 4(%[txq])\n\t"
                    "lw %[txq_st], 8(%[txq])\n\t"
                    "lw %[comp], 0(%[cptr])\n\t"
                    "addi %[sent], %[sent], 1\n\t"
                    "addi %[free], %[free], -1\n\t"
                    "lui %[lreg], %[lreg_hi]\n\t"
                    "addi %[lreg], %[lreg], %[lreg_lo]\n\t"
                    "li %[cmd_wr], 4\n\t"
                    "slli %[txq_st], %[txq_st], 15\n\t"
                    "sub %[comp], %[comp], %[proc]\n\t"
                    : [txq_st] "=&r"(txq_st),
                      [comp] "=&r"(comp_delta),
                      [cmd_wr] "=&r"(cmd_wr),
                      [lreg] "=&r"(lreg_addr),
                      [sent] "+r"(sent),
                      [free] "+r"(free_rem)
                    : [txq] "r"(txq_base),
                      [cptr] "r"(cptr),
                      [proc] "r"(proc),
                      [lreg_hi] "i"(((ct_lreg_addr + 0x800u) >> 12) & 0xFFFFFu),
                      [lreg_lo] "i"(static_cast<int32_t>(ct_lreg_addr << 20) >> 20)
                    : "memory");
                if (static_cast<int32_t>(txq_st) < 0) [[unlikely]] {
                    while (SenderLinkTransport::is_busy()) {
                    }
                }
                volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);
                txq_regs[ETH_TXQ_DEST_ADDR >> 2] = rem_stream_addr;
                txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
                txq_regs[ETH_TXQ_CMD >> 2] = cmd_wr;
                outbound_to_receiver_channel_pointers.num_free_slots = free_rem;
                if constexpr (fabric_exp_tx_lean_completions) {
                    sender_channel_from_receiver_credits.packets_sent = sent;
                }
                if constexpr (fabric_exp_tx_skip_idle_channels && exp_skip_idle_channel(sender_channel_index)) {
                    exp_fwd_in_flight++;
                }
                if constexpr (exp_tx_upstream_moved_off_this_erisc) {
                    increment_local_update_ptr_val<exp_upstream_in_flight_stream_id>(1);
                }
                if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                    update_bw_counters(pkt_header, local_fabric_telemetry);
                }
                *reinterpret_cast<volatile uint32_t*>(lreg_addr) = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
                completions_since_last_check = static_cast<int32_t>(comp_delta);
            } else {
                constexpr uint32_t stream0_update_addr =
                    STREAM_REG_ADDR(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
                uint32_t lreg = sender_channel_free_slots_stream_id;
                asm volatile(
                    "lw %[txq_st], 4(%[txq])\n\t"
                    "lw %[txq_st], 8(%[txq])\n\t"
                    "lw %[comp], 0(%[cptr])\n\t"
                    "addi %[sent], %[sent], 1\n\t"
                    "addi %[free], %[free], -1\n\t"
                    "zext.b %[lreg], %[lreg]\n\t"
                    "slli %[lreg], %[lreg], 12\n\t"
                    "add %[lreg], %[lreg], %[sbase]\n\t"
                    "li %[cmd_wr], 4\n\t"
                    "slli %[txq_st], %[txq_st], 15\n\t"
                    "sub %[comp], %[comp], %[proc]\n\t"
                    : [txq_st] "=&r"(txq_st),
                      [comp] "=&r"(comp_delta),
                      [cmd_wr] "=&r"(cmd_wr),
                      [sent] "+r"(sent),
                      [free] "+r"(free_rem),
                      [lreg] "+r"(lreg)
                    : [txq] "r"(txq_base), [cptr] "r"(cptr), [sbase] "r"(stream0_update_addr), [proc] "r"(proc)
                    : "memory");
                if (static_cast<int32_t>(txq_st) < 0) [[unlikely]] {
                    while (SenderLinkTransport::is_busy()) {
                    }
                }
                volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);
                txq_regs[ETH_TXQ_DEST_ADDR >> 2] = rem_stream_addr;
                txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
                txq_regs[ETH_TXQ_CMD >> 2] = cmd_wr;
                outbound_to_receiver_channel_pointers.num_free_slots = free_rem;
                if constexpr (fabric_exp_tx_lean_completions) {
                    sender_channel_from_receiver_credits.packets_sent = sent;
                }
                if constexpr (fabric_exp_tx_skip_idle_channels && exp_skip_idle_channel(sender_channel_index)) {
                    exp_fwd_in_flight++;
                }
                if constexpr (exp_tx_upstream_moved_off_this_erisc) {
                    increment_local_update_ptr_val<exp_upstream_in_flight_stream_id>(1);
                }
                if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                    update_bw_counters(pkt_header, local_fabric_telemetry);
                }
                *reinterpret_cast<volatile uint32_t*>(lreg) = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
                completions_since_last_check = static_cast<int32_t>(comp_delta);
            }
        } else {
            if constexpr (fabric_exp_tx_lean_completions) {
                sender_channel_from_receiver_credits.note_packet_sent();
            }
            if constexpr (fabric_exp_tx_skip_idle_channels && exp_skip_idle_channel(sender_channel_index)) {
                exp_fwd_in_flight++;
            }
            if constexpr (exp_tx_upstream_moved_off_this_erisc) {
                increment_local_update_ptr_val<exp_upstream_in_flight_stream_id>(1);
            }
            // Update local TX counters: split responsibility in multi-ERISC mode
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(pkt_header, local_fabric_telemetry);
            }
            if constexpr (fabric_exp_tx_send_ct_free_slot_stream) {
                increment_local_update_ptr_val<
                    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)]>(1);
            } else {
                increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
            }
        }
        phase_profiler.close<PHASE_TX_SEND>(send_start);
        phase_profiler.count_pkt();
        fabric_experiment_spin<fabric_exp_tx_delay>();
    } else if (has_unsent_packet) {
        phase_profiler.mark_blocked();
        phase_profiler.count_event(receiver_has_space_for_packet ? EVENT_TX_BLOCKED_TXQ : EVENT_TX_BLOCKED_CREDIT);
    }

    // Process COMPLETIONs from receiver
    if (!exp_overlap_send_completion || !can_send) {
        if (!fabric_exp_tx_lean_completions || sender_channel_from_receiver_credits.has_completions_pending()) {
            completions_since_last_check =
                sender_channel_from_receiver_credits
                    .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
    }
    if (completions_since_last_check) {
        const uint32_t completion_start = phase_profiler.open<PHASE_TX_COMPLETION>();
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_channel_from_receiver_credits.increment_num_processed_completions(completions_since_last_check);
        if constexpr (fabric_exp_tx_skip_idle_channels && exp_skip_idle_channel(sender_channel_index)) {
            exp_fwd_in_flight -= completions_since_last_check;
        }

        // When first level ack is enabled, then credits can be sent to upstream workers as soon as we see
        // the ack, we don't need to wait for the completion from receiver. Therefore, only when we have
        // first level ack disabled will we send credits to workers on receipt of completion acknowledgements.
        if constexpr (!enable_first_level_ack) {
            if constexpr (fabric_exp_tx_amortize_worker_credits) {
                exp_pending_upstream_credits[sender_channel_index] += completions_since_last_check;
            } else if constexpr (!exp_tx_upstream_moved_off_this_erisc) {  // else E1 returns the credits
                send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                    local_sender_channel_worker_interface,
                    completions_since_last_check,
                    channel_connection_established);
            }
        }
        phase_profiler.close<PHASE_TX_COMPLETION>(completion_start);
    }

    // Process ACKs from receiver
    // ACKs are processed second to avoid any sort of races. If we process acks second,
    // we are guaranteed to see equal to or greater the number of acks than completions
    if constexpr (enable_first_level_ack) {
        uint32_t acks_since_last_check = 0;
        if (!fabric_exp_tx_lean_completions || sender_channel_from_receiver_credits.has_acks_pending()) {
            acks_since_last_check = sender_channel_from_receiver_credits
                                        .template get_num_unprocessed_acks_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        if (acks_since_last_check > 0) {
            const uint32_t ack_start = phase_profiler.open<PHASE_TX_COMPLETION>();
            sender_channel_from_receiver_credits.increment_num_processed_acks(acks_since_last_check);
            if constexpr (fabric_exp_tx_amortize_worker_credits) {
                exp_pending_upstream_credits[sender_channel_index] += acks_since_last_check;
            } else {
                send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                    local_sender_channel_worker_interface, acks_since_last_check, channel_connection_established);
            }
            phase_profiler.close<PHASE_TX_COMPLETION>(ack_start);
        }
    }

    if constexpr (fabric_exp_tx_amortize_worker_credits) {
        constexpr int32_t credit_batch =
            WorkerInterfaceT::num_buffers >= 8
                ? 4
                : (WorkerInterfaceT::num_buffers >= 2 ? WorkerInterfaceT::num_buffers / 2 : 1);
        const int32_t pending = exp_pending_upstream_credits[sender_channel_index];
        if (pending >= credit_batch || (pending > 0 && !has_unsent_packet)) {
            const uint32_t credit_start = phase_profiler.open<PHASE_TX_COMPLETION>();
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, pending, channel_connection_established);
            exp_pending_upstream_credits[sender_channel_index] = 0;
            phase_profiler.close<PHASE_TX_COMPLETION>(credit_start);
        }
    }

    if constexpr (!SKIP_CONNECTION_LIVENESS_CHECK && !exp_tx_upstream_moved_off_this_erisc) {
        bool poll_connection = true;
        if constexpr (fabric_exp_tx_lazy_conn_check) {
            poll_connection = (exp_conn_poll_counter[sender_channel_index]++ & 0xF) == 0;
        }
        auto check_connection_status =
            poll_connection &&
            (!channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request());
        if (check_connection_status) {
            const uint32_t conn_start = phase_profiler.open<PHASE_TX_CONN>();
            check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                local_sender_channel_worker_interface,
                channel_connection_established,
                sender_channel_free_slots_stream_id);
            phase_profiler.close<PHASE_TX_CONN>(conn_start);
        }
    }
    return progress;
};

template <
    uint8_t VC_RECEIVER_CHANNEL,
    uint8_t sender_channel_index,
    bool enable_first_level_ack,
    typename EthSenderChannels,
    typename EdmChannelWorkerIFs,
    typename RemoteEthReceiverChannels,
    typename ReceiverPointersT,
    size_t NUM_SENDER_CHANNELS,
    typename LocalTelemetryT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_sender_channel_step(
        EthSenderChannels& local_sender_channels,
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        RemoteEthReceiverChannels& remote_receiver_channels,
        std::array<bool, NUM_SENDER_CHANNELS>& channel_connection_established,
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        SenderChannelFromReceiverCredits<NUM_SENDER_CHANNELS>& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t exp_grouped_free_slots_raw = 0) {
    if constexpr (is_sender_channel_serviced[sender_channel_index]) {
        const uint32_t visit_start = phase_profiler.open<PHASE_TX_VISIT>();
        // the cache is invalidated here because the channel will read some
        // L1 locations to see if it can make progress
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

        const bool progress = run_sender_channel_step_impl<
            sender_channel_index,
            to_receiver_packets_sent_streams[VC_RECEIVER_CHANNEL],
            sender_ch_live_check_skip[sender_channel_index],
            enable_first_level_ack,
            VC_RECEIVER_CHANNEL>(
            local_sender_channels.template get<sender_channel_index>(),
            local_sender_channel_worker_interfaces.template get<sender_channel_index>(),
            outbound_to_receiver_channel_pointers,
            remote_receiver_channels.template get<VC_RECEIVER_CHANNEL>(),
            channel_connection_established[sender_channel_index],
            local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)],
            sender_channel_from_receiver_credits.template get<sender_channel_index>(),
            perf_telemetry_recorder,
            local_fabric_telemetry,
            exp_grouped_free_slots_raw);
        phase_profiler.close<PHASE_TX_VISIT>(visit_start);
        return progress;
    }
    return false;
}

// Experiment (EXP_RX_GROUPED_READS): the receiver step's register reads issued back to back so their latencies
// overlap (RESULTS §17: up to four register reads in flight cost ~4-8 cycles, against ~7-8 each used at once). The
// TX queue's command register is read and discarded before its status, as eth_txq_is_busy does (BH-55).
template <uint32_t pkts_sent_stream_id>
FORCE_INLINE void exp_read_rx_group(
    uint32_t write_sent_reg_addr, uint32_t& pkts_raw, uint32_t& txq_status_raw, uint32_t& write_sent_raw) {
    constexpr uint32_t pkts_addr =
        STREAM_REG_ADDR(pkts_sent_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    constexpr uint32_t txq_cmd_addr = ETH_TXQ0_REGS_START + receiver_txq_id * ETH_TXQ_REGS_SIZE + ETH_TXQ_CMD;
    constexpr uint32_t txq_status_addr = ETH_TXQ0_REGS_START + receiver_txq_id * ETH_TXQ_REGS_SIZE + ETH_TXQ_STATUS;
    asm volatile("lw %0, 0(%3)\n lw %1, 0(%4)\n lw %1, 0(%5)\n lw %2, 0(%6)\n"
                 : "=&r"(pkts_raw), "=&r"(txq_status_raw), "=&r"(write_sent_raw)
                 : "r"(pkts_addr), "r"(txq_cmd_addr), "r"(txq_status_addr), "r"(write_sent_reg_addr)
                 : "memory");
}

// EXP_RX_GROUPED_NO_PREFETCH: the same group without the packets-received count.
FORCE_INLINE void exp_read_rx_completion_group(
    uint32_t write_sent_reg_addr, uint32_t& txq_status_raw, uint32_t& write_sent_raw) {
    constexpr uint32_t txq_cmd_addr = ETH_TXQ0_REGS_START + receiver_txq_id * ETH_TXQ_REGS_SIZE + ETH_TXQ_CMD;
    constexpr uint32_t txq_status_addr = ETH_TXQ0_REGS_START + receiver_txq_id * ETH_TXQ_REGS_SIZE + ETH_TXQ_STATUS;
    asm volatile("lw %0, 0(%2)\n lw %0, 0(%3)\n lw %1, 0(%4)\n"
                 : "=&r"(txq_status_raw), "=&r"(write_sent_raw)
                 : "r"(txq_cmd_addr), "r"(txq_status_addr), "r"(write_sent_reg_addr)
                 : "memory");
}

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT,
    typename ReceiverChannelResponseCreditSenderT>
FORCE_INLINE bool run_receiver_channel_step_impl(
    ReceiverChannelBufferT& local_receiver_channel,
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    [[maybe_unused]] LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    [[maybe_unused]] std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSenderT& receiver_channel_response_credit_sender,
    [[maybe_unused]] const tt::tt_fabric::routing_l1_info_t& routing_table,
    [[maybe_unused]] uint16_t my_mesh_id,
    [[maybe_unused]] uint8_t my_mesh_coord_y,
    [[maybe_unused]] uint8_t my_mesh_coord_x,
    LocalTelemetryT& local_fabric_telemetry) {
    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    // EXP_RX_GROUPED_READS: the packets-received count was read at the end of the previous step, together with
    // that step's completion reads (see the completion section below).
    constexpr bool exp_grouped_reads =
        fabric_exp_rx_grouped_reads && !enable_first_level_ack && WriteTridTracker::SINGLE_NOC;
    int32_t pkts_received_since_last_check;
    if constexpr (exp_grouped_reads && fabric_exp_rx_grouped_prefetch) {
        pkts_received_since_last_check = receiver_channel_pointers.exp_prefetched_pkts_received;
    } else {
        pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();
    }

    bool unwritten_packets;
    if constexpr (enable_first_level_ack) {
        auto& ack_counter = receiver_channel_pointers.ack_counter;
        bool pkts_received = pkts_received_since_last_check > 0;
        bool can_send_ack = pkts_received;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_ack = can_send_ack && !ReceiverLinkTransport::is_busy();
        }
        if (can_send_ack) {
            // currently only support processing one packet at a time, so we only decrement by 1
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);

            uint8_t src_ch_id;
            if constexpr (skip_src_ch_id_update) {
                // skip_src_ch_id_update implies something like mux mode is disabled and there is only a single
                // sender channel so we don't dynamically fetch it off the packet header
                src_ch_id = receiver_channel_pointers.get_src_chan_id();
            } else {
                auto receiver_buffer_index = ack_counter.get_buffer_index();
                tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
                    local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
                src_ch_id = receiver_channel_pointers.get_src_chan_id(receiver_buffer_index);
            }

            const uint32_t ack_start = phase_profiler.open<PHASE_RX_ACK>();
            receiver_send_received_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            ack_counter.increment();
            phase_profiler.close<PHASE_RX_ACK>(ack_start);
        }
        unwritten_packets = !wr_sent_counter.is_caught_up_to(ack_counter);

    } else {
        unwritten_packets = pkts_received_since_last_check != 0;
    }

    // Code profiling timer for receiver channel forward
    NamedProfiler<CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD, code_profiling_enabled_timers_bitfield, code_profiling_buffer_base_addr> receiver_forward_timer;
    receiver_forward_timer.set_should_dump(unwritten_packets);
    receiver_forward_timer.open();

    if (unwritten_packets) {
        const uint32_t rx_start = phase_profiler.open<PHASE_RX_DECODE_ADMIT>();
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

#if defined(FABRIC_DRISC_OFFLOAD)
        bool can_push = get_ptr_val<drisc_free_slots_stream_id>() != 0;
        const bool admitted = can_push;
        if constexpr (enable_trid_flush_check_on_noc_txn) {
            can_push = can_push && receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        }
        if (can_push) {
            const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);
            if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
            }
            did_something = true;
            progress = true;
            phase_profiler.mark_progress();
            phase_profiler.close<PHASE_RX_DECODE_ADMIT>(rx_start);
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);
            const uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                receiver_buffer_index);
            const uint32_t local_start = phase_profiler.open<PHASE_RX_LOCAL>();
            drisc_offload_push_packet(
                reinterpret_cast<uint32_t>(packet_header),
                packed.payload_size_bytes + sizeof(PACKET_HEADER_TYPE),
                trid);
            phase_profiler.close<PHASE_RX_LOCAL>(local_start);
            increment_local_update_ptr_val<drisc_free_slots_stream_id>(-1);
            wr_sent_counter.increment();
            if constexpr (!enable_first_level_ack) {
                increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            }
            phase_profiler.count_pkt();
            fabric_experiment_spin<fabric_exp_rx_delay>();
        } else {
            phase_profiler.mark_blocked();
            phase_profiler.count_event(admitted ? EVENT_RX_BLOCKED_TRID : EVENT_RX_BLOCKED_ADMIT);
        }
#else
        {
            ROUTING_FIELDS_TYPE cached_routing_fields{};
#ifndef FABRIC_2D
            cached_routing_fields = packet_header->routing_fields;
#endif
            if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
            }
            // Decode action is the dispatch word: LIVE slots test action_bit(dirs[slot]). Local
            // delivery stays a separate flag (bit 5).
            uint8_t action = 0;
            bool local_deliver = false;
            // Set only when this chip is the mesh's exit for the packet; carries the resolved INTERMESH
            // downstream slot from admission to the forward phase.
            bool intermesh_egress = false;
            uint8_t intermesh_egress_index = 0;
            bool can_send_to_all_local_chip_receivers;
            if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                // need the FABRIC_2D ifdef since the packet header for 1D does not have route_buffer field in it.
                {
                    // 2D action-map RX path: landing intercept (boundary receivers only) -> decode ->
                    // intermesh-exit intercept -> linear LIVE admit on action bits. No hop program
                    // and no header mutation. Self is not in fwd_dirs so no arm tests it; extra bits
                    // outside LIVE are ignored. boundary_dir_is_addressable is the only remaining
                    // bounds check, because a bad intermesh exit index walks off the compact array.
                    if constexpr (receiver_channel_is_intermesh_ingress[receiver_channel]) {
                        // Runs before decode, which would otherwise consume stale source-mesh action maps. The
                        // landing encode replaces them from this mesh's destination-major route table.
                        fabric_set_2d_intermesh_landing_route(packet_header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
                        // Rebuilt map requires Y action decode first.
                        action = Routing2DCodec::decode_action_y_first(
                            packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
                    } else {
                        action = Routing2DCodec::decode_action<static_cast<eth_chan_directions>(my_direction)>(
                            packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
                    }
                    // This chip is the exit when the maps say deliver here but the final mesh is
                    // elsewhere. CT-gated so interior routers skip the mesh-id compare.
                    bool intermesh_exit = false;
                    if constexpr (is_intramesh_router_on_edge) {
                        intermesh_exit = Routing2DCodec::action_is_intermesh_exit(
                            action, packet_header->dst_start_mesh_id, my_mesh_id);
                    }
                    if (intermesh_exit) {
                        // Forward the packet as-is on the INTERMESH egress in boundary_dir, with no
                        // local delivery and no map rebuild.
                        const auto boundary_dir = static_cast<eth_chan_directions>(
                            routing_table.inter_mesh_direction_table.get_original_direction(
                                packet_header->dst_start_mesh_id));
                        // boundary_dir is chosen per destination mesh, so unlike a decoded action it is
                        // not constrained to the arms this router was built with. Indexing the compact
                        // array unchecked would read past its end, so fail-stop instead.
                        const bool boundary_dir_is_addressable =
                            boundary_dir < eth_chan_directions::COUNT &&
                            boundary_dir != static_cast<eth_chan_directions>(my_direction) &&
                            get_downstream_edm_interface_index(boundary_dir) < downstream_edm_interfaces.size();
                        if (!boundary_dir_is_addressable) {
                            ASSERT(false);
                            can_send_to_all_local_chip_receivers = false;
                        } else {
                            intermesh_egress = true;
                            intermesh_egress_index =
                                static_cast<uint8_t>(get_downstream_edm_interface_index(boundary_dir));
                            can_send_to_all_local_chip_receivers =
                                downstream_edm_interfaces[intermesh_egress_index]
                                    .template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
                        }
                    } else {
                        local_deliver = (action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
                        can_send_to_all_local_chip_receivers =
                            admit_2d_dispatch(action, local_deliver, downstream_edm_interfaces, local_relay_interface);
                    }
                }
#endif
            } else {
#ifndef FABRIC_2D
                can_send_to_all_local_chip_receivers =
                    can_forward_packet_completely(cached_routing_fields, downstream_edm_interfaces[0]);
#endif
            }
            const bool admitted = can_send_to_all_local_chip_receivers;
            if constexpr (enable_trid_flush_check_on_noc_txn) {
                bool trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
                can_send_to_all_local_chip_receivers &= trid_flushed;
            }
            if (can_send_to_all_local_chip_receivers) {
                did_something = true;
                progress = true;
                phase_profiler.mark_progress();
                phase_profiler.close<PHASE_RX_DECODE_ADMIT>(rx_start);
                // Count RX bytes/packets (header + payload) when consuming a packet from receiver buffer
                if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                    update_bw_counters(packet_header, local_fabric_telemetry);
                }
                channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);
                if constexpr (!is_receiver_channel_forwarding_disabled[receiver_channel]) {
                    uint8_t trid =
                        receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                            receiver_buffer_index);
                    if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                        {
                            if (intermesh_egress) {
                                // The exit chip's only output is the INTERMESH egress, sent as-is.
                                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                                    packet_header,
                                    packet_header->payload_size_bytes,
                                    cached_routing_fields,
                                    downstream_edm_interfaces[intermesh_egress_index],
                                    trid);
                                record_2d_forwarded_slots<receiver_channel>(1u << intermesh_egress_index);
                            } else {
                                // Same action byte as admission. Local delivery stays outside the scan.
                                const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                                const uint8_t forwarded_slot_mask = forward_2d_dispatch(
                                    action, packet_header, cached_routing_fields, downstream_edm_interfaces, trid);
                                record_2d_forwarded_slots<receiver_channel>(forwarded_slot_mask);
                                if (forwarded_slot_mask != 0) {
                                    phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                                }
                                if (local_deliver) {
                                    forward_to_local_destination<receiver_channel>(
                                        local_relay_interface, packet_header, packet_header->payload_size_bytes, trid);
                                }
                            }
                        }
#endif
                    } else {
#ifndef FABRIC_2D
                        receiver_forward_packet<receiver_channel>(
                            packet_header, cached_routing_fields, downstream_edm_interfaces[0], trid);
#endif
                    }
                }
                wr_sent_counter.increment();
                // decrement the to_receiver_pkts_sent_id stream register by 1 since current packet has been processed.
                if constexpr (!enable_first_level_ack) {
                    increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
                }
                phase_profiler.count_pkt();
                fabric_experiment_spin<fabric_exp_rx_delay>();
            } else {
                phase_profiler.mark_blocked();
                phase_profiler.count_event(admitted ? EVENT_RX_BLOCKED_TRID : EVENT_RX_BLOCKED_ADMIT);
            }
        }
#endif
    }

    // Close the code profiling timer
    receiver_forward_timer.close();

    if constexpr (!fuse_receiver_flush_and_completion_ptr) {
        auto& wr_flush_counter = receiver_channel_pointers.wr_flush_counter;
        bool unflushed_writes = !wr_flush_counter.is_caught_up_to(wr_sent_counter);
        if (unflushed_writes) {
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
            if (next_trid_flushed) {
                wr_flush_counter.increment();
                receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
            }
        }

        auto& completion_counter = receiver_channel_pointers.completion_counter;
        bool unsent_completions = !completion_counter.is_caught_up_to(completion_counter, wr_flush_counter);
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            unsent_completions = unsent_completions && !ReceiverLinkTransport::is_busy();
        }
        if (unsent_completions) {
            const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
            // completion ptr incremented in callee
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender,
                receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
            completion_counter.increment();
            phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
        }
    } else if constexpr (fabric_exp_rx_batch_completion && !enable_first_level_ack) {
        constexpr uint32_t batch_target = RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel] >= 8 ? 4 : 1;
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        const uint32_t uncompleted = completion_counter.distance_behind(wr_sent_counter);
#if defined(FABRIC_DRISC_OFFLOAD)
        // With DRISC offload, E1's push takes ~35 cycles while packets arrive over Ethernet every ~130 cycles, so
        // progress is false on 2-3 polls between every arriving packet. Flush a full batch of 4 as soon as the 4th
        // slot's write has left E1 L1, and only flush a partial 1..3 tail once the DRISC ring has completely drained.
        const bool should_check_completion =
            uncompleted > batch_target ||
            (!progress &&
             (uncompleted >= batch_target || (uncompleted != 0 && get_ptr_val<drisc_free_slots_stream_id>() ==
                                                                      static_cast<int32_t>(drisc_ring_slots))));
#else
        // While packets keep arriving (progress == true), wait until a 5th slot has been issued so the first 4 slots
        // have already drained from cmd_buf 0; when no packet arrived in this step (!progress), flush whatever slots
        // are pending (1..4) as soon as they finish.
        const bool should_check_completion = uncompleted > batch_target || (uncompleted != 0 && !progress);
#endif
        if (should_check_completion) {
            const uint32_t batch = uncompleted > batch_target ? batch_target : uncompleted;
            const auto base_idx = completion_counter.get_buffer_index();
            const auto last_idx = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment_n<RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel]>(
                    base_idx.get(), batch - 1)};
            bool can_send_completion = receiver_channel_trid_tracker.transaction_flushed(last_idx);
#if !defined(FABRIC_DRISC_OFFLOAD)
            if (!progress && batch > 1) {
                can_send_completion =
                    can_send_completion && receiver_channel_trid_tracker.transaction_flushed(base_idx);
            }
#endif
            if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
                can_send_completion = can_send_completion && !ReceiverLinkTransport::is_busy();
            }
            if (can_send_completion) {
                const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
                if constexpr (skip_src_ch_id_update) {
                    receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                        receiver_channel_response_credit_sender, receiver_channel_pointers.get_src_chan_id(), batch);
                } else {
                    auto idx = base_idx;
                    for (uint32_t k = 0; k < batch; k++) {
                        receiver_channel_response_credit_sender.accumulate_completion_credit(
                            receiver_channel_pointers.get_src_chan_id(idx), 1);
                        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(idx);
                        idx = tt::tt_fabric::BufferIndex{
                            tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel]>(idx.get())};
                    }
                    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
                        while (ReceiverLinkTransport::is_busy()) {
                        }
                    }
                    receiver_channel_response_credit_sender.flush_completion_credits();
                }
                completion_counter.increment_n(batch);
                phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
            }
        }
    } else {
        // flush and completion are fused, so we only need to update one of the counters
        // update completion since other parts of the code check against completion
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        // Currently unclear if it's better to loop here or not...
        bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
        auto receiver_buffer_index = completion_counter.get_buffer_index();
        bool next_trid_flushed;
        bool exp_txq_busy = false;
        if constexpr (exp_grouped_reads) {
            // Read this slot's write-sent status and the TX queue's busy check where today's code reads them, together
            // with the packets-received count for the next step, so the four reads overlap.
            uint32_t txq_status_raw, write_sent_raw;
            if constexpr (fabric_exp_rx_grouped_prefetch) {
                uint32_t pkts_raw;
                exp_read_rx_group<to_receiver_pkts_sent_id>(
                    receiver_channel_trid_tracker.transaction_sent_reg_addr(receiver_buffer_index),
                    pkts_raw,
                    txq_status_raw,
                    write_sent_raw);
                receiver_channel_pointers.exp_prefetched_pkts_received =
                    pkts_raw & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
            } else {
                exp_read_rx_completion_group(
                    receiver_channel_trid_tracker.transaction_sent_reg_addr(receiver_buffer_index),
                    txq_status_raw,
                    write_sent_raw);
            }
            next_trid_flushed = write_sent_raw == 0;
            exp_txq_busy = ((txq_status_raw >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 0x1) != 0;
        } else {
            next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        }
        bool can_send_completion = unflushed_writes && next_trid_flushed;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            if constexpr (exp_grouped_reads) {
                can_send_completion = can_send_completion && !exp_txq_busy;
            } else {
                can_send_completion = can_send_completion && !ReceiverLinkTransport::is_busy();
            }
        }
        if (can_send_completion) {
            const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
            uint8_t src_ch_id;
            if constexpr (skip_src_ch_id_update) {
                src_ch_id = receiver_channel_pointers.get_src_chan_id();
            } else {
                src_ch_id = receiver_channel_pointers.get_src_chan_id(receiver_buffer_index);
            }
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
            completion_counter.increment();
            phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
        }
    }
    return progress;
};

template <
    uint8_t receiver_channel,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename EthReceiverChannels,
    typename WriteTridTracker,
    typename ReceiverChannelPointersT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step(
    EthReceiverChannels& local_receiver_channels,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSenders<NUM_RECEIVER_CHANNELS>& receiver_channel_response_credit_senders,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    uint16_t my_mesh_id,
    uint8_t my_mesh_coord_y,
    uint8_t my_mesh_coord_x,
    LocalTelemetryT& local_fabric_telemetry) {
    if constexpr (is_receiver_channel_serviced[receiver_channel]) {
        const uint32_t visit_start = phase_profiler.open<PHASE_RX_VISIT>();
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        const bool progress = run_receiver_channel_step_impl<
            receiver_channel,
            to_receiver_packets_sent_streams[receiver_channel],
            enable_first_level_ack,
            DOWNSTREAM_EDM_SIZE,  // Pass size explicitly
            WriteTridTracker,
            decltype(local_receiver_channels.template get<receiver_channel>()),
            ReceiverChannelPointersT,
            DownstreamSenderT,
            LocalRelayInterfaceT>(
            local_receiver_channels.template get<receiver_channel>(),
            downstream_edm_interfaces,
            local_relay_interface,
            receiver_channel_pointers,
            receiver_channel_trid_tracker,
            port_direction_table,
            receiver_channel_response_credit_senders.template get<receiver_channel>(),
            routing_table,
            my_mesh_id,
            my_mesh_coord_y,
            my_mesh_coord_x,
            local_fabric_telemetry);
        phase_profiler.close<PHASE_RX_VISIT>(visit_start);
        return progress;
    }
    return false;
}

template <typename OutboundReceiverChannelPointers, typename RemoteEthReceiverChannels>
FORCE_INLINE void configure_outbound_to_receiver_channel_pointers(
    OutboundReceiverChannelPointers& outbound_to_receiver_channel_pointers,
    RemoteEthReceiverChannels& remote_receiver_channels) {
    static_assert(OutboundReceiverChannelPointers::N == RemoteEthReceiverChannels::num_channels);

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((outbound_to_receiver_channel_pointers.template get<Is>().init(
            reinterpret_cast<uint32_t>(remote_receiver_channels.template get<Is>().channel_base_address()),
            remote_receiver_channels.template get<Is>().get_max_eth_payload_size())), ...);
    }(std::make_index_sequence<OutboundReceiverChannelPointers::N>{});
}

// Input State(s): PAUSED
// Output State(s): PAUSED
void run_retrain_step(tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    // Placeholder
    state_manager_l1->state = RouterState::RETRAINING;
    run_routing_without_noc_sync();
    while (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::RETRAIN && !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        // Wait for confirmation from host to avoid the WAW hazard:
        // Host issues retrain
        // Router enters retrain state
        // Router completes retrain
        // Router goes back to pause state, sets current state to PAUSED
        // Host status read lands - it doesn't know if the RETRAIN landed and is done or was raced ahead
        //
        // For the time being, the implementation is conservative
        //
        // This can be avoided if the host guarantees that the reads are queued strictly behind writes
        // For the entirety of the H->D datapath. For the time being, this is too strong of a requirement
        // for cases where router code is not running in traditional H->D server/PC configurations
        // (i.e. some custom IP integrations may not easily satisfy this guarantee this)
    }

    // PAUSED is the only legal output state
    ASSERT(reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::PAUSE);
};

// Input State(s): PAUSED
// Output State(s): DRAINING
void run_drain_step(tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    state_manager_l1->state = RouterState::DRAINING;
    ASSERT(false); // not implemented
    while (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::DRAIN && !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
    }

    ASSERT(reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::PAUSE);
};

__attribute__((noinline))
void execute_pause_command(tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (MY_ERISC_ID == 0) {
        // Master erisc will handle the pause command but we need to make sure the other erisc is in its wait loop
        // before we proceed. This coordination is not implemented yet. When mainlined, it will be integrated here.

        bool keep_running_pause = true;
        while (keep_running_pause && !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            state_manager_l1->state = RouterState::PAUSED;
            switch (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
                case RouterCommand::RUN:
                    keep_running_pause = false;
                    break;
                case RouterCommand::PAUSE:
                    // nothing to do, we are paused, we may as well context switch to base FW
                    run_routing_without_noc_sync();
                    break;
                case RouterCommand::DRAIN:
                    run_drain_step(state_manager_l1, termination_signal_ptr);
                    break;
                case RouterCommand::RETRAIN:
                    run_retrain_step(state_manager_l1, termination_signal_ptr);
                    break;
                default:
                    ASSERT(false); // illegal state transition
            }
        }

        //
        // Likewise, we should resume other eriscs here
        //
    }
};

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE bool continue_running_main_run_loop(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr, tt_l1_ptr RouterStateManager* state_manager_l1) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        if constexpr (!IS_RETRAIN_SYNC_MASTER()) {
            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr);
        } else {
            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) &&
                   !state_manager_l1->is_non_run_command_pending /*<ENABLE_RISC_CPU_DATA_CACHE>*/ ();
        }
    } else {
        return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) && !state_manager_l1->is_non_run_command_pending();
    }
}

// Include the speedy path functions (must come after all helper function definitions above)
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp"

/*
 * Main control loop for fabric EDM. Run indefinitely until a termination signal is received
 *
 * Every loop iteration visit a sender channel and the receiver channel. Switch between sender
 * channels every iteration unless it is unsafe/undesirable to do so (e.g. for performance reasons).
 */
template <
    size_t NUM_RECEIVER_CHANNELS,
    typename DownstreamSenderVC0T,
    typename DownstreamSenderVC1T,
    typename LocalRelayInterfaceT,
    typename EthSenderChannels,
    typename EthReceiverChannels,
    typename RemoteEthReceiverChannels,
    typename EdmChannelWorkerIFs,
    typename TransactionIdTrackerCH0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    typename TransactionIdTrackerCH1
#endif  // FABRIC_2D_VC1_ACTIVE
#if defined(FABRIC_2D_VC2_SERVICED)
    ,
    typename TransactionIdTrackerCH2
#endif  // FABRIC_2D_VC2_SERVICED
    >
FORCE_INLINE void run_fabric_edm_main_loop(
    EthReceiverChannels& local_receiver_channels,
    EthSenderChannels& local_sender_channels,
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    std::array<DownstreamSenderVC0T, VC0_DOWNSTREAM_EDM_SIZE>& downstream_edm_noc_interfaces_vc0,
    std::array<DownstreamSenderVC1T, VC1_DOWNSTREAM_EDM_SIZE>& downstream_edm_noc_interfaces_vc1,
    LocalRelayInterfaceT& local_relay_interface,
    RemoteEthReceiverChannels& remote_receiver_channels,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    TransactionIdTrackerCH0& receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
    TransactionIdTrackerCH1& receiver_channel_1_trid_tracker,
#endif  // FABRIC_2D_VC1_ACTIVE
#if defined(FABRIC_2D_VC2_SERVICED)
    TransactionIdTrackerCH2& receiver_channel_2_trid_tracker,
#endif  // FABRIC_2D_VC2_SERVICED
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    size_t did_nothing_count = 0;
    using FabricTelemetryT = FabricTelemetry;
    FabricTelemetryT local_fabric_telemetry{};
    auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetryT*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);

    // Read the routing table in place
    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);

    // Keep hot routing coordinates in address-untaken scalars so they remain in registers across the
    // main loop; the RISC data cache is disabled for this build.
    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;

    // Copy the mesh id as well: routing_table is passed by reference and invalidated during receiver
    // steps, while this scalar can remain in a register for the intermesh-exit comparison.
    const uint16_t my_mesh_id = routing_table.my_mesh_id;

    // May want to promote to part of the handshake but for now we just initialize in this standalone way
    // TODO: flatten all of these arrays into a single object (one array lookup) OR
    //       (probably better) pack most of these into single words (e.g. we could hold a read, write, and ackptr in a
    //       single word) this way - especially if power of 2 wraps, we can handle both channels literally at once with
    //       math ops on single individual words (or half words)
    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, REMOTE_RECEIVER_NUM_BUFFERS_ARRAY>::make();

    configure_outbound_to_receiver_channel_pointers(outbound_to_receiver_channel_pointers, remote_receiver_channels);

    // Workaround the perf regression in RingAsLinear test.
    auto outbound_to_receiver_channel_pointer_ch0 =
        outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers = ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    // Workaround the perf regression in RingAsLinear test.
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

#if defined(FABRIC_2D_VC1_SERVICED)
    // VC1 receiver channel pointer for inter-mesh routing. Gated on _SERVICED (not _ACTIVE) because
    // these locals are only consumed inside the FABRIC_2D_VC1_SERVICED block further down; on routers
    // that have VC1 channels (ACTIVE) but do not service them (e.g. inter-mesh routers without
    // pass-through), defining them here trips -Wunused-but-set-variable in the kernel JIT compile.
    auto outbound_to_receiver_channel_pointer_ch1 =
        outbound_to_receiver_channel_pointers.template get<VC1_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch1 = receiver_channel_pointers.template get<1>();
    receiver_channel_pointers_ch1.reset();
#endif  // FABRIC_2D_VC1_SERVICED

#if defined(FABRIC_2D_VC2_SERVICED)
    auto outbound_to_receiver_channel_pointer_ch2 =
        outbound_to_receiver_channel_pointers.template get<VC2_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch2 = receiver_channel_pointers.template get<VC2_RECEIVER_CHANNEL>();
    receiver_channel_pointers_ch2.reset();
#endif  // FABRIC_2D_VC2_SERVICED

    if constexpr (skip_src_ch_id_update) {
        receiver_channel_pointers_ch0.set_src_chan_id(BufferIndex{0}, remote_worker_sender_channel);
    }

    std::array<bool, NUM_SENDER_CHANNELS> channel_connection_established =
        initialize_array<NUM_SENDER_CHANNELS, bool, false>();

    PerfTelemetryRecorder inner_loop_perf_telemetry_collector = build_perf_telemetry_recorder<perf_telemetry_mode>();
    auto local_perf_telemetry_buffer =
        build_perf_telemetry_buffer(reinterpret_cast<uint32_t*>(perf_telemetry_buffer_addr));

    auto receiver_channel_response_credit_senders =
        init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();
    auto sender_channel_from_receiver_credits =
        init_sender_channel_from_receiver_credits_flow_controllers<NUM_SENDER_CHANNELS>();

    // This value defines the number of loop iterations we perform of the main control sequence before exiting
    // to check for termination and context switch. Removing the these checks from the inner loop can drastically
    // improve performance. The value of 32 was chosen somewhat empirically and then raised up slightly.
    static ActualSpeedySenderState<super_speedy_mode> persistent_speedy_sender_state;
    static ActualSpeedyReceiverState<super_speedy_mode, 0> persistent_speedy_receiver_state;
#if defined(FABRIC_2D_VC2_SERVICED)
    static ActualSpeedySenderState<true> persistent_speedy_sender_state_vc2;
    static ActualSpeedyReceiverState<true, 2> persistent_speedy_receiver_state_vc2;
#endif

    phase_profiler.init<count_serviced(is_sender_channel_serviced), count_serviced(is_receiver_channel_serviced)>();
    exp_push_drisc_next_slot = exp_push_drisc_l1_base;
    exp_push_set_state();
    exp_pending_upstream_credits.fill(0);
    exp_conn_poll_counter.fill(0);
    exp_fwd_in_flight = 0;

    uint16_t fabric_heartbeat_counter = 0;
#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
#else
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x1F80;
#endif
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);

    auto execute_main_loop = [&]() __attribute__((always_inline)) {
        ActualSpeedySenderState<super_speedy_mode> local_speedy_sender_state;
        ActualSpeedyReceiverState<super_speedy_mode, 0> local_speedy_receiver_state;
#if defined(FABRIC_2D_VC2_SERVICED)
        ActualSpeedySenderState<true> local_speedy_sender_state_vc2;
        ActualSpeedyReceiverState<true, 2> local_speedy_receiver_state_vc2;
#endif
        speedy_state_copy_in<super_speedy_mode, 0>(
            local_speedy_sender_state,
            local_speedy_receiver_state,
            persistent_speedy_sender_state,
            persistent_speedy_receiver_state);
#if defined(FABRIC_2D_VC2_SERVICED)
        speedy_state_copy_in<true, 2>(
            local_speedy_sender_state_vc2,
            local_speedy_receiver_state_vc2,
            persistent_speedy_sender_state_vc2,
            persistent_speedy_receiver_state_vc2);
#endif
        while (continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr, state_manager_l1)) {
            did_something = false;

            uint32_t tx_progress = 0;
            uint32_t rx_progress = 0;

            if constexpr (any_sender_channel_serviced) {
                open_perf_recording_window(inner_loop_perf_telemetry_collector);
            }

            uint64_t loop_start_cycles;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                loop_start_cycles = get_timestamp();
            }
#if defined(FABRIC_2D_VC2_SERVICED)
            if constexpr (is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START]) {
                auto check_connection_status =
                    !channel_connection_established[VC2_LOCAL_CHANNEL_START] ||
                    local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>()
                        .has_worker_teardown_request();
                if (check_connection_status) {
                    check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                        local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>(),
                        channel_connection_established[VC2_LOCAL_CHANNEL_START],
                        local_sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START]);
                }
            }
#endif
            for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
                if constexpr (super_speedy_mode) {
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

                    if constexpr (is_sender_channel_serviced[0]) {
                        const uint32_t tx_visit_start = phase_profiler.open<PHASE_TX_VISIT>();
                        tx_progress |= run_sender_channel_step_speedy<
                            0,
                            to_receiver_packets_sent_streams[VC0_RECEIVER_CHANNEL],
                            SENDER_CREDIT_AMORTIZATION_FREQUENCY,
                            true /*MANAGE_CONNECTION_LIVENESS_IN_SPEEDY_HELPER*/>(
                            local_sender_channels.template get<0>(),
                            local_sender_channel_worker_interfaces.template get<0>(),
                            outbound_to_receiver_channel_pointer_ch0,
                            remote_receiver_channels.template get<VC0_RECEIVER_CHANNEL>(),
                            channel_connection_established[0],
                            local_sender_channel_free_slots_stream_ids[0],
                            sender_channel_from_receiver_credits.template get<0>(),
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            local_speedy_sender_state);
                        phase_profiler.close<PHASE_TX_VISIT>(tx_visit_start);
                    }

                    if constexpr (is_receiver_channel_serviced[0]) {
#if defined(FABRIC_DRISC_OFFLOAD)
                        rx_progress |= run_receiver_channel_step<
                            0,
                            ENABLE_FIRST_LEVEL_ACK_VC0,
                            VC0_DOWNSTREAM_EDM_SIZE,
                            DownstreamSenderVC0T,
                            decltype(local_relay_interface)>(
                            local_receiver_channels,
                            downstream_edm_noc_interfaces_vc0,
                            local_relay_interface,
                            receiver_channel_pointers_ch0,
                            receiver_channel_0_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders,
                            routing_table,
                            my_mesh_id,
                            my_mesh_coord_y,
                            my_mesh_coord_x,
                            local_fabric_telemetry);
#else
                        const uint32_t rx_visit_start = phase_profiler.open<PHASE_RX_VISIT>();
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
                        rx_progress |= run_receiver_channel_step_speedy<
                            0,
                            0,  // producer sender channel index
                            to_receiver_packets_sent_streams[0],
                            RECEIVER_CREDIT_AMORTIZATION_FREQUENCY,
                            decltype(receiver_channel_0_trid_tracker)>(
                            local_receiver_channels.template get<0>(),
                            local_relay_interface,
                            receiver_channel_pointers_ch0,
                            receiver_channel_0_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders.template get<0>(),
                            routing_table,
                            local_fabric_telemetry,
                            local_speedy_receiver_state);
#else
                        rx_progress |= run_receiver_channel_step_speedy<
                            0,
                            0,  // producer sender channel index
                            to_receiver_packets_sent_streams[0],
                            RECEIVER_CREDIT_AMORTIZATION_FREQUENCY,
                            decltype(receiver_channel_0_trid_tracker)>(
                            local_receiver_channels.template get<0>(),
                            local_relay_interface,
                            receiver_channel_pointers_ch0,
                            receiver_channel_0_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders.template get<0>(),
                            routing_table,
                            local_fabric_telemetry,
                            local_speedy_receiver_state);
#endif
                        phase_profiler.close<PHASE_RX_VISIT>(rx_visit_start);
#endif
                    }
                } else {  // non-speedy mode
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
                    uint32_t exp_free_slots_raw[4] = {0, 0, 0, 0};
                    if constexpr (fabric_exp_tx_grouped_packet_check && any_sender_channel_serviced) {
                        exp_read_grouped_free_slots(exp_free_slots_raw);
                    }
                    if constexpr (exp_tx_upstream_on_this_erisc) {
                        // Packets in flight that are not credited upstream yet, or a connection-check iteration.
                        const bool check_connections = (exp_upstream_iteration++ & 15) == 0;
                        if (get_ptr_val<exp_upstream_in_flight_stream_id>() != 0 || check_connections) {
                            exp_run_upstream_steps(
                                local_sender_channel_worker_interfaces,
                                channel_connection_established,
                                sender_channel_from_receiver_credits,
                                local_sender_channel_free_slots_stream_ids,
                                check_connections);
                        }
                    }
                    // Capture these to see if we made progress

                    // There are some cases, mainly for performance, where we don't want to switch between sender
                    // channels so we interoduce this to provide finer grain control over when we disable the automatic
                    // switching
                    tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 0, ENABLE_FIRST_LEVEL_ACK_VC0>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch0,
                        remote_receiver_channels,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry,
                        exp_free_slots_raw[0]);
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
                    // Inter-mesh routers receive neighbor mesh's locally generated traffic on VC0.
                    // This VC0 traffic needs to be forwarded over VC1 in the receiving mesh.
                    rx_progress |= run_receiver_channel_step<
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0,
                        VC1_DOWNSTREAM_EDM_SIZE,  // Explicit size for VC1 downstream interfaces
                        DownstreamSenderVC1T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc1,
                        local_relay_interface,
                        receiver_channel_pointers_ch0,
                        receiver_channel_0_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry);
#else
                    rx_progress |= run_receiver_channel_step<
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0,
                        VC0_DOWNSTREAM_EDM_SIZE,  // Explicit size for VC0 downstream interfaces
                        DownstreamSenderVC0T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc0,
                        local_relay_interface,
                        receiver_channel_pointers_ch0,
                        receiver_channel_0_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry);
#endif
                    // EXP_TX_SKIP_IDLE_CHANNELS: one branch over the forwarding channels while none has work.
                    if (!fabric_exp_tx_skip_idle_channels || !exp_forwarding_channels_idle(exp_free_slots_raw)) {
                        tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 1, ENABLE_FIRST_LEVEL_ACK_VC0>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch0,
                            remote_receiver_channels,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            exp_free_slots_raw[1]);
                        if constexpr (is_2d_fabric) {
                            tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 2, ENABLE_FIRST_LEVEL_ACK_VC0>(
                                local_sender_channels,
                                local_sender_channel_worker_interfaces,
                                outbound_to_receiver_channel_pointer_ch0,
                                remote_receiver_channels,
                                channel_connection_established,
                                local_sender_channel_free_slots_stream_ids,
                                sender_channel_from_receiver_credits,
                                inner_loop_perf_telemetry_collector,
                                local_fabric_telemetry,
                                exp_free_slots_raw[2]);
                            tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 3, ENABLE_FIRST_LEVEL_ACK_VC0>(
                                local_sender_channels,
                                local_sender_channel_worker_interfaces,
                                outbound_to_receiver_channel_pointer_ch0,
                                remote_receiver_channels,
                                channel_connection_established,
                                local_sender_channel_free_slots_stream_ids,
                                sender_channel_from_receiver_credits,
                                inner_loop_perf_telemetry_collector,
                                local_fabric_telemetry,
                                exp_free_slots_raw[3]);
                        }
                    }
                    if constexpr (is_2d_fabric && ACTUAL_VC0_SENDER_CHANNELS > 4) {
                        tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 4, ENABLE_FIRST_LEVEL_ACK_VC0>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch0,
                            remote_receiver_channels,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry);
                    }
                }  // end of if constexpr (super_speedy_mode) else block
                if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D_VC1_SERVICED)
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        remote_receiver_channels,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS + 1,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        remote_receiver_channels,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS + 2,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        remote_receiver_channels,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    if constexpr (ACTUAL_VC1_SENDER_CHANNELS > 3) {
                        tx_progress |= run_sender_channel_step<
                            VC1_RECEIVER_CHANNEL,
                            ACTUAL_VC0_SENDER_CHANNELS + 3,
                            ENABLE_FIRST_LEVEL_ACK_VC1>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch1,
                            remote_receiver_channels,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry);
                    }
                    rx_progress |= run_receiver_channel_step<
                        1,
                        ENABLE_FIRST_LEVEL_ACK_VC1,
                        VC1_DOWNSTREAM_EDM_SIZE,  // Explicit size for VC1 downstream interfaces
                        DownstreamSenderVC1T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc1,
                        local_relay_interface,
                        receiver_channel_pointers_ch1,
                        receiver_channel_1_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry);
#endif  // FABRIC_2D_VC1_SERVICED

#if defined(FABRIC_2D_VC2_SERVICED)
                    if constexpr (is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START]) {
                        constexpr size_t SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2 = 1;
                        tx_progress |= run_sender_channel_step_speedy<
                            VC2_LOCAL_CHANNEL_START,
                            to_receiver_packets_sent_streams[VC2_RECEIVER_CHANNEL],
                            SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2,
                            false /*MANAGE_CONNECTION_LIVENESS_IN_SPEEDY_HELPER*/>(
                            local_sender_channels.template get<VC2_LOCAL_CHANNEL_START>(),
                            local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>(),
                            outbound_to_receiver_channel_pointer_ch2,
                            remote_receiver_channels.template get<VC2_RECEIVER_CHANNEL>(),
                            channel_connection_established[VC2_LOCAL_CHANNEL_START],
                            local_sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START],
                            sender_channel_from_receiver_credits.template get<VC2_LOCAL_CHANNEL_START>(),
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            local_speedy_sender_state_vc2);
                    }
                    if constexpr (is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL]) {
                        constexpr size_t RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2 = 1;
                        rx_progress |= run_receiver_channel_step_speedy<
                            VC2_RECEIVER_CHANNEL,
                            VC2_LOCAL_CHANNEL_START,
                            to_receiver_packets_sent_streams[VC2_RECEIVER_CHANNEL],
                            RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2,
                            decltype(receiver_channel_2_trid_tracker)>(
                            local_receiver_channels.template get<VC2_RECEIVER_CHANNEL>(),
                            local_relay_interface,
                            receiver_channel_pointers_ch2,
                            receiver_channel_2_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders.template get<VC2_RECEIVER_CHANNEL>(),
                            routing_table,
                            local_fabric_telemetry,
                            local_speedy_receiver_state_vc2);
                    }
                    static_assert(
                        is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START] ^
                            is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL],
                        "VC2 receiver channel not serviced");
                    static_assert(
                        !is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START] ||
                            !sender_ch_live_check_skip[VC2_LOCAL_CHANNEL_START],
                        "VC2 receiver channel not serviced");
#endif  // FABRIC_2D_VC2_SERVICED
                }
                phase_profiler.end_iteration();
            }
            phase_profiler.flush();
            exp_scan_stack();

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, rx_progress, local_fabric_telemetry);
            }
            // Compute idle conditions and update heartbeats in one helper
            if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
                update_telemetry(
                    local_sender_channel_free_slots_stream_ids,
                    tx_progress,
                    rx_progress,
                    local_fabric_telemetry,
                    fabric_telemetry);
            }

            if ((++fabric_heartbeat_counter & 0x3F) == 0) {
                *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
            }

            if constexpr (enable_context_switch) {
                // shouldn't do noc counter sync since we are not incrementing them
                if constexpr (IDLE_CONTEXT_SWITCHING) {
                    if (did_something) {
                        did_nothing_count = 0;
                    } else {
                        if (did_nothing_count++ > SWITCH_INTERVAL) {
                            did_nothing_count = 0;
                            run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                        }
                    }
                } else {
                    if (did_nothing_count++ > SWITCH_INTERVAL) {
                        did_nothing_count = 0;
                        run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                    }
                }
            } else {
                run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
            }

            if constexpr (any_sender_channel_serviced) {
                close_perf_recording_window(inner_loop_perf_telemetry_collector);
                if constexpr (perf_telemetry_mode != PerfTelemetryRecorderType::NONE) {
                    if (captured_an_event(inner_loop_perf_telemetry_collector) ||
                        any_sender_channels_active(local_sender_channel_free_slots_stream_ids)) {
                        write_perf_recording_window_results(
                            inner_loop_perf_telemetry_collector, local_perf_telemetry_buffer);
                    }
                }
            }
#if defined(PROFILE_STREAMING) && defined(ARCH_BLACKHOLE)
            link_sync::RouterHook::step(fabric_heartbeat_counter);
#endif
        }

        speedy_state_copy_out<super_speedy_mode, 0>(
            persistent_speedy_sender_state,
            persistent_speedy_receiver_state,
            local_speedy_sender_state,
            local_speedy_receiver_state);
#if defined(FABRIC_2D_VC2_SERVICED)
        speedy_state_copy_out<true, 2>(
            persistent_speedy_sender_state_vc2,
            persistent_speedy_receiver_state_vc2,
            local_speedy_sender_state_vc2,
            local_speedy_receiver_state_vc2);
#endif
    };

    uint64_t loop_start_cycles;

    if constexpr (MY_ERISC_ID == 0) {
        while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            switch (reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
                case RouterCommand::RUN:
                    state_manager_l1->state = RouterState::RUNNING;
                    execute_main_loop();
                    break;
                case RouterCommand::PAUSE:
                    // Signal erisc1 to pause before entering pause command handling
                    if constexpr (NUM_ACTIVE_ERISCS > 1) {
                        coordinated_context_switch_start_as_master(termination_signal_ptr);
                    }
                    execute_pause_command(state_manager_l1, termination_signal_ptr);
                    // Signal erisc1 to resume after exiting pause command handling
                    if constexpr (NUM_ACTIVE_ERISCS > 1) {
                        coordinated_context_switch_finish_as_master(termination_signal_ptr);
                    }
                    break;
                default:
                    ASSERT(false); // illegal state transition
                    // Wait that's illegal
                    break;
            };
        }
    } else {
        // We are erisc1, a purely subordinate erisc. We never do anything specific to the state machine so we
        // immediately jump into the main loop. Any time the the master erisc is not in the run state (e.g. it
        // is paused, draining, retraining, etc.), we will enter a busy wait loop in the main run loop
        while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            execute_main_loop();
        }
    }
}

static constexpr uint32_t CONTEXT_SWITCH_TIMEOUT_SHORT = 100000;
template <typename EdmChannelWorkerIFs>
void
#ifdef FABRIC_2D
    __attribute__((noinline))
#endif
    wait_for_static_connection_to_ready(
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    auto establish_static_connection_from_receiver_side = [&](auto& interface, size_t sender_channel_idx) {
        if (!sender_ch_wait_static_connection[sender_channel_idx]) {
            return;
        }
        uint32_t count = 0;
        while (!connect_is_requested(*interface.connection_live_semaphore)
#ifndef ARCH_WORMHOLE
               && !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)
#endif
        ) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
#ifndef ARCH_WORMHOLE
            if constexpr (MY_ERISC_ID == 0) {
                if (count++ == CONTEXT_SWITCH_TIMEOUT_SHORT) {
                    count = 0;
                    run_routing();
                }
            }
#endif
        }
        establish_edm_connection(
            interface,
            local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_idx)]);
    };
    if constexpr (multi_txq_enabled) {
        tuple_for_each_constexpr(
            local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
                if constexpr (
                    is_sender_channel_serviced[idx] ||
                    (exp_tx_upstream_on_this_erisc && peer_sender_channel_serviced[idx])) {
                    establish_static_connection_from_receiver_side(interface, idx);
                }
            });
    } else {
        // Very slight performance regression on WH if we commonize to the above path, so we preserve this path
        // too
        tuple_for_each(
            local_sender_channel_worker_interfaces.channel_worker_interfaces,
            [&](auto& interface, size_t idx) { establish_static_connection_from_receiver_side(interface, idx); });
    }
}

// Returns the number of starting credits for the specified sender channel `i`
// Generally, we will always start with `SENDER_NUM_BUFFERS` of credits,
// except for channels which service transient/worker connections. Those
// sender channels use counter based credit schemes so they are initialized
// to 0.
template <size_t i>
constexpr size_t get_credits_init_val() {
    if (i == 0) {
        // channel 0 is ALWAYS a worker kernel
        return 0;
    }
#if defined(FABRIC_2D_VC2_SERVICED)
    if (i == VC2_LOCAL_CHANNEL_START) {
        // VC2 only supports neighbour exchange with "transient" connections
        return 0;
    }
#endif
    return SENDER_NUM_BUFFERS_ARRAY[i];
};

// SFINAE helper to initialize a single sender channel worker interface
// Only enabled when I < NUM_SENDER_CHANNELS
template <size_t I, size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
FORCE_INLINE typename std::enable_if<(I < NUM_SENDER_CHANNELS), void>::type init_sender_channel_worker_interface(
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_live_semaphore_addresses,
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_info_addresses,
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces) {
    auto connection_live_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_connection_live_semaphore_addresses[I]);
    auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
        local_sender_connection_info_addresses[I]);
    new (&local_sender_channel_worker_interfaces.template get<I>()) tt::tt_fabric::
        StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY[I]>(
            connection_worker_info_ptr,
            0,  // Not used for credits.
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_live_semaphore_ptr),
            sender_channel_ack_cmd_buf_ids[I],
            get_credits_init_val<I>(),
            notify_worker_of_read_counter_update_src_address);
}

// SFINAE overload - no-op when I >= NUM_SENDER_CHANNELS
template <size_t I, size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
typename std::enable_if<(I >= NUM_SENDER_CHANNELS), void>::type init_sender_channel_worker_interface(
    std::array<size_t, NUM_SENDER_CHANNELS>&, std::array<size_t, NUM_SENDER_CHANNELS>&, EdmChannelWorkerIFs&) {
    // No-op when channel index is out of range
}

template <size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
__attribute__((optimize("Os")))
void
#ifdef FABRIC_2D
    __attribute__((noinline))
#endif
    init_local_sender_channel_worker_interfaces(
        std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_live_semaphore_addresses,
        std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_info_addresses,
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces) {
    // manual unrol because previously, going from having this in a loop to unrolling this would
    // lead to a performance regression. Having these unrolled is needed to enable some performance optimizations
    // because setup will differ in that each will be a different type. Keeping them unrolled here let's us
    // stay safe from perf regression due to weirdness of codegen.
    init_sender_channel_worker_interface<0, NUM_SENDER_CHANNELS>(
        local_sender_connection_live_semaphore_addresses,
        local_sender_connection_info_addresses,
        local_sender_channel_worker_interfaces);
    if constexpr (NUM_SENDER_CHANNELS > 1) {
        init_sender_channel_worker_interface<1, NUM_SENDER_CHANNELS>(
            local_sender_connection_live_semaphore_addresses,
            local_sender_connection_info_addresses,
            local_sender_channel_worker_interfaces);
    }
#ifdef FABRIC_2D
    // Use compile-time loop to initialize remaining sender channels (2-6) for code size optimization
    if constexpr (NUM_SENDER_CHANNELS > 2) {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]<size_t I>() {
                 if constexpr (NUM_SENDER_CHANNELS > I) {
                     init_sender_channel_worker_interface<I, NUM_SENDER_CHANNELS>(
                         local_sender_connection_live_semaphore_addresses,
                         local_sender_connection_info_addresses,
                         local_sender_channel_worker_interfaces);
                 }
             }.template operator()<Is + 2>()),
             ...);
        }(std::make_index_sequence<7>{});  // Indices 0-6 map to channels 2-8
    }
#endif
}

// copy the sender_channel_free_slots_stream_ids (in L1) to local memory for performance.
void populate_local_sender_channel_free_slots_stream_id_ordered_map(
    uint32_t has_downstream_edm_vc0_buffer_connection,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    // The whole fabric table is copied, not just this router's compact count, since positions are
    // fabric-flat and a narrower router would otherwise truncate the tail.
    local_sender_channel_free_slots_stream_ids = sender_channel_free_slots_stream_ids;
}

constexpr bool IS_TEARDOWN_MASTER() { return MY_ERISC_ID == 0; }

// Experiment (TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT): a TX queue with nothing to send returns the acknowledgements for the
// packets arriving on its paired RX queue only in a keepalive every ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT cycles, which caps
// one-way traffic at ~128 Ethernet packets per period (fabric_exp/RESULTS.md section 16). ERISC0 sets the period on
// both of this tile's queues before the handshake, whether or not it sends anything itself (a trimmed receive-only
// router still returns the acknowledgements), and restores the firmware value in teardown.
constexpr bool sets_seq_update_timeout = fabric_exp_seq_update_timeout != 0 && IS_TEARDOWN_MASTER();
uint32_t firmware_seq_update_timeouts[2];

__attribute__((optimize("Os"))) void exp_set_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        firmware_seq_update_timeouts[0] = eth_txq_reg_read(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, fabric_exp_seq_update_timeout);
        if constexpr (receiver_txq_id != sender_txq_id) {
            firmware_seq_update_timeouts[1] = eth_txq_reg_read(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, fabric_exp_seq_update_timeout);
        }
    }
}

__attribute__((optimize("Os"))) void exp_restore_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[0]);
        if constexpr (receiver_txq_id != sender_txq_id) {
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[1]);
        }
    }
}

// Note: No termination check is added here intentionally. Both local ERISCs share
// the same termination signal, so both will see it and exit their respective wait
// loops naturally. Adding a termination check here would be unsafe — if one ERISC
// broke out of the sync early and skipped its scratch register write, the other
// could spin forever waiting for a value that never arrives.
__attribute__((optimize("Os"))) void wait_for_other_local_erisc() {
    constexpr uint32_t multi_erisc_sync_start_value = 0x0fed;
    constexpr uint32_t multi_erisc_sync_step2_value = 0x1bad;
    if constexpr (IS_TEARDOWN_MASTER()) {
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(multi_erisc_sync_start_value);
        while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) !=
               multi_erisc_sync_step2_value) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(0);
    } else {
        while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) !=
               multi_erisc_sync_start_value) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(multi_erisc_sync_step2_value);
    }
}

__attribute__((optimize("Os"))) void teardown(
    volatile tt_l1_ptr tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    volatile tt_l1_ptr tt::tt_fabric::EDMStatus* edm_status_ptr,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[0],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[1],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[1],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_1_trid_tracker
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    ,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[VC2_RECEIVER_CHANNEL],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[VC2_RECEIVER_CHANNEL],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_2_trid_tracker
#endif  // FABRIC_2D_VC2_SERVICED
) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (is_receiver_channel_serviced[0]) {
        receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();
#if defined(FABRIC_DRISC_OFFLOAD)
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        noc_async_posted_writes_flushed(noc);
        noc_clear_packet_tags(noc);
        noc_init_one(noc, MEM_NOC_ATOMIC_RET_VAL_ADDR);
        noc_async_write_one_packet(
            reinterpret_cast<uint32_t>(termination_signal_ptr),
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_termination_signal_l1_addr, noc),
            sizeof(uint32_t),
            noc);
        noc_async_write_barrier(noc);
#endif
    }
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (is_receiver_channel_serviced[1]) {
        receiver_channel_1_trid_tracker.all_buffer_slot_transactions_acked();
    }
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    if constexpr (is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL]) {
        receiver_channel_2_trid_tracker.all_buffer_slot_transactions_acked();
    }
#endif  // FABRIC_2D_VC2_SERVICED

    // at minimum, the below call must be updated because in dynamic noc mode, the counters would be shared, so you'd
    // want a sync before this and coordination about which erisc should do the reset (only one of them should do it)
    static_assert(
        noc_mode != DM_DYNAMIC_NOC,
        "The fabric router implementation doesn't support dynamic noc mode. The implementation must be updated to "
        "support this");
    // re-init the noc counters as the noc api used is not incrementing them
    ncrisc_noc_counters_init();

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (wait_for_host_signal) {
        if constexpr (is_local_handshake_master) {
            notify_subordinate_routers(
                edm_channels_mask,
                local_handshake_master_eth_chan,
                (uint32_t)termination_signal_ptr,
                *termination_signal_ptr);
        }
    }

    // write barrier should be coordinated for dynamic noc mode. Safest is probably to do a `wait_for_other_local_erisc`
    // followed by master core doing barrier
    static_assert(noc_mode != DM_DYNAMIC_NOC, "Update here when enabling dynamic noc mode");
    noc_async_write_barrier();
    noc_async_atomic_barrier();

#if defined(FABRIC_DRISC_OFFLOAD)
    if constexpr (is_receiver_channel_serviced[0]) {
        while (get_ptr_val<drisc_free_slots_stream_id>() < static_cast<int32_t>(drisc_ring_slots + 1)) {
        }
        init_ptr_val<drisc_free_slots_stream_id>(0);
    }
#endif

#ifdef ARCH_BLACKHOLE
    // Tagged write completion does not reset command-buffer transaction IDs.
    // Restore this ERISC's dedicated NoC before the kernel-return checks.
    noc_clear_packet_tags(noc_index);
#endif

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    exp_restore_seq_update_timeouts();
    if constexpr (IS_TEARDOWN_MASTER()) {
        *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;
    }
}

__attribute__((optimize("Os"))) void initialize_state_for_txq1_active_mode() {
    eth_enable_packet_mode(receiver_txq_id);
    for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)[i] = 0;
    }
    eth_txq_reg_write(receiver_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
}
__attribute__((optimize("Os"))) void initialize_state_for_txq1_active_mode_sender_side() {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address)[i] = 0;
    }
}

// Initialize fabric telemetry structure in L1
// This must be called early in kernel_main to ensure telemetry is properly initialized
void initialize_fabric_telemetry() {
    // Get pointer to telemetry structure in L1
    volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
        reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);

    // Get pointer to routing table, which contains some information we need to re-export
    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);

    // Zero the entire telemetry structure to clear any garbage values
    memset(const_cast<FabricTelemetry*>(fabric_telemetry), 0, sizeof(FabricTelemetry));

    // Populate static_info fields with compile-time topology information
    fabric_telemetry->static_info.mesh_id = routing_table_l1->my_mesh_id;
    fabric_telemetry->static_info.device_id = routing_table_l1->my_device_id;
    fabric_telemetry->static_info.direction = static_cast<uint8_t>(my_direction);
    fabric_telemetry->static_info.version = FABRIC_TELEMETRY_VERSION;
    fabric_telemetry->static_info.fabric_config = 0;  // Reserved for future use

    // Initialize neighbor info to sentinel values (will be populated during handshake)
    // Using 0xFFFF/0xFF as sentinel to distinguish "not yet set" from valid value 0
    fabric_telemetry->static_info.neighbor_mesh_id = 0xFFFF;
    fabric_telemetry->static_info.neighbor_device_id = 0xFF;

    // Set supported_stats bitmask to enable all telemetry features
    fabric_telemetry->static_info.supported_stats = static_cast<DynamicStatistics>(
        DynamicStatistics::BANDWIDTH | DynamicStatistics::ROUTER_STATE | DynamicStatistics::HEARTBEAT_TX |
        DynamicStatistics::HEARTBEAT_RX);

    // Initialize router_state to STANDBY (will transition to ACTIVE when router starts)
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        // In dual-ERISC mode, each ERISC initializes its own entry
        fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = RouterState::INITIALIZING;
    } else {
        // In single-ERISC mode, initialize both entries (only [0] will be used)
        fabric_telemetry->dynamic_info.erisc[0].router_state = RouterState::INITIALIZING;
        fabric_telemetry->dynamic_info.erisc[1].router_state = RouterState::INITIALIZING;
    }
}

void kernel_main() {
    exp_paint_stack();
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.reset();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
#endif

    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
    state_manager_l1->state = RouterState::INITIALIZING;

    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();

    // Experimental primitive-cost microbenchmark; must run before any stateful command-buffer setup.
    if constexpr (phase_profile_mode >= 3 && any_receiver_channel_serviced) {
        run_fabric_erisc_microbench<
            phase_profile_microbench_results_addr,
            phase_profile_microbench_scratch_addr,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::local_chip_data_cmd_buf,
            sender_txq_id>();
    }

    // Initialize fabric telemetry early to ensure valid values before router starts
    initialize_fabric_telemetry();

    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    exp_set_seq_update_timeouts();
    asm volatile("nop");
    static_assert(
        receiver_txq_id == sender_txq_id || receiver_txq_id == 1,
        "For multi-txq mode, the only currently supported configuration is sender_txq_id=0 and receiver_txq_id=1");
    if constexpr (receiver_txq_id != sender_txq_id) {
        // Role selection is structural (2-erisc split), not workload-driven: use the untrimmed role
        // flags, not the trimming-aware per-channel serviced bits (whose channel 0 can now be
        // legitimately trimmed while a sibling channel of the same role stays active).
        if constexpr (risc_services_receiver_channels) {
            initialize_state_for_txq1_active_mode();
        }
        if constexpr (risc_services_sender_channels) {
            initialize_state_for_txq1_active_mode_sender_side();
        }
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);
#endif
    //
    // COMMON CT ARGS (not specific to sender or receiver)
    //

    // Initialize stream register state for credit management across the Ethernet link.
    // We make sure to do this before we handshake to guarantee that the registers are
    // initialized before the other side has any possibility of modifying them.
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
#if defined(FABRIC_DRISC_OFFLOAD)
    init_ptr_val<drisc_free_slots_stream_id>(0);
#endif
    if constexpr (exp_tx_upstream_on_this_erisc) {
        init_ptr_val<exp_upstream_in_flight_stream_id>(0);
    }
    init_stream_reg_if_used<to_sender_packets_acked_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[1]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[1]>(0);
    // The first sender channel in the array is always for the transient/worker connection
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);  // LOCAL WORKER
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);  // Compact index 0

    // Init retrain sync reg
    if (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION));
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

    if constexpr (is_2d_fabric) {
        // Receiver channels 1 and 2 are only populated when a VC densifies onto them, so guard the
        // init rather than writing through an unassigned entry.
        init_stream_reg_if_used<to_receiver_packets_sent_streams[1]>(0);
        init_stream_reg_if_used<to_receiver_packets_sent_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[3]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[4]>(0);

        // Initialize completion streams and sender channel free slots for channels 2..MAX-1 using compile-time loop.
        // Index sequence covers Is=0..7 → channels 2..9 (MAX_NUM_SENDER_CHANNELS=10).
        // Positions this router does not service are skipped; the buffer count comes from the compact
        // array.
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 init_stream_reg_if_used<to_sender_packets_completed_streams[Is + 2]>(0);
                 if constexpr (services_fabric_position(Is + 2)) {
                     init_ptr_val<sender_channel_free_slots_stream_ids[Is + 2]>(
                         SENDER_NUM_BUFFERS_ARRAY[compact_sender_for_fabric_position(Is + 2)]);
                 }
             }()),
             ...);
        }(std::make_index_sequence<8>{});
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);
#endif

    if constexpr (code_profiling_enabled_timers_bitfield != 0) {
        clear_code_profiling_buffer(code_profiling_buffer_base_addr);
    }

    // TODO: CONVERT TO SEMAPHORE
    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    volatile auto edm_local_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(edm_local_sync_ptr_addr);
    volatile auto edm_status_ptr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);

    // In persistent mode, we must rely on static addresses for our local semaphores that are locally
    // initialized, rather than metal device APIs. This way different subdevice programs can reliably
    // resolve the semaphore addresses on the EDM core

    size_t arg_idx = 0;
    ///////////////////////
    // Common runtime args:
    ///////////////////////
    // Read the fixed maximum sender-channel connection semaphore array.
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_semaphore_addrs;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_semaphore_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    // Read the fixed maximum sender-channel connection buffer-index array.
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_buffer_index_ids;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_buffer_index_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    // downstream EDM VC0 connection info
    const auto has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++);

    // For 2D: read densely packed connection data for up to four compact direction slots.
    // For 1D: reads as 1D and only uses first element
#if defined(FABRIC_2D)
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    [[maybe_unused]] const auto downstream_edm_vc0_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
#endif

    [[maybe_unused]] const auto downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_registration_ids;
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_location_info_addresses;
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0>
        downstream_edm_vc0_buffer_index_semaphore_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_index_semaphore_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    [[maybe_unused]] const auto downstream_edm_vc0_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_buffer_index_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
#endif

#if defined(FABRIC_2D_VC1_ACTIVE)
    const auto has_downstream_edm_vc1_buffer_connection = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    const auto downstream_edm_vc1_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc1_noc_y = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_location_info_addresses;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_buffer_index_semaphore_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_buffer_index_semaphore_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#endif  // FABRIC_2D_VC1_ACTIVE
    // unused - to be deleted
    [[maybe_unused]]
    const auto downstream_vc0_noc_interface_buffer_index_local_addr = 0;
    const auto downstream_vc1_noc_interface_buffer_index_local_addr = 0;

    // Read teardown semaphores (MAX_NUM_SENDER_CHANNELS channels)
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> my_sem_for_teardown_from_edm;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        my_sem_for_teardown_from_edm[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    ///////////////////////////////////////////////
    // Local tensix (relay) connection runtime args
    // UDM mode only - packed at end of runtime args
    ///////////////////////////////////////////////
    const auto has_local_tensix_relay_connection = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_tensix_relay_buffer_base_address = 0;
    uint32_t local_tensix_relay_noc_x = 0;
    uint32_t local_tensix_relay_noc_y = 0;
    uint32_t local_tensix_relay_worker_registration_id = 0;
    uint32_t local_tensix_relay_worker_location_info_address = 0;
    uint32_t local_tensix_relay_free_slots_stream_id = 0;
    uint32_t local_tensix_relay_connection_buffer_index_id = 0;
    if constexpr (udm_mode) {
        if (has_local_tensix_relay_connection) {
            local_tensix_relay_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_noc_x = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_noc_y = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_free_slots_stream_id = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    //  initialize the statically allocated "semaphores"
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (([&]<size_t I>() {
             if constexpr (is_sender_channel_serviced[I]) {
                 *reinterpret_cast<volatile uint32_t*>(local_sender_channel_connection_semaphore_addrs[I]) = 0;
                 // Zero the whole SenderChannelProducerCursor block, not just its first word: the
                 // producer reads the block back in one NOC read at connection open, so every field
                 // must be initialized here. This is the only time the router touches it.
                 auto* const producer_cursor =
                     reinterpret_cast<volatile uint32_t*>(local_sender_channel_connection_buffer_index_ids[I]);
#pragma GCC unroll 1
                 for (size_t w = 0; w < sizeof(tt::tt_fabric::SenderChannelProducerCursor) / sizeof(uint32_t); ++w) {
                     producer_cursor[w] = 0;
                 }
             }
         }.template operator()<Is>()),
         ...);
    }(std::make_index_sequence<NUM_SENDER_CHANNELS>{});
    asm volatile("nop");

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STARTED);
#endif
    *edm_status_ptr = tt::tt_fabric::EDMStatus::STARTED;
    asm volatile("nop");

    //////////////////////////////
    //////////////////////////////
    //        Object Setup
    //////////////////////////////
    //////////////////////////////

    // Hack for mux mode until all remaining VC1 logic is removed from fabric
    // Needed so `downstream_edm_noc_interfaces_vc0` can be initialized properly below
    // Issue #33360 TODO: Create a new array for downstream receiver stream IDs
    // so we can remove this hack.
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids;
    // std::array<uint32_t, NUM_SENDER_CHANNELS == 1 ? 2 : NUM_SENDER_CHANNELS>
    // local_sender_channel_free_slots_stream_ids;

    [[maybe_unused]] const auto& local_sem_for_teardown_from_downstream_edm =
        take_first_n_elements<NUM_DOWNSTREAM_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(my_sem_for_teardown_from_edm);

    auto remote_receiver_channels = tt::tt_fabric::ReceiverChannelBuffersFromAllocs<
        PACKET_HEADER_TYPE,
        eth_remote_channel_allocs,
        REMOTE_RECEIVER_TO_ENTRY_IDX>::make();

    auto local_receiver_channels = tt::tt_fabric::
        ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();

#if !defined(FABRIC_2D_VC2_SERVICED)
    constexpr bool exp_use_lean_stride_sender_channels =
        fabric_exp_tx_lean_vc0_loop && fabric_exp_tx_send_stride_slot && !super_speedy_mode;
#else
    constexpr bool exp_use_lean_stride_sender_channels = false;
#endif
    using SenderChannelsAllocT = std::conditional_t<
        exp_use_lean_stride_sender_channels,
        tt::tt_fabric::LeanSenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>,
        tt::tt_fabric::SenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>>;
    auto local_sender_channels = SenderChannelsAllocT::make();

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            local_sender_channel_connection_semaphore_addrs);
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_channel_0_connection_info_addr,
                local_sender_channel_1_connection_info_addr,
                local_sender_channel_2_connection_info_addr,
                local_sender_channel_3_connection_info_addr,
                local_sender_channel_4_connection_info_addr,
                local_sender_channel_5_connection_info_addr,
                local_sender_channel_6_connection_info_addr,
                local_sender_channel_7_connection_info_addr,
                local_sender_channel_8_connection_info_addr,
                local_sender_channel_9_connection_info_addr});

    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }
    // create the sender channel worker interfaces with input array of number of buffers
    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
#endif

    // TODO: change to TMP.
    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>, VC0_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc0;

    // Ensure array size is at least 1 to avoid undefined behavior
    static_assert(VC0_DOWNSTREAM_EDM_SIZE > 0, "VC0_DOWNSTREAM_EDM_SIZE must be at least 1");
#if defined(FABRIC_2D)
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc =
        use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>();
#else
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc = false;
#endif

    populate_local_sender_channel_free_slots_stream_id_ordered_map(
        has_downstream_edm_vc0_buffer_connection, local_sender_channel_free_slots_stream_ids);

#if !defined(FABRIC_DRISC_OFFLOAD)
    if (has_downstream_edm_vc0_buffer_connection) {
        // Only bit 0 is set for 1D
        // For 2D: 3 bits set for compact indices 0, 1, 2 (excluding router's own direction)
        uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;  // 4-bit mask
        uint32_t compact_index = 0;
        uint32_t dense_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address = local_sem_for_teardown_from_downstream_edm[dense_index];
                // reset the handshake addresses to 0 (this is for router -> router handshake for connections over noc)
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
#if defined(FABRIC_2D)
                auto receiver_channel_free_slots_stream_id = StreamId{vc0_local_free_slots_stream_id(compact_index)};
#else
                auto receiver_channel_free_slots_stream_id = StreamId{vc0_local_free_slots_stream_id(0)};
#endif
                new (&downstream_edm_noc_interfaces_vc0[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>(
                        // persistent_mode -> hardcode to false for 1D because for 1D, EDM -> EDM
                        // connections we must always use semaphore lookup
                        // For 2D, downstream_edm_vc0_semaphore_id is an address.
                        is_persistent_fabric,
                        (downstream_edm_vc0_noc_x >> (dense_index * 8)) & 0xFF,
                        (downstream_edm_vc0_noc_y >> (dense_index * 8)) & 0xFF,
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_base_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_base_address,
#endif
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC0,
#if defined(FABRIC_2D)
                        // connection handshake address on downstream edm
                        downstream_edm_vc0_worker_registration_ids[dense_index],
                        // worker location info address on downstream edm
                        // written by this interface when it connects to the downstream edm
                        // so that the downstream edm knows who its upstream peer is
                        downstream_edm_vc0_worker_location_info_addresses[dense_index],
#else
                        downstream_edm_vc0_worker_registration_id,
                        downstream_edm_vc0_worker_location_info_address,
#endif
                        channel_buffer_size,
                // Used to park current write pointer value at the downstream edm
                // when this interface disconnects from the downstream edm.
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_index_semaphore_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_index_semaphore_address,
#endif
                        0,  // Unused for Router->Router connections. Router->Router always uses stream registers for
                            // credits. Used by Worker->Router connections. This is an address in the worker's L1. The
                            // Router that a Worker adapter is connected to writes its read counter to this address. The
                            // worker uses this to calculate free slots in the router's sender channel.
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        downstream_vc0_noc_interface_buffer_index_local_addr,  // keep common, since its a scratch noc
                                                                               // read dest.

#if defined(FABRIC_2D)
                        get_vc0_downstream_sender_channel_free_slots_stream_id(compact_index),
#else
                        // Issue #33360 TODO: Create a new array for explicitly holding downstream receiver stream IDs
                        // so we can remove this hack.
                        sender_channel_1_free_slots_stream_id,
#endif
                        // This is our local stream register for the copy of the downstream router's
                        // free slots
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[0],
                        receiver_channel_forwarding_sync_cmd_buf_ids[0]);
#if defined(FABRIC_2D)
                ASSERT(
                    downstream_edm_noc_interfaces_vc0[compact_index].get_worker_credits_stream_id() ==
                    vc0_local_free_slots_stream_id(compact_index));
#else
                ASSERT(
                    downstream_edm_noc_interfaces_vc0[compact_index].get_worker_credits_stream_id() ==
                    vc0_local_free_slots_stream_id(0));
#endif
                // Only receiver channel servicing cores should be setting up the noc cmd buf.
                if constexpr (
                    NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !use_vc0_single_live_stateful_noc) {
                    downstream_edm_noc_interfaces_vc0[compact_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
#if defined(FABRIC_2D)
        if constexpr (use_vc0_single_live_stateful_noc) {
            // LIVE == 0b0001 above pins the only destination to compact slot 0. Program it after
            // construction so it is the retained state for both the data and credit command buffers.
            if (has_downstream_edm_vc0_buffer_connection & 0x1) {
                downstream_edm_noc_interfaces_vc0[0]
                    .template setup_edm_noc_cmd_buf<
                        tt::tt_fabric::edm_to_downstream_noc,
                        tt::tt_fabric::forward_and_local_write_noc_vc>();
            } else {
                ASSERT(false);
            }
        }
#endif
    }
#endif  // !FABRIC_DRISC_OFFLOAD

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>, VC1_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc1;
#if defined(FABRIC_2D_VC1_ACTIVE)
    // Ensure array size is at least 1 to avoid undefined behavior
    static_assert(VC1_DOWNSTREAM_EDM_SIZE > 0, "VC1_DOWNSTREAM_EDM_SIZE must be at least 1");
    if (has_downstream_edm_vc1_buffer_connection) {
        uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0xF;  // 4-bit mask
        uint32_t compact_index = 0;
        uint32_t dense_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address =
                    local_sem_for_teardown_from_downstream_edm[dense_index + NUM_DOWNSTREAM_SENDERS_VC0];
                // reset the handshake addresses to 0 (this is for router -> router handshake for connections over
                // noc)
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
                auto receiver_channel_free_slots_stream_id = StreamId{vc1_local_free_slots_stream_id(compact_index)};
                new (&downstream_edm_noc_interfaces_vc1[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>(
                        is_persistent_fabric,
                        (downstream_edm_vc1_noc_x >> (dense_index * 8)) & 0xFF,
                        (downstream_edm_vc1_noc_y >> (dense_index * 8)) & 0xFF,
                        downstream_edm_vc1_buffer_base_addresses[dense_index],
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC1,
                        downstream_edm_vc1_worker_registration_ids[dense_index],
                        downstream_edm_vc1_worker_location_info_addresses[dense_index],
                        channel_buffer_size,
                        downstream_edm_vc1_buffer_index_semaphore_addresses[dense_index],
                        0,
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        downstream_vc1_noc_interface_buffer_index_local_addr,
                        get_vc1_downstream_sender_channel_free_slots_stream_id(compact_index),
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[1],
                        receiver_channel_forwarding_sync_cmd_buf_ids[1]);
                ASSERT(
                    downstream_edm_noc_interfaces_vc1[compact_index].get_worker_credits_stream_id() ==
                    vc1_local_free_slots_stream_id(compact_index));
                // Only receiver channel servicing cores should be setting up the noc cmd buf.
                if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                    downstream_edm_noc_interfaces_vc1[compact_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
    }
#endif  // FABRIC_2D_VC1_ACTIVE

    // Setup local tensix relay connection (UDM mode only)
    // This is a separate connection path from downstream EDM connections
    // Relay handles forwarding packets to local chip workers
    // Uses dedicated stream IDs and L1 locations to avoid assumptions about direction indexing
    // LOCAL_RELAY_NUM_BUFFERS comes from compile-time args (propagated from relay config)
    RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS> local_relay_interface;
    if constexpr (udm_mode) {
        if (has_local_tensix_relay_connection) {
            // Reuse RouterToRouterSender for relay connection
            // Relay is just another sender interface, but pointing to local tensix instead of remote router

            new (&local_relay_interface) RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS>(
                true,  // persistent_mode - relay is always a persistent connection
                local_tensix_relay_noc_x,
                local_tensix_relay_noc_y,
                local_tensix_relay_buffer_base_address,
                LOCAL_RELAY_NUM_BUFFERS,  // Use compile-time constant
                local_tensix_relay_worker_registration_id,
                local_tensix_relay_worker_location_info_address,
                channel_buffer_size,
                local_tensix_relay_connection_buffer_index_id,  // From runtime args - dedicated L1 location for relay
                                                                // connection
                0,        // worker read counter address - unused for Router->Relay (uses stream registers)
                nullptr,  // teardown semaphore - router never calls close on relay
                0,        // buffer_index_local_addr - scratch space for noc reads
                // Remote stream: relay's free slots stream (what relay publishes) - from runtime args
                StreamId{local_tensix_relay_free_slots_stream_id},
                // Local stream: our copy of relay's free slots - dedicated stream ID for relay
                StreamId{tensix_relay_local_free_slots_stream_id},
                receiver_channel_forwarding_data_cmd_buf_ids[0],
                receiver_channel_forwarding_sync_cmd_buf_ids[0]);

            // Setup NOC command buffer for relay interface
            if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                local_relay_interface.template setup_edm_noc_cmd_buf<
                    tt::tt_fabric::edm_to_downstream_noc,
                    tt::tt_fabric::forward_and_local_write_noc_vc>();
            }
        }
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);
#endif

    // initialize the local receiver channel buffers
    local_receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init<eth_remote_channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));

    // initialize the local sender channel buffers
    local_sender_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));

    // Interface construction is a structural sender-RISC obligation. Trimming can disable
    // sender channel 0 while leaving static forwarding sender channels active, so channel 0
    // cannot be used as a proxy for whether these interfaces must be initialized.
    if constexpr (risc_services_sender_channels || exp_tx_upstream_on_this_erisc) {
        init_local_sender_channel_worker_interfaces(
            local_sender_connection_live_semaphore_addresses,
            local_sender_connection_info_addresses,
            local_sender_channel_worker_interfaces);
    }

    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[0],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_0_trid_tracker;
    receiver_channel_0_trid_tracker.init();

#if defined(FABRIC_2D_VC1_ACTIVE)
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[1],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[1],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_1_trid_tracker;
    receiver_channel_1_trid_tracker.init();
#endif  // FABRIC_2D_VC1_ACTIVE

#if defined(FABRIC_2D_VC2_SERVICED)
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[VC2_RECEIVER_CHANNEL],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[VC2_RECEIVER_CHANNEL],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_2_trid_tracker;
    receiver_channel_2_trid_tracker.init();
#endif  // FABRIC_2D_VC2_SERVICED

#ifdef ARCH_BLACKHOLE
    // A Blackhole hardware bug requires all noc inline writes to be non-posted so we hardcode to false here
    // A more detailed description can be found in `noc_inline_dw_write` in the `dataflow_api` header file
    [[maybe_unused]] constexpr bool use_posted_writes_for_connection_open = false;
#else
    [[maybe_unused]] constexpr bool use_posted_writes_for_connection_open = true;
#endif

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        // This barrier is here just in case the initialization process of any of the sender/receiver channel
        // implementations require any assumptions about channel contents or anything similar. Without it there
        // is possibility of a race. The race would be where the the risc core responsible for Ethernet level handshake
        // completes before the other risc finishes setup of channel/credit datastructures. If that happened, then
        // it would be possible for the other (remote) Ethernet core to start sending packets/credits to our core before
        // all of our cores are done setup, leading to potentially undefined behavior.
        //
        // Whether or not there truly is a race in a given snapshot/commit hash is not relevant. The intention with this
        // is to avoid all possible footguns as implementations of underlying datastructures potenntially change over
        // time.
        wait_for_other_local_erisc();
    }
    if constexpr (enable_ethernet_handshake) {
        if constexpr (is_handshake_sender) {
            erisc::datamover::handshake::fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        } else {
            erisc::datamover::handshake::fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        }

        // After handshake completes, extract neighbor info and populate telemetry
        {
            volatile tt_l1_ptr auto* handshake_info =
                reinterpret_cast<volatile tt_l1_ptr erisc::datamover::handshake::handshake_info_t*>(handshake_addr);
            volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
                reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(
                    eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
            fabric_telemetry->static_info.neighbor_mesh_id = handshake_info->neighbor_mesh_id;
            fabric_telemetry->static_info.neighbor_device_id = handshake_info->neighbor_device_id;
        }

        *edm_status_ptr = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;
        asm volatile("nop");

        if constexpr (wait_for_host_signal) {
            if constexpr (is_local_handshake_master) {
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms - 1, termination_signal_ptr);
                // This master sends notification to self for multi risc in single eth core case,
                // This still send to self even though with single risc core case, but no side effects
                constexpr uint32_t exclude_eth_chan = std::numeric_limits<uint32_t>::max();
                notify_subordinate_routers(
                    edm_channels_mask, exclude_eth_chan, (uint32_t)edm_local_sync_ptr, num_local_edms);
            } else {
                notify_master_router(local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr);
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms, termination_signal_ptr);
            }

            *edm_status_ptr = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;

            // 1. All risc cores wait for READY_FOR_TRAFFIC signal
            // 2. All risc cores in master eth core receive signal from host and exits from this wait
            //    Other subordinate risc cores wait for this signal
            // 4. The other subordinate risc cores receive the READY_FOR_TRAFFIC signal and exit from this wait
            wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                (uint32_t)edm_status_ptr, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC, termination_signal_ptr);

            if constexpr (is_local_handshake_master) {
                // 3. Only master risc core notifies all subordinate risc cores (except subordinate riscs in master eth
                // core)
                notify_subordinate_routers(
                    edm_channels_mask,
                    local_handshake_master_eth_chan,
                    (uint32_t)edm_status_ptr,
                    tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
            }
        }
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);
#endif
    // if enable the tensix extension, then before open downstream connection, need to wait for downstream tensix ready
    // for connection.
    if constexpr (num_ds_or_local_tensix_connections) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
            (uint32_t)edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections, termination_signal_ptr);
    }

#if defined(FABRIC_DRISC_OFFLOAD)
    if constexpr (is_receiver_channel_serviced[0]) {
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        noc_async_write_one_packet(
            ROUTING_TABLE_BASE,
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_routing_table_l1_addr, noc),
            sizeof(tt::tt_fabric::routing_l1_info_t),
            noc);
        noc_async_write_barrier(noc);
        noc_async_write_one_packet(
            reinterpret_cast<uint32_t>(edm_status_ptr),
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_e1_ready_l1_addr, noc),
            sizeof(uint32_t),
            noc);
        noc_async_write_barrier(noc);
        while (get_ptr_val<drisc_free_slots_stream_id>() == 0 &&
               !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        }
        drisc_offload_next_slot_addr = drisc_ring_base_addr;
        drisc_offload_init_noc_state();
    }
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
#else
    if constexpr (is_2d_fabric) {
        // Helper function to open downstream EDM connections, works for both VC0 and VC1
        auto open_downstream_edm_connections = [](auto& downstream_edm_noc_interfaces, uint32_t has_downstream_edm) {
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    // open connections with available downstream edms
                    downstream_edm_noc_interfaces[edm_index]
                        .template open<
                            false,
                            use_posted_writes_for_connection_open,
                            tt::tt_fabric::worker_handshake_noc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
        };

        // Open gates are builder-resolved (trimming- and VC1-mode-aware): VC1 downstreams are
        // consumed by receiver 0's crossover step on crossover routers, not by receiver 1.
        if constexpr (open_downstream_vc0_connections) {
            open_downstream_edm_connections(
                downstream_edm_noc_interfaces_vc0, has_downstream_edm_vc0_buffer_connection & 0xF);
        }
#if defined(FABRIC_2D_VC1_ACTIVE)
        if constexpr (open_downstream_vc1_connections) {
            open_downstream_edm_connections(
                downstream_edm_noc_interfaces_vc1, has_downstream_edm_vc1_buffer_connection & 0xF);
        }
#endif
        if constexpr (udm_mode) {
            if (has_local_tensix_relay_connection) {
                // open connection here to relay kernel
                local_relay_interface
                    .template open<false, use_posted_writes_for_connection_open, tt::tt_fabric::worker_handshake_noc>();
            }
        }
    } else {
        // We can check just the first index because all receiver channels are serviced by the same core
        if constexpr (is_receiver_channel_serviced[0]) {
            if (has_downstream_edm_vc0_buffer_connection) {
                downstream_edm_noc_interfaces_vc0[0]
                    .template open<false, use_posted_writes_for_connection_open, tt::tt_fabric::worker_handshake_noc>();
                ASSERT(
                    get_ptr_val(downstream_edm_noc_interfaces_vc0[0].get_worker_credits_stream_id()) ==
                    DOWNSTREAM_SENDER_NUM_BUFFERS_VC0);
            }
        }
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
    if constexpr (open_downstream_vc0_connections and NUM_ACTIVE_ERISCS > 1) {
        // Two erisc mode requires us to reorder the cmd buf programming/state setting
        // because we need to reshuffle some of our cmd_buf/noc assignments around for
        // just the fabric bringup phase. These calls are also located earlier for the
        // single erisc mode
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;  // 3-bit mask
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    downstream_edm_noc_interfaces_vc0[edm_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
        }
    }
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (open_downstream_vc1_connections and NUM_ACTIVE_ERISCS > 1) {
        // Two erisc mode requires us to reorder the cmd buf programming/state setting
        // because we need to reshuffle some of our cmd_buf/noc assignments around for
        // just the fabric bringup phase. These calls are also located earlier for the
        // single erisc mode
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0xF;  // 3-bit mask
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    downstream_edm_noc_interfaces_vc1[edm_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
        }
    }
#endif  // FABRIC_2D_VC1_ACTIVE
#endif  // FABRIC_DRISC_OFFLOAD
    std::array<uint8_t, num_eth_ports> port_direction_table;

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);
#endif

    WAYPOINT("FSCW");
    wait_for_static_connection_to_ready(
        local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids, termination_signal_ptr);
    WAYPOINT("FSCD");

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE);
#endif
    if constexpr (MY_ERISC_ID == 0) {
        const auto* routing_table_l1 =
            reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
        auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
        state_manager_l1->state = RouterState::INITIALIZING;
    }

    //////////////////////////////
    //////////////////////////////
    //        MAIN LOOP
    //////////////////////////////
    //////////////////////////////
    run_fabric_edm_main_loop<
        NUM_RECEIVER_CHANNELS,
        RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>,
        RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>,
        RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS>>(
        local_receiver_channels,
        local_sender_channels,
        local_sender_channel_worker_interfaces,
        downstream_edm_noc_interfaces_vc0,
        downstream_edm_noc_interfaces_vc1,
        // pass in the relay adpator
        local_relay_interface,
        remote_receiver_channels,
        termination_signal_ptr,
        receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
        receiver_channel_1_trid_tracker,
#endif  // FABRIC_2D_VC1_ACTIVE
#if defined(FABRIC_2D_VC2_SERVICED)
        receiver_channel_2_trid_tracker,
#endif  // FABRIC_2D_VC2_SERVICED
        port_direction_table,
        local_sender_channel_free_slots_stream_ids);
    WAYPOINT("LPDN");
    // make sure all the noc transactions are acked before re-init the noc counters
    teardown(
        termination_signal_ptr,
        edm_status_ptr,
        receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
        ,
        receiver_channel_1_trid_tracker
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
        ,
        receiver_channel_2_trid_tracker
#endif  // FABRIC_2D_VC2_SERVICED
    );

    exp_scan_stack(true);
    set_l1_data_cache<false>();
    WAYPOINT("DONE");
}
