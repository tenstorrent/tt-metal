// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"

#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/edm_fabric_counters.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
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
#include "hostdevcommon/fabric_common.h"
#include "hostdev/fabric_telemetry_msgs.h"
#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edge_node_router.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace tt::tt_fabric;

// Type alias for cleaner access to 2D mesh routing constants
using MeshRoutingFields = tt::tt_fabric::RoutingFieldsConstants::Mesh;
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

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using SenderEthChannel = StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>;

static constexpr bool PERF_TELEMETRY_DISABLED = perf_telemetry_mode == PerfTelemetryRecorderType::NONE;
static constexpr bool PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH =
    perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH;
using PerfTelemetryRecorder = std::conditional_t<
    PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH,
    LowResolutionBandwidthTelemetry,
    std::conditional_t<PERF_TELEMETRY_DISABLED, bool, std::nullptr_t>>;

// Currently, we enable elastic channels in an all-or-nothing manner for router -> router
// connections.

constexpr bool ANY_SENDER_CHANNELS_ARE_ELASTIC() {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        if (IS_ELASTIC_SENDER_CHANNEL[i]) {
            return true;
        }
    }
    return false;
}

constexpr bool PERSISTENT_SENDER_CHANNELS_ARE_ELASTIC = ANY_SENDER_CHANNELS_ARE_ELASTIC();

// Stubbed out the elastic channel writer adapter until elastic channels implemented
// Issue: https://github.com/tenstorrent/tt-metal/issues/26311
template <uint8_t SLOTS_PER_CHUNK, uint16_t CHUNK_SIZE_BYTES>
struct RouterElasticChannelWriterAdapter {};

template <uint8_t SENDER_NUM_BUFFERS>
using RouterToRouterSender = std::conditional_t<
    PERSISTENT_SENDER_CHANNELS_ARE_ELASTIC,
    tt::tt_fabric::RouterElasticChannelWriterAdapter<CHUNK_N_PKTS, channel_buffer_size>,
    tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>>;

constexpr bool is_spine_direction(eth_chan_directions direction) {
    return direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;
}

static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_free_slots_stream_ids = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id,
    sender_channel_5_free_slots_stream_id,
    sender_channel_6_free_slots_stream_id,
    sender_channel_7_free_slots_stream_id};
static_assert(sender_channel_free_slots_stream_ids[0] == 21);
static_assert(sender_channel_free_slots_stream_ids[1] == 22);
static_assert(sender_channel_free_slots_stream_ids[2] == 23);
static_assert(sender_channel_free_slots_stream_ids[3] == 24);
static_assert(sender_channel_free_slots_stream_ids[4] == 25);
static_assert(sender_channel_free_slots_stream_ids[5] == 26);
static_assert(sender_channel_free_slots_stream_ids[6] == 27);
static_assert(sender_channel_free_slots_stream_ids[7] == 28);

// For 2D fabric: maps compact index to downstream direction for each my_direction
// For 1D fabric: only 1 downstream direction per router (EAST forwards to WEST in 1D linear topology)
#if defined(FABRIC_2D)
constexpr uint32_t edm_index_to_edm_direction[eth_chan_directions::COUNT][NUM_DOWNSTREAM_SENDERS_VC0] = {
    {eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH},  // EAST router
    {eth_chan_directions::EAST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH},  // WEST router
    {eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::SOUTH},   // NORTH router
    {eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH},   // SOUTH router
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
// Compact indices 0, 1, 2 map to sender channels 4, 5, 6 (VC1 channels)
// Direction rules are identical to VC0, but offset by 3 to skip VC0's channels 1-3
//
// VC1 sender channel mapping:
//   [4] → VC1 channel 0 (compact index varies by direction)
//   [5] → VC1 channel 1 (compact index varies by direction)
//   [6] → VC1 channel 2 (compact index varies by direction)
constexpr uint32_t get_vc1_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        // downstream sender channel = 3 + my_direction (maps to channels 4-6)
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[3 + my_direction];
    } else {
        // downstream sender channel = 4 + my_direction (maps to channels 4-6)
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[4 + my_direction];
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

// Determine which sender channels are "turn" channels (i.e., north/south for east/west routers)
// Channel 0 is always for local workers, so it's never a turn channel
// For 2D fabric, channels 1-3 correspond to compact indices 0-2, which map to actual directions
constexpr auto get_sender_channel_turn_statuses() -> std::array<bool, MAX_NUM_SENDER_CHANNELS_VC0> {
    std::array<bool, MAX_NUM_SENDER_CHANNELS_VC0> turn_statuses = {};  // Initialize to false

    // Channel 0 is always for local workers, never a turn channel
    // Only non-spine routers (EAST/WEST) have turn channels
    if constexpr (!is_spine_direction(static_cast<eth_chan_directions>(my_direction))) {
        // Check each sender channel (1-3) to see if it goes to a spine direction (NORTH/SOUTH)
        // Sender channel i (for i=1,2,3) corresponds to compact index (i-1)
        for (size_t sender_channel = 1; sender_channel < MAX_NUM_SENDER_CHANNELS_VC0; sender_channel++) {
            size_t compact_index = sender_channel - 1;
            eth_chan_directions actual_direction = map_compact_index_to_direction(compact_index);
            turn_statuses[sender_channel] = is_spine_direction(actual_direction);
        }
    }

    return turn_statuses;
}

// Map downstream direction to compact array index [0-2], excluding my_direction
// This function assumes 2D fabric where routers don't forward to themselves
// Examples:
// - EAST router (my_direction=0): WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
// - WEST router (my_direction=1): EAST(0)→0, NORTH(2)→1, SOUTH(3)→2
// - NORTH router (my_direction=2): EAST(0)→0, WEST(1)→1, SOUTH(3)→2
// - SOUTH router (my_direction=3): EAST(0)→0, WEST(1)→1, NORTH(2)→2
constexpr size_t direction_to_compact_index_map[eth_chan_directions::COUNT][eth_chan_directions::COUNT] = {
    {0, 0, 1, 2},  // EAST router -> WEST, NORTH, SOUTH
    {0, 0, 1, 2},  // WEST router -> EAST, NORTH, SOUTH
    {0, 1, 0, 2},  // NORTH router -> EAST, WEST, SOUTH
    {0, 1, 2, 0},  // SOUTH router -> EAST, WEST, NORTH
};

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index() {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index(eth_chan_directions downstream_direction) {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

static constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS_VC0> sender_channels_turn_status =
    get_sender_channel_turn_statuses();

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_0_free_slots_stream_ids = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    0};

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_1_free_slots_stream_ids = {
    vc_1_free_slots_from_downstream_edge_1_stream_id,
    vc_1_free_slots_from_downstream_edge_2_stream_id,
    vc_1_free_slots_from_downstream_edge_3_stream_id,
    0};

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

/////////////////////////////////////////////
//   SENDER SIDE HELPERS
/////////////////////////////////////////////

// Add helper function
template <uint8_t SENDER_CHANNEL_INDEX>
FORCE_INLINE void update_packet_header_before_eth_send(volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
#if defined(FABRIC_2D)
    constexpr bool IS_FORWARDED_TRAFFIC_FROM_ROUTER = SENDER_CHANNEL_INDEX != 0;
    // For VC1 sender channels, we need to adjust the index to map the channel index back to corresponding VC0 channel
    // index for turn status checking.
    constexpr bool IS_TURN = SENDER_CHANNEL_INDEX < MAX_NUM_SENDER_CHANNELS_VC0
                                 ? sender_channels_turn_status[SENDER_CHANNEL_INDEX]
                                 : sender_channels_turn_status[SENDER_CHANNEL_INDEX - MAX_NUM_SENDER_CHANNELS_VC0 + 1];
    static_assert(
        my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST ||
        my_direction == eth_chan_directions::NORTH || my_direction == eth_chan_directions::SOUTH);
    static_assert(
        is_spine_direction(eth_chan_directions::NORTH) || is_spine_direction(eth_chan_directions::SOUTH),
        "Only spine direction of NORTH and SOUTH is supported with this code. If additional spine directions are being "
        "added, please update the code below to support them.");
    if constexpr (IS_FORWARDED_TRAFFIC_FROM_ROUTER) {
        ROUTING_FIELDS_TYPE cached_routing_fields;
        cached_routing_fields.value = packet_header->routing_fields.value;

        if constexpr (IS_TURN) {
            if constexpr (my_direction == eth_chan_directions::EAST) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            } else {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
        } else {
            cached_routing_fields.value = cached_routing_fields.value + 1;
        }
        packet_header->routing_fields.value = cached_routing_fields.value;
    }
#endif
}

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename ReceiverChannelT>
FORCE_INLINE void send_next_data(
    SenderChannelT& sender_buffer_channel,
    WorkerInterfaceT& sender_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    ReceiverChannelT& receiver_buffer_channel,
    PerfTelemetryRecorder& perf_telemetry_recorder) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;

    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    auto dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    if constexpr (!skip_src_ch_id_update) {
        pkt_header->src_ch_id = sender_channel_index;
    }

    if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        while (internal_::eth_txq_is_busy(sender_txq_id)) {
        };
    }
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    // Note: We can only advance to the next buffer index if we have fully completed the send (both the payload and sync
    // messages)
    sender_worker_interface.template update_write_counter_for_send<SKIP_CONNECTION_LIVENESS_CHECK>();

    // Advance receiver buffer pointers
    outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_index();
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.advance_to_next_cached_buffer_slot_addr();
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;

    record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);

    while (internal_::eth_txq_is_busy(sender_txq_id)) {
    };
    remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(packets_to_forward);
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////


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

template <typename DownstreamSenderVC0T, eth_chan_directions DIRECTION>
FORCE_INLINE bool check_downstream_has_space(
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0) {
    if constexpr (DIRECTION == my_direction) {
        return true;
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index(DIRECTION);
        return downstream_edm_interfaces_vc0[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <typename DownstreamSenderVC0T, typename LocalRelayInterfaceT, eth_chan_directions DIRECTION>
FORCE_INLINE bool check_downstream_has_space(
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
    LocalRelayInterfaceT& local_relay_interface) {
    if constexpr (DIRECTION == my_direction) {
        if constexpr (udm_mode) {
            return local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
        } else {
            return true;
        }
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index<DIRECTION>();
        return downstream_edm_interfaces_vc0[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <typename DownstreamSenderVC0T, typename LocalRelayInterfaceT, eth_chan_directions... DIRECTIONS>
FORCE_INLINE bool downstreams_have_space(
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
    LocalRelayInterfaceT& local_relay_interface) {
    return (
        ... && check_downstream_has_space<DownstreamSenderVC0T, LocalRelayInterfaceT, DIRECTIONS>(
                   downstream_edm_interfaces_vc0, local_relay_interface));
}

#ifdef FABRIC_2D
template <typename DownstreamSenderVC0T, typename LocalRelayInterfaceT>
FORCE_INLINE __attribute__((optimize("jump-tables"))) bool can_forward_packet_completely(
    uint32_t hop_cmd,
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
    LocalRelayInterfaceT& local_relay_interface) {
    bool ret_val = false;

    using eth_chan_directions::EAST;
    using eth_chan_directions::NORTH;
    using eth_chan_directions::SOUTH;
    using eth_chan_directions::WEST;

    switch (hop_cmd) {
        case MeshRoutingFields::NOOP: break;
        case MeshRoutingFields::FORWARD_EAST:
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::FORWARD_WEST:
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, WEST>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_EW:
            // Line Mcast East<->West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, WEST>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::FORWARD_NORTH:
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, NORTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::FORWARD_SOUTH:
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NS:
            // Line Mcast North<->South
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, NORTH, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSEW:
            // 2D Mcast Trunk: North<->South
            // 2D Mcast Branch: East and West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, WEST, NORTH, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSE:
            // 2D Mcast Trunk: North<->South
            // 2D Mcast Branch: East
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, NORTH, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSW:
            // 2D Mcast Trunk: North<->South
            // 2D Mcast Branch: West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, WEST, NORTH, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SEW:
            // 2D Mcast Trunk: Last hop North
            // 2D Mcast Branch: East and West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, WEST, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NEW:
            // 2D Mcast Trunk: Last hop South
            // 2D Mcast Branch: East and West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, WEST, NORTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SE:
            // 2D Mcast Trunk: Last hop North
            // 2D Mcast Branch: East
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SW:
            // 2D Mcast Trunk: Last hop North
            // 2D Mcast Branch: West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, WEST, SOUTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NE:
            // 2D Mcast Trunk: Last hop South
            // 2D Mcast Branch: East
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, EAST, NORTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NW:
            // 2D Mcast Trunk: Last hop South
            // 2D Mcast Branch: West
            ret_val = downstreams_have_space<DownstreamSenderVC0T, LocalRelayInterfaceT, WEST, NORTH>(
                downstream_edm_interfaces_vc0, local_relay_interface);
            break;
        default: __builtin_unreachable();
    }
    return ret_val;
}

#else

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
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
        }
        if (start_distance_is_terminal_value) {
            execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
        }
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        switch (routing) {
            case LowLatencyFields::WRITE_ONLY:
                execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            case LowLatencyFields::FORWARD_ONLY:
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                break;
            case LowLatencyFields::WRITE_AND_FORWARD:
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            default: {
                ASSERT(false);
            }
        }
    }
}

#endif

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
        execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
    }
}

// !!!WARNING!!! - MAKE SURE CONSUMER HAS SPACE BEFORE CALLING
template <uint8_t rx_channel_id, typename DownstreamSenderVC0T, typename LocalRelayInterfaceT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    __attribute__((optimize("jump-tables"))) void
    receiver_forward_packet(
        tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
        ROUTING_FIELDS_TYPE cached_routing_fields,
        std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
        LocalRelayInterfaceT& local_relay_interface,
        uint8_t transaction_id,
        uint32_t hop_cmd) {
    uint16_t payload_size_bytes = packet_start->payload_size_bytes;

    using eth_chan_directions::EAST;
    using eth_chan_directions::NORTH;
    using eth_chan_directions::SOUTH;
    using eth_chan_directions::WEST;

    switch (hop_cmd) {
        case MeshRoutingFields::NOOP: break;
        case MeshRoutingFields::FORWARD_EAST:
            if constexpr (my_direction == EAST) {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::FORWARD_WEST:
            if constexpr (my_direction == WEST) {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_EW:
            if constexpr (my_direction == WEST) {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            forward_to_local_destination<rx_channel_id>(
                local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            break;
        case MeshRoutingFields::FORWARD_NORTH:
            if constexpr (my_direction == NORTH) {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::FORWARD_SOUTH:
            if constexpr (my_direction == SOUTH) {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NS:
            if constexpr (my_direction == SOUTH) {
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            forward_to_local_destination<rx_channel_id>(
                local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSEW:
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.value++;
            }
            if constexpr (my_direction == SOUTH) {
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            forward_to_local_destination<rx_channel_id>(
                local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSE:
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.value++;
            }
            if constexpr (my_direction == SOUTH) {
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            forward_to_local_destination<rx_channel_id>(
                local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NSW:
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.value++;
            }
            if constexpr (my_direction == SOUTH) {
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            forward_to_local_destination<rx_channel_id>(
                local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NEW:
            if constexpr (my_direction == SOUTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SEW:
            if constexpr (my_direction == NORTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NE:
            if constexpr (my_direction == SOUTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_NW:
            if constexpr (my_direction == SOUTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<NORTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SE:
            if constexpr (my_direction == NORTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_east_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<EAST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        case MeshRoutingFields::WRITE_AND_FORWARD_SW:
            if constexpr (my_direction == NORTH) {
                if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                    cached_routing_fields.value++;
                }
                constexpr auto edm_index = get_downstream_edm_interface_index<SOUTH>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            } else {
                forward_to_local_destination<rx_channel_id>(
                    local_relay_interface, packet_start, payload_size_bytes, transaction_id);
            }
            if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
                cached_routing_fields.hop_index = cached_routing_fields.branch_west_offset;
            }
            {
                constexpr auto edm_index = get_downstream_edm_interface_index<WEST>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false, !UPDATE_PKT_HDR_ON_RX_CH>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0[edm_index],
                    transaction_id);
            }
            break;
        default: __builtin_unreachable();
    }
}
#endif

template <typename EdmChannelWorkerIFs>
FORCE_INLINE void establish_edm_connection(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface, uint32_t stream_id) {
    local_sender_channel_worker_interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
}

bool any_sender_channels_active(
    const std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        if (get_ptr_val(local_sender_channel_free_slots_stream_ids[i]) !=
            static_cast<int32_t>(SENDER_NUM_BUFFERS_ARRAY[i])) {
            return true;
        }
    }
    return false;
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_telemetry(
    const std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids_ordered,
    bool tx_progress,
    bool rx_progress,
    LocalTelemetryT& local_fabric_telemetry,
    volatile tt_l1_ptr LocalTelemetryT* fabric_telemetry) {
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
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename ReceiverChannelT,
    typename LocalTelemetryT>
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
        SenderChannelFromReceiverCredits& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry) {
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

    bool receiver_has_space_for_packet;
    if constexpr (use_bubble_flow_control) {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.num_free_slots >=
                                        BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    }
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }
    if (can_send) {
        did_something = true;
        progress = true;

        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        if constexpr (!UPDATE_PKT_HDR_ON_RX_CH) {
            update_packet_header_before_eth_send<sender_channel_index>(pkt_header);
        }
        send_next_data<sender_channel_index, to_receiver_pkts_sent_id, SKIP_CONNECTION_LIVENESS_CHECK>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            remote_receiver_channel,
            perf_telemetry_recorder);
        // Update local TX counters: split responsibility in multi-ERISC mode
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check =
        sender_channel_from_receiver_credits.template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_channel_from_receiver_credits.increment_num_processed_completions(completions_since_last_check);

        // When first level ack is enabled, then credits can be sent to upstream workers as soon as we see
        // the ack, we don't need to wait for the completion from receiver. Therefore, only when we have
        // first level ack disabled will we send credits to workers on receipt of completion acknowledgements.
        if constexpr (!enable_first_level_ack) {
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, completions_since_last_check, channel_connection_established);
        }
    }

    // Process ACKs from receiver
    // ACKs are processed second to avoid any sort of races. If we process acks second,
    // we are guaranteed to see equal to or greater the number of acks than completions
    if constexpr (enable_first_level_ack) {
        auto acks_since_last_check = sender_channel_from_receiver_credits.template get_num_unprocessed_acks_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        if (acks_since_last_check > 0) {
            sender_channel_from_receiver_credits.increment_num_processed_acks(acks_since_last_check);
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, acks_since_last_check, channel_connection_established);
        }
    }

    if constexpr (!SKIP_CONNECTION_LIVENESS_CHECK) {
        auto check_connection_status =
            !channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request();
        if (check_connection_status) {
            check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                local_sender_channel_worker_interface,
                channel_connection_established,
                sender_channel_free_slots_stream_id);
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
        std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry) {
    if constexpr (is_sender_channel_serviced[sender_channel_index]) {
        // the cache is invalidated here because the channel will read some
        // L1 locations to see if it can make progress
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

        return run_sender_channel_step_impl<
            sender_channel_index,
            to_receiver_packets_sent_streams[VC_RECEIVER_CHANNEL],
            sender_ch_live_check_skip[sender_channel_index],
            enable_first_level_ack>(
            local_sender_channels.template get<sender_channel_index>(),
            local_sender_channel_worker_interfaces.template get<sender_channel_index>(),
            outbound_to_receiver_channel_pointers,
            remote_receiver_channels.template get<VC_RECEIVER_CHANNEL>(),
            channel_connection_established[sender_channel_index],
            local_sender_channel_free_slots_stream_ids[sender_channel_index],
            sender_channel_from_receiver_credits[sender_channel_index],
            perf_telemetry_recorder,
            local_fabric_telemetry);
    }
    return false;
}

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename DownstreamSenderVC0T,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step_impl(
    ReceiverChannelBufferT& local_receiver_channel,
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    LocalTelemetryT& local_fabric_telemetry) {
    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();

    bool unwritten_packets;
    if constexpr (enable_first_level_ack) {
        auto& ack_counter = receiver_channel_pointers.ack_counter;
        bool pkts_received = pkts_received_since_last_check > 0;
        bool can_send_ack = pkts_received;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_ack = can_send_ack && !internal_::eth_txq_is_busy(receiver_txq_id);
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

            receiver_send_received_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            ack_counter.increment();
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
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        ROUTING_FIELDS_TYPE cached_routing_fields;
#if !defined(FABRIC_2D) || !defined(DYNAMIC_ROUTING_ENABLED)
        cached_routing_fields = packet_header->routing_fields;
#endif
        if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
            receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
        }
        uint32_t hop_cmd;
        bool can_send_to_all_local_chip_receivers;
        if constexpr (is_2d_fabric) {
            // read in the hop command from route buffer.
            // Hop command is 4 bits. Each of the 4 bits signal one of the 4 possible outcomes for a packet.
            // [0]->Forward East
            // [1]->Forward West
            // [2]->Forward North
            // [3]->Forward South
            // The hop command (4-bits) gets decoded as a local write and/or forward to the "other" 3 directions.
            // Other 3 directions depend on the direction of fabric router.
            // For example, a router that is connected West can write locally or forard East, North or South.
            // A local write is encoded by setting the bit corresponding to fabric router's own direction to 1.
            // For a West facing fabric router:
            //  - Hop command of [0010] instructs fabric router to write the packet locally.
            //  - Hop command of [0011] instructs fabric router to write the packet locally AND forward East (a line
            //  mcast)
#if defined(FABRIC_2D)
            // need this ifdef since the packet header for 1D does not have router_buffer field in it.
            hop_cmd = get_cmd_with_mesh_boundary_adjustment(packet_header, cached_routing_fields, routing_table);
            can_send_to_all_local_chip_receivers =
                can_forward_packet_completely(hop_cmd, downstream_edm_interfaces_vc0, local_relay_interface);
#endif
        } else {
#ifndef FABRIC_2D
            can_send_to_all_local_chip_receivers =
                can_forward_packet_completely(cached_routing_fields, downstream_edm_interfaces_vc0[receiver_channel]);
#endif
        }
        if constexpr (enable_trid_flush_check_on_noc_txn) {
            bool trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
            can_send_to_all_local_chip_receivers &= trid_flushed;
        }
        if (can_send_to_all_local_chip_receivers) {
            did_something = true;
            progress = true;
            // Count RX bytes/packets (header + payload) when consuming a packet from receiver buffer
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                receiver_buffer_index);
            if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                receiver_forward_packet<receiver_channel>(
                    packet_header,
                    cached_routing_fields,
                    downstream_edm_interfaces_vc0,
                    local_relay_interface,
                    trid,
                    hop_cmd);
#endif
            } else {
#ifndef FABRIC_2D
                receiver_forward_packet<receiver_channel>(
                    packet_header, cached_routing_fields, downstream_edm_interfaces_vc0[0], trid);
#endif
            }
            wr_sent_counter.increment();
            // decrement the to_receiver_pkts_sent_id stream register by 1 since current packet has been processed.
            if constexpr (!enable_first_level_ack) {
                increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            }
        }
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
            unsent_completions = unsent_completions && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (unsent_completions) {
            // completion ptr incremented in callee
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender,
                receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
            completion_counter.increment();
        }
    } else {
        // flush and completion are fused, so we only need to update one of the counters
        // update completion since other parts of the code check against completion
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        // Currently unclear if it's better to loop here or not...
        bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
        auto receiver_buffer_index = completion_counter.get_buffer_index();
        bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        bool can_send_completion = unflushed_writes && next_trid_flushed;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (can_send_completion) {
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
        }
    }
    return progress;
};

template <
    uint8_t receiver_channel,
    bool enable_first_level_ack,
    typename DownstreamSenderVC0T,
    typename LocalRelayInterfaceT,
    typename EthReceiverChannels,
    typename WriteTridTracker,
    typename ReceiverChannelPointersT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step(
    EthReceiverChannels& local_receiver_channels,
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_interfaces_vc0,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    std::array<ReceiverChannelResponseCreditSender, NUM_RECEIVER_CHANNELS>& receiver_channel_response_credit_senders,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    LocalTelemetryT& local_fabric_telemetry) {
    if constexpr (is_receiver_channel_serviced[receiver_channel]) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        return run_receiver_channel_step_impl<
            receiver_channel,
            to_receiver_packets_sent_streams[receiver_channel],
            enable_first_level_ack,
            WriteTridTracker,
            decltype(local_receiver_channels.template get<receiver_channel>()),
            ReceiverChannelPointersT,
            DownstreamSenderVC0T,
            LocalRelayInterfaceT>(
            local_receiver_channels.template get<receiver_channel>(),
            downstream_edm_interfaces_vc0,
            local_relay_interface,
            receiver_channel_pointers,
            receiver_channel_trid_tracker,
            port_direction_table,
            receiver_channel_response_credit_senders[receiver_channel],
            routing_table,
            local_fabric_telemetry);
    }
    return false;
}

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
    size_t NUM_SENDER_CHANNELS,
    typename EthSenderChannels,
    typename EthReceiverChannels,
    typename RemoteEthReceiverChannels,
    typename EdmChannelWorkerIFs,
    typename TransactionIdTrackerCH0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    typename TransactionIdTrackerCH1
#endif  // FABRIC_2D_VC1_ACTIVE
    >
FORCE_INLINE void run_fabric_edm_main_loop(
    EthReceiverChannels& local_receiver_channels,
    EthSenderChannels& local_sender_channels,
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    std::array<DownstreamSenderVC0T, NUM_DOWNSTREAM_SENDERS_VC0>& downstream_edm_noc_interfaces_vc0,
    std::array<DownstreamSenderVC1T, NUM_DOWNSTREAM_SENDERS_VC1>& downstream_edm_noc_interfaces_vc1,
    LocalRelayInterfaceT& local_relay_interface,
    RemoteEthReceiverChannels& remote_receiver_channels,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    TransactionIdTrackerCH0& receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
    TransactionIdTrackerCH1& receiver_channel_1_trid_tracker,
#endif  // FABRIC_2D_VC1_ACTIVE
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    size_t did_nothing_count = 0;
    using FabricTelemetryT = FabricTelemetry;
    FabricTelemetryT local_fabric_telemetry{};
    auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetryT*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);
    *termination_signal_ptr = tt::tt_fabric::TerminationSignal::KEEP_RUNNING;

    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    tt::tt_fabric::routing_l1_info_t routing_table = *routing_table_l1;

    // May want to promote to part of the handshake but for now we just initialize in this standalone way
    // TODO: flatten all of these arrays into a single object (one array lookup) OR
    //       (probably better) pack most of these into single words (e.g. we could hold a read, write, and ackptr in a
    //       single word) this way - especially if power of 2 wraps, we can handle both channels literally at once with
    //       math ops on single individual words (or half words)
    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, REMOTE_RECEIVER_NUM_BUFFERS_ARRAY>::make();
    // Workaround the perf regression in RingAsLinear test.
    auto outbound_to_receiver_channel_pointer_ch0 =
        outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers = ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    // Workaround the perf regression in RingAsLinear test.
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

#if defined(FABRIC_2D_VC1_ACTIVE)
    // VC1 receiver channel pointer for inter-mesh routing
    auto outbound_to_receiver_channel_pointer_ch1 =
        outbound_to_receiver_channel_pointers.template get<VC1_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch1 = receiver_channel_pointers.template get<1>();
    receiver_channel_pointers_ch1.reset();
#endif  // FABRIC_2D_VC1_ACTIVE

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

    uint64_t loop_start_cycles;
    while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        did_something = false;

        uint32_t tx_progress = 0;
        uint32_t rx_progress = 0;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            loop_start_cycles = get_timestamp();
        }

        if constexpr (is_sender_channel_serviced[0]) {
            open_perf_recording_window(inner_loop_perf_telemetry_collector);
        }

        for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
            // Capture these to see if we made progress

            // There are some cases, mainly for performance, where we don't want to switch between sender channels
            // so we interoduce this to provide finer grain control over when we disable the automatic switching
            tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 0, ENABLE_FIRST_LEVEL_ACK_VC0>(
                local_sender_channels,
                local_sender_channel_worker_interfaces,
                outbound_to_receiver_channel_pointer_ch0,
                remote_receiver_channels,
                channel_connection_established,
                local_sender_channel_free_slots_stream_ids,
                sender_channel_from_receiver_credits,
                inner_loop_perf_telemetry_collector,
                local_fabric_telemetry);
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
            // Inter-mesh routers receive neighbor mesh's locally generated traffic on VC0.
            // This VC0 traffic needs to be forwarded over VC1 in the receiving mesh.
            rx_progress |= run_receiver_channel_step<
                0,
                ENABLE_FIRST_LEVEL_ACK_VC0,
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
                local_fabric_telemetry);
#else
            rx_progress |= run_receiver_channel_step<
                0,
                ENABLE_FIRST_LEVEL_ACK_VC0,
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
                local_fabric_telemetry);
#endif
            tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 1, ENABLE_FIRST_LEVEL_ACK_VC0>(
                local_sender_channels,
                local_sender_channel_worker_interfaces,
                outbound_to_receiver_channel_pointer_ch0,
                remote_receiver_channels,
                channel_connection_established,
                local_sender_channel_free_slots_stream_ids,
                sender_channel_from_receiver_credits,
                inner_loop_perf_telemetry_collector,
                local_fabric_telemetry);
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
                    local_fabric_telemetry);
                tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 3, ENABLE_FIRST_LEVEL_ACK_VC0>(
                    local_sender_channels,
                    local_sender_channel_worker_interfaces,
                    outbound_to_receiver_channel_pointer_ch0,
                    remote_receiver_channels,
                    channel_connection_established,
                    local_sender_channel_free_slots_stream_ids,
                    sender_channel_from_receiver_credits,
                    inner_loop_perf_telemetry_collector,
                    local_fabric_telemetry);
#if defined(FABRIC_2D_VC1_SERVICED)
                tx_progress |= run_sender_channel_step<VC1_RECEIVER_CHANNEL, 4, ENABLE_FIRST_LEVEL_ACK_VC1>(
                    local_sender_channels,
                    local_sender_channel_worker_interfaces,
                    outbound_to_receiver_channel_pointer_ch1,
                    remote_receiver_channels,
                    channel_connection_established,
                    local_sender_channel_free_slots_stream_ids,
                    sender_channel_from_receiver_credits,
                    inner_loop_perf_telemetry_collector,
                    local_fabric_telemetry);
                tx_progress |= run_sender_channel_step<VC1_RECEIVER_CHANNEL, 5, ENABLE_FIRST_LEVEL_ACK_VC1>(
                    local_sender_channels,
                    local_sender_channel_worker_interfaces,
                    outbound_to_receiver_channel_pointer_ch1,
                    remote_receiver_channels,
                    channel_connection_established,
                    local_sender_channel_free_slots_stream_ids,
                    sender_channel_from_receiver_credits,
                    inner_loop_perf_telemetry_collector,
                    local_fabric_telemetry);
                tx_progress |= run_sender_channel_step<VC1_RECEIVER_CHANNEL, 6, ENABLE_FIRST_LEVEL_ACK_VC1>(
                    local_sender_channels,
                    local_sender_channel_worker_interfaces,
                    outbound_to_receiver_channel_pointer_ch1,
                    remote_receiver_channels,
                    channel_connection_established,
                    local_sender_channel_free_slots_stream_ids,
                    sender_channel_from_receiver_credits,
                    inner_loop_perf_telemetry_collector,
                    local_fabric_telemetry);
                rx_progress |= run_receiver_channel_step<
                    1,
                    ENABLE_FIRST_LEVEL_ACK_VC1,
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
                    local_fabric_telemetry);
#endif  // FABRIC_2D_VC1_SERVICED
            }
        }

        // Compute idle conditions and update heartbeats in one helper
        if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, rx_progress, local_fabric_telemetry);
            }
            update_telemetry(
                local_sender_channel_free_slots_stream_ids,
                tx_progress,
                rx_progress,
                local_fabric_telemetry,
                fabric_telemetry);
        }

        if constexpr (enable_context_switch) {
            // shouldn't do noc counter sync since we are not incrementing them
            if constexpr (IDLE_CONTEXT_SWITCHING) {
                if (did_something) {
                    did_nothing_count = 0;
                } else {
                    if (did_nothing_count++ > SWITCH_INTERVAL) {
                        did_nothing_count = 0;
                        run_routing_without_noc_sync();
                    }
                }
            } else {
                if (did_nothing_count++ > SWITCH_INTERVAL) {
                    did_nothing_count = 0;
                    run_routing_without_noc_sync();
                }
            }
        }

        if constexpr (is_sender_channel_serviced[0]) {
            close_perf_recording_window(inner_loop_perf_telemetry_collector);
            if constexpr (perf_telemetry_mode != PerfTelemetryRecorderType::NONE) {
                if (captured_an_event(inner_loop_perf_telemetry_collector) ||
                    any_sender_channels_active(local_sender_channel_free_slots_stream_ids)) {
                    write_perf_recording_window_results(
                        inner_loop_perf_telemetry_collector, local_perf_telemetry_buffer);
                }
            }
        }
    }
}

template <typename EdmChannelWorkerIFs, size_t NUM_SENDER_CHANNELS>
void
#ifdef FABRIC_2D
    __attribute__((noinline))
#endif
    wait_for_static_connection_to_ready(
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
        std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    auto establish_static_connection_from_receiver_side = [&](auto& interface, size_t sender_channel_idx) {
        if (!sender_ch_live_check_skip[sender_channel_idx]) {
            return;
        }
        while (!connect_is_requested(*interface.connection_live_semaphore)) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        establish_edm_connection(interface, local_sender_channel_free_slots_stream_ids[sender_channel_idx]);
    };
    if constexpr (multi_txq_enabled) {
        tuple_for_each_constexpr(
            local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
                if constexpr (is_sender_channel_serviced[idx]) {
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
    return i == 0 ? 0 : SENDER_NUM_BUFFERS_ARRAY[i];
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
        }(std::make_index_sequence<5>{});  // Indices 0-4 map to channels 2-6
    }
#endif
}

// copy the sender_channel_free_slots_stream_ids (in L1) to local memory for performance.
template <size_t NUM_SENDER_CHANNELS>
void populate_local_sender_channel_free_slots_stream_id_ordered_map(
    uint32_t has_downstream_edm_vc0_buffer_connection,
    std::array<uint32_t, NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_free_slots_stream_ids[i] = sender_channel_free_slots_stream_ids[i];
    }
}

constexpr bool IS_TEARDOWN_MASTER() { return MY_ERISC_ID == 0; }

void wait_for_other_local_erisc() {
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

FORCE_INLINE void teardown(
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
) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (is_receiver_channel_serviced[0]) {
        receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();
    }
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (is_receiver_channel_serviced[1]) {
        receiver_channel_1_trid_tracker.all_buffer_slot_transactions_acked();
    }
#endif

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

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (IS_TEARDOWN_MASTER()) {
        *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;
    }
}

void initialize_state_for_txq1_active_mode() {
    eth_enable_packet_mode(receiver_txq_id);
    for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)[i] = 0;
    }
    eth_txq_reg_write(receiver_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
}
void initialize_state_for_txq1_active_mode_sender_side() {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address)[i] = 0;
    }
}

void kernel_main() {
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    asm volatile("nop");
    static_assert(
        receiver_txq_id == sender_txq_id || receiver_txq_id == 1,
        "For multi-txq mode, the only currently supported configuration is sender_txq_id=0 and receiver_txq_id=1");
    if constexpr (receiver_txq_id != sender_txq_id) {
        constexpr bool is_erisc_that_sets_up_second_txq = is_receiver_channel_serviced[0];
        if constexpr (is_erisc_that_sets_up_second_txq) {
            initialize_state_for_txq1_active_mode();
        }
        if constexpr (is_sender_channel_serviced[0]) {
            initialize_state_for_txq1_active_mode_sender_side();
        }
    }
    POSTCODE(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);

    //
    // COMMON CT ARGS (not specific to sender or receiver)
    //

    // Initialize stream register state for credit management across the Ethernet link.
    // We make sure to do this before we handshake to guarantee that the registers are
    // initialized before the other side has any possibility of modifying them.
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_ptr_val<to_sender_packets_acked_streams[0]>(0);
    init_ptr_val<to_sender_packets_acked_streams[1]>(0);
    init_ptr_val<to_sender_packets_completed_streams[0]>(0);
    init_ptr_val<to_sender_packets_completed_streams[1]>(0);
    // The first sender channel in the array is always for the transient/worker connection
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);  // LOCAL WORKER
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);  // Compact index 0

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

    if constexpr (is_2d_fabric) {
        init_ptr_val<to_receiver_packets_sent_streams[1]>(0);
        init_ptr_val<to_sender_packets_acked_streams[2]>(0);
        init_ptr_val<to_sender_packets_acked_streams[3]>(0);

        // Initialize completion streams and sender channel free slots for channels 2-7 using compile-time loop
        // SENDER_NUM_BUFFERS_ARRAY[] is sized to NUM_SENDER_CHANNELS, which is the number of used sender channels.
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 init_ptr_val<to_sender_packets_completed_streams[Is + 2]>(0);
                 if constexpr (NUM_SENDER_CHANNELS > (Is + 2)) {
                     init_ptr_val<sender_channel_free_slots_stream_ids[Is + 2]>(SENDER_NUM_BUFFERS_ARRAY[Is + 2]);
                 }
             }()),
             ...);
        }(std::make_index_sequence<6>{});
    }

    POSTCODE(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);

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
    const size_t local_sender_channel_0_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_1_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_2_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_3_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_4_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_5_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_6_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_7_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_0_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_1_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_2_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_3_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_4_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_5_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_6_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_7_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);

    // downstream EDM VC0 connection info
    const auto has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++);

    // For 2D: read 3 buffer base addresses, NOC coords, and registration addresses (one per compact index)
    // For 1D: reads as 1D and only uses first element
#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    const auto downstream_edm_vc0_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
#endif

    const auto downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_location_info_addresses;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_index_semaphore_addresses;
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
    const auto downstream_edm_vc0_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_buffer_index_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
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

    // Read MAX_NUM_SENDER_CHANNELS teardown semaphores (host packs builder_config::num_max_sender_channels = 8)
    const auto my_sem_for_teardown_from_edm_0 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_1 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_2 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_3 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_4 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_5 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_6 = get_arg_val<uint32_t>(arg_idx++);
    const auto my_sem_for_teardown_from_edm_7 = get_arg_val<uint32_t>(arg_idx++);

    ////////////////////////
    // Sender runtime args
    ////////////////////////
    // Read MAX_NUM_SENDER_CHANNELS sender worker semaphore pointers (host packs builder_config::num_max_sender_channels
    // = 8)
    auto sender0_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender1_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender2_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender3_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender4_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender5_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender6_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    auto sender7_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));

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

    const size_t local_sender_channel_0_connection_buffer_index_addr =
        local_sender_channel_0_connection_buffer_index_id;
    //  initialize the statically allocated "semaphores"
    if constexpr (is_sender_channel_serviced[0]) {
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_semaphore_addr) = 0;
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_buffer_index_addr) = 0;
        *sender0_worker_semaphore_ptr = 0;
    }
    if constexpr (is_sender_channel_serviced[1]) {
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_1_connection_semaphore_addr) = 0;
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_1_connection_buffer_index_id) = 0;
        *sender1_worker_semaphore_ptr = 0;
    }
    if constexpr (is_sender_channel_serviced[2]) {
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_2_connection_semaphore_addr) = 0;
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_2_connection_buffer_index_id) = 0;
        *sender2_worker_semaphore_ptr = 0;
    }
    if constexpr (is_2d_fabric) {
        if constexpr (is_sender_channel_serviced[3]) {
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_3_connection_semaphore_addr) = 0;
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_3_connection_buffer_index_id) = 0;
            *sender3_worker_semaphore_ptr = 0;
        }
        if constexpr (is_sender_channel_serviced[4]) {
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_4_connection_semaphore_addr) = 0;
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_4_connection_buffer_index_id) = 0;
            *sender4_worker_semaphore_ptr = 0;
        }
        if constexpr (is_sender_channel_serviced[5]) {
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_5_connection_semaphore_addr) = 0;
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_5_connection_buffer_index_id) = 0;
            *sender5_worker_semaphore_ptr = 0;
        }
        if constexpr (is_sender_channel_serviced[6]) {
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_6_connection_semaphore_addr) = 0;
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_6_connection_buffer_index_id) = 0;
            *sender6_worker_semaphore_ptr = 0;
        }
        if constexpr (is_sender_channel_serviced[7]) {
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_7_connection_semaphore_addr) = 0;
            *reinterpret_cast<volatile uint32_t*>(local_sender_channel_7_connection_buffer_index_id) = 0;
            *sender7_worker_semaphore_ptr = 0;
        }
    }
    asm volatile("nop");

    POSTCODE(tt::tt_fabric::EDMStatus::STARTED);
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
    std::array<uint32_t, NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids;
    // std::array<uint32_t, NUM_SENDER_CHANNELS == 1 ? 2 : NUM_SENDER_CHANNELS>
    // local_sender_channel_free_slots_stream_ids;

    const auto& local_sem_for_teardown_from_downstream_edm =
        take_first_n_elements<NUM_DOWNSTREAM_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                my_sem_for_teardown_from_edm_0,
                my_sem_for_teardown_from_edm_1,
                my_sem_for_teardown_from_edm_2,
                my_sem_for_teardown_from_edm_3,
                my_sem_for_teardown_from_edm_4,
                my_sem_for_teardown_from_edm_5,
                my_sem_for_teardown_from_edm_6,
                my_sem_for_teardown_from_edm_7,
            });

    // create the remote receiver channel buffers using multi-pool system
    auto remote_receiver_channels = tt::tt_fabric::MultiPoolEthChannelBuffers<
        PACKET_HEADER_TYPE,
        eth_remote_channel_pools_args,
        REMOTE_RECEIVER_TO_POOL_TYPE,
        REMOTE_RECEIVER_TO_POOL_IDX>::make();

    auto local_receiver_channels =
        tt::tt_fabric::MultiPoolEthChannelBuffers<
            PACKET_HEADER_TYPE,
            channel_pools_args,
            RECEIVER_TO_POOL_TYPE,
            RECEIVER_TO_POOL_IDX
        >::make();

    auto local_sender_channels = tt::tt_fabric::MultiPoolSenderEthChannelBuffers<
        PACKET_HEADER_TYPE,
        channel_pools_args,
        SENDER_TO_POOL_TYPE,
        SENDER_TO_POOL_IDX>::make();

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_channel_0_connection_semaphore_addr,
                local_sender_channel_1_connection_semaphore_addr,
                local_sender_channel_2_connection_semaphore_addr,
                local_sender_channel_3_connection_semaphore_addr,
                local_sender_channel_4_connection_semaphore_addr,
                local_sender_channel_5_connection_semaphore_addr,
                local_sender_channel_6_connection_semaphore_addr,
                local_sender_channel_7_connection_semaphore_addr});
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
                local_sender_channel_7_connection_info_addr});

    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }
    // create the sender channel worker interfaces with input array of number of buffers
    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);

    // TODO: change to TMP.
    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>, NUM_DOWNSTREAM_SENDERS_VC0>
        downstream_edm_noc_interfaces_vc0;
    populate_local_sender_channel_free_slots_stream_id_ordered_map(
        has_downstream_edm_vc0_buffer_connection, local_sender_channel_free_slots_stream_ids);

    if (has_downstream_edm_vc0_buffer_connection) {
        // Only bit 0 is set for 1D
        // For 2D: 3 bits set for compact indices 0, 1, 2 (excluding router's own direction)
        uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0x7;  // 3-bit mask
        uint32_t compact_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address = local_sem_for_teardown_from_downstream_edm[compact_index];
                // reset the handshake addresses to 0 (this is for router -> router handshake for connections over noc)
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
#if defined(FABRIC_2D)
                auto receiver_channel_free_slots_stream_id = StreamId{vc_0_free_slots_stream_ids[compact_index]};
#else
                auto receiver_channel_free_slots_stream_id = StreamId{vc_0_free_slots_stream_ids[0]};
#endif
                new (&downstream_edm_noc_interfaces_vc0[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>(
                        // persistent_mode -> hardcode to false for 1D because for 1D, EDM -> EDM
                        // connections we must always use semaphore lookup
                        // For 2D, downstream_edm_vc0_semaphore_id is an address.
                        is_persistent_fabric,
                        (downstream_edm_vc0_noc_x >> (compact_index * 8)) & 0xFF,
                        (downstream_edm_vc0_noc_y >> (compact_index * 8)) & 0xFF,
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_base_addresses[compact_index],
#else
                        downstream_edm_vc0_buffer_base_address,
#endif
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC0,
#if defined(FABRIC_2D)
                        // connection handshake address on downstream edm
                        downstream_edm_vc0_worker_registration_ids[compact_index],
                        // worker location info address on downstream edm
                        // written by this interface when it connects to the downstream edm
                        // so that the downstream edm knows who its upstream peer is
                        downstream_edm_vc0_worker_location_info_addresses[compact_index],
#else
                        downstream_edm_vc0_worker_registration_id,
                        downstream_edm_vc0_worker_location_info_address,
#endif
                        channel_buffer_size,
                // Used to park current write pointer value at the downstream edm
                // when this interface disconnects from the downstream edm.
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_index_semaphore_addresses[compact_index],
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
                // Only receiver channel servicing cores should be setting up the noc cmd buf.
                if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                    downstream_edm_noc_interfaces_vc0[compact_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
    }

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>, NUM_DOWNSTREAM_SENDERS_VC1>
        downstream_edm_noc_interfaces_vc1;
#if defined(FABRIC_2D_VC1_ACTIVE)
    if (has_downstream_edm_vc1_buffer_connection) {
        uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0x7;  // 3-bit mask
        uint32_t compact_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address =
                    local_sem_for_teardown_from_downstream_edm[compact_index + NUM_DOWNSTREAM_SENDERS_VC0];
                // reset the handshake addresses to 0 (this is for router -> router handshake for connections over
                // noc)
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
                auto receiver_channel_free_slots_stream_id = StreamId{vc_1_free_slots_stream_ids[compact_index]};
                new (&downstream_edm_noc_interfaces_vc1[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>(
                        is_persistent_fabric,
                        (downstream_edm_vc1_noc_x >> (compact_index * 8)) & 0xFF,
                        (downstream_edm_vc1_noc_y >> (compact_index * 8)) & 0xFF,
                        downstream_edm_vc1_buffer_base_addresses[compact_index],
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC1,
                        downstream_edm_vc1_worker_registration_ids[compact_index],
                        downstream_edm_vc1_worker_location_info_addresses[compact_index],
                        channel_buffer_size,
                        downstream_edm_vc1_buffer_index_semaphore_addresses[compact_index],
                        0,
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        downstream_vc1_noc_interface_buffer_index_local_addr,
                        get_vc1_downstream_sender_channel_free_slots_stream_id(compact_index),
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[1],
                        receiver_channel_forwarding_sync_cmd_buf_ids[1]);
                // Only receiver channel servicing cores should be setting up the noc cmd buf.
                if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                    downstream_edm_noc_interfaces_vc1[compact_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
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

    POSTCODE(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);

    // initialize the local receiver channel buffers
    local_receiver_channels.init<channel_pools_args>(
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE));

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init<eth_remote_channel_pools_args>(
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE));

    // initialize the local sender channel worker interfaces
    local_sender_channels.init<channel_pools_args>(
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE));

    // initialize the local sender channel worker interfaces
    // Sender channel 0 is always for local worker in the new design
    constexpr auto sender_channel = 0;
    if constexpr (is_sender_channel_serviced[sender_channel]) {
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

#ifdef ARCH_BLACKHOLE
    // A Blackhole hardware bug requires all noc inline writes to be non-posted so we hardcode to false here
    // A more detailed description can be found in `noc_inline_dw_write` in the `dataflow_api` header file
    constexpr bool use_posted_writes_for_connection_open = false;
#else
    constexpr bool use_posted_writes_for_connection_open = true;
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
            erisc::datamover::handshake::sender_side_handshake(
                handshake_addr, DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        } else {
            erisc::datamover::handshake::receiver_side_handshake(
                handshake_addr, DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        }

        *edm_status_ptr = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;
        asm volatile("nop");

        if constexpr (wait_for_host_signal) {
            if constexpr (is_local_handshake_master) {
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>((uint32_t)edm_local_sync_ptr, num_local_edms - 1);
                // This master sends notification to self for multi risc in single eth core case,
                // This still send to self even though with single risc core case, but no side effects
                constexpr uint32_t exclude_eth_chan = std::numeric_limits<uint32_t>::max();
                notify_subordinate_routers(
                    edm_channels_mask, exclude_eth_chan, (uint32_t)edm_local_sync_ptr, num_local_edms);
            } else {
                notify_master_router(local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr);
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>((uint32_t)edm_local_sync_ptr, num_local_edms);
            }

            *edm_status_ptr = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;

            // 1. All risc cores wait for READY_FOR_TRAFFIC signal
            // 2. All risc cores in master eth core receive signal from host and exits from this wait
            //    Other subordinate risc cores wait for this signal
            // 4. The other subordinate risc cores receive the READY_FOR_TRAFFIC signal and exit from this wait
            wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>((uint32_t)edm_status_ptr, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);

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

    POSTCODE(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);
    // if enable the tensix extension, then before open downstream connection, need to wait for downstream tensix ready
    // for connection.
    if constexpr (num_ds_or_local_tensix_connections) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>((uint32_t)edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections);
    }

    if constexpr (is_2d_fabric) {
        // Helper function to open downstream EDM connections, works for both VC0 and VC1
        auto open_downstream_edm_connections =
            [](auto& downstream_edm_noc_interfaces, uint32_t has_downstream_edm, int receiver_channel_idx) {
                uint32_t edm_index = 0;
                if (is_receiver_channel_serviced[receiver_channel_idx]) {
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
                }
            };

        open_downstream_edm_connections(
            downstream_edm_noc_interfaces_vc0, has_downstream_edm_vc0_buffer_connection & 0x7, 0);
#if defined(FABRIC_2D_VC1_ACTIVE)
        open_downstream_edm_connections(
            downstream_edm_noc_interfaces_vc1, has_downstream_edm_vc1_buffer_connection & 0x7, 1);
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

    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);

    if constexpr (is_receiver_channel_serviced[0] and NUM_ACTIVE_ERISCS > 1) {
        // Two erisc mode requires us to reorder the cmd buf programming/state setting
        // because we need to reshuffle some of our cmd_buf/noc assignments around for
        // just the fabric bringup phase. These calls are also located earlier for the
        // single erisc mode
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0x7;  // 3-bit mask
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
    if constexpr (is_receiver_channel_serviced[1] and NUM_ACTIVE_ERISCS > 1) {
        // Two erisc mode requires us to reorder the cmd buf programming/state setting
        // because we need to reshuffle some of our cmd_buf/noc assignments around for
        // just the fabric bringup phase. These calls are also located earlier for the
        // single erisc mode
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0x7;  // 3-bit mask
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
    std::array<uint8_t, num_eth_ports> port_direction_table;

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

    POSTCODE(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);

    WAYPOINT("FSCW");
    wait_for_static_connection_to_ready(
        local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids);
    WAYPOINT("FSCD");

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE);

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
        port_direction_table,
        local_sender_channel_free_slots_stream_ids);
    WAYPOINT("LPDN");

    // we force these values to a non-zero value so that if we run the fabric back to back,
    // and we can reliably probe from host that this kernel has initialized properly.
    // Sender channel 0 is always for local worker in both 1D and 2D
    *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_semaphore_addr) = 99;
    *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_buffer_index_addr) = 99;
    *sender0_worker_semaphore_ptr = 99;

    // make sure all the noc transactions are acked before re-init the noc counters
    teardown(
        termination_signal_ptr,
        edm_status_ptr,
        receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
        ,
        receiver_channel_1_trid_tracker
#endif
    );

    set_l1_data_cache<false>();
    WAYPOINT("DONE");
}
