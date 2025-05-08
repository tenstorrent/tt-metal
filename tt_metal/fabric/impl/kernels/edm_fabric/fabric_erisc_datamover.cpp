// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/api/tt-metalium/edm_fabric_counters.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"

#include "noc_overlay_parameters.h"
#include "tt_metal/hw/inc/utils/utils.h"

#include <array>
#include <cstddef>
#include <cstdint>

using namespace tt::tt_fabric;

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
by the worker (the EDM is a slave in this protocol).

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

Each EDM channel is built from one or more buffers. Each buffer is the same size and can hold atmost one packet.
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

// This will be an atomic register read to the register
template <uint32_t stream_id>
FORCE_INLINE int32_t get_ptr_val() {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}
FORCE_INLINE int32_t get_ptr_val(uint8_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

// Writing to this register will leverage the built-in stream hardware which will automatically perform an atomic
// increment on the register. This can save precious erisc cycles by offloading a lot of pointer manipulation.
// Additionally, these registers are accessible via eth_reg_write calls which can be used to write a value,
// inline the eth command (without requiring source L1)
template <uint32_t stream_id>
FORCE_INLINE void increment_local_update_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG_FIELD(
        stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, REMOTE_DEST_BUF_WORDS_FREE_INC, val);
}
FORCE_INLINE void increment_local_update_ptr_val(uint8_t stream_id, int32_t val) {
    NOC_STREAM_WRITE_REG_FIELD(
        stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, REMOTE_DEST_BUF_WORDS_FREE_INC, val);
}

template <uint32_t stream_id>
FORCE_INLINE void remote_update_ptr_val(int32_t val) {
    constexpr uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg_no_txq_check(DEFAULT_ETH_TXQ, addr, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}
FORCE_INLINE void remote_update_ptr_val(uint32_t stream_id, int32_t val) {
    const uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg_no_txq_check(DEFAULT_ETH_TXQ, addr, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}

template <uint32_t stream_id>
FORCE_INLINE void init_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, val);
}

/*
 * Tracks receiver channel pointers (from sender side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    BufferIndex remote_receiver_buffer_index{0};
    uint8_t num_free_slots = RECEIVER_NUM_BUFFERS;

    FORCE_INLINE bool has_space_for_packet() const { return num_free_slots; }
};

/*
 * Tracks receiver channel pointers (from receiver side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct ReceiverChannelPointers {
    ChannelCounter<RECEIVER_NUM_BUFFERS> wr_sent_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> wr_flush_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> ack_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> completion_counter;
    std::array<uint8_t, RECEIVER_NUM_BUFFERS> src_chan_ids;

    FORCE_INLINE void set_src_chan_id(BufferIndex buffer_index, uint8_t src_chan_id) {
        src_chan_ids[buffer_index.get()] = src_chan_id;
    }

    FORCE_INLINE uint8_t get_src_chan_id(BufferIndex buffer_index) const { return src_chan_ids[buffer_index.get()]; }
};

struct PacketHeaderRecorder {
    volatile uint32_t* buffer_ptr;
    size_t buffer_n_headers;
    size_t buffer_index;

    PacketHeaderRecorder(volatile uint32_t* buffer_ptr, size_t buffer_n_headers) :
        buffer_ptr(buffer_ptr), buffer_n_headers(buffer_n_headers), buffer_index(0) {}

    void record_packet_header(volatile uint32_t* packet_header_ptr) {
        uint32_t dest_l1_addr = (uint32_t)buffer_ptr + buffer_index * sizeof(PACKET_HEADER_TYPE);
        noc_async_write(
            (uint32_t)packet_header_ptr,
            get_noc_addr(my_x[0], my_y[0], dest_l1_addr),
            sizeof(PACKET_HEADER_TYPE),
            1 - noc_index  // avoid the contention on main noc
        );
        buffer_index++;
        if (buffer_index == buffer_n_headers) {
            buffer_index = 0;
        }
    }
};

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

template <uint8_t SENDER_NUM_BUFFERS, uint8_t RECEIVER_NUM_BUFFERS, uint8_t to_receiver_pkts_sent_id>
FORCE_INLINE void send_next_data(
    tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>& sender_buffer_channel,
    tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>& sender_worker_interface,
    OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& receiver_buffer_channel,
    uint8_t sender_channel_index) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    auto& local_sender_wrptr = sender_worker_interface.local_wrptr;
    auto local_sender_wrptr_buffer_index = local_sender_wrptr.get_buffer_index();

    // TODO: TUNING - experiment with only conditionally breaking the transfer up into multiple packets if we are
    //       a certain threshold less than full packet
    //       we can precompute this value even on host and pass it in so we can get away with a single integer
    //       compare
    //       NOTE: if we always send full packet, then we don't need the second branch below dedicated for
    //             channel sync
    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
        sender_buffer_channel.get_buffer_address(local_sender_wrptr_buffer_index));
    ASSERT(tt::tt_fabric::is_valid(*const_cast<PACKET_HEADER_TYPE*>(pkt_header)));
    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    pkt_header->src_ch_id = sender_channel_index;

    auto src_addr = (uint32_t)pkt_header;
    auto dest_addr = receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index);
    while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
    };
    internal_::eth_send_packet_bytes_unsafe(DEFAULT_ETH_TXQ, src_addr, dest_addr, payload_size_bytes);

    // Note: We can only advance to the next buffer index if we have fully completed the send (both the payload and sync
    // messages)
    local_sender_wrptr.increment();

    // TODO: Put in fn
    remote_receiver_buffer_index =
        BufferIndex{wrap_increment<RECEIVER_NUM_BUFFERS>(remote_receiver_buffer_index.get())};
    remote_receiver_num_free_slots--;

    // update the remote reg
    static constexpr uint32_t words_to_forward = 1;
    while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
    };
    remote_update_ptr_val<to_receiver_pkts_sent_id>(words_to_forward);
}

/////////////////////////////////////////////
//   RECEIVER SIDE HELPERS
/////////////////////////////////////////////

/*
 * Acting the receiver, we are looking at our receiver channel and acking the sender who sent us the latest packet.
 * Doesn't check to see if indeed a new message is available. It's assumed the caller has handled that separately.
 * MUST CHECK !is_eth_txq_busy() before calling
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
FORCE_INLINE void receiver_send_received_ack(
    // currently the pointer is working multiple jobs (ack, completion, read) because we haven't implemented the
    // decoupling of those jobs yet to separate pointrers
    BufferIndex receiver_buffer_index,
    const tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& local_receiver_buffer_channel) {
    // Set the acknowledgement bits
    volatile tt_l1_ptr auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        local_receiver_buffer_channel.get_buffer_address(receiver_buffer_index));
    const auto src_id = pkt_header->src_ch_id;
    while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
    };
    remote_update_ptr_val(to_sender_packets_acked_streams[src_id], 1);
}

// MUST CHECK !is_eth_txq_busy() before calling
FORCE_INLINE void receiver_send_completion_ack(uint8_t src_id) {
    while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
    };
    remote_update_ptr_val(to_sender_packets_completed_streams[src_id], 1);
}

template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields,
    tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>& downstream_edm_interface) {
    // We always check if it is the terminal mcast packet value. We can do this because all unicast packets have the
    // mcast terminal value masked in to the routing field. This simplifies the check here to a single compare.
    bool deliver_locally_only;
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        deliver_locally_only = cached_routing_fields.value == tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        deliver_locally_only = (cached_routing_fields.value & tt::tt_fabric::LowLatencyRoutingFields::FIELD_MASK) ==
                               tt::tt_fabric::LowLatencyRoutingFields::WRITE_ONLY;
    }
    return deliver_locally_only || downstream_edm_interface.edm_has_space_for_packet();
}

// !!!WARNING!!! - MAKE SURE CONSUMER HAS SPACE BEFORE CALLING
template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE void receiver_forward_packet(
    // TODO: have a separate cached copy of the packet header to save some additional L1 loads
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>& downstream_edm_interface,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        // If the packet is a terminal packet, then we can just deliver it locally
        bool start_distance_is_terminal_value =
            (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
            tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        bool not_last_destination_device = cached_routing_fields.value != tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
        if (not_last_destination_device) {
            forward_payload_to_downstream_edm<SENDER_NUM_BUFFERS, enable_ring_support>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
        }
        if (start_distance_is_terminal_value) {
            execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
        }
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        uint32_t routing = cached_routing_fields.value & tt::tt_fabric::LowLatencyRoutingFields::FIELD_MASK;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        switch (routing) {
            case tt::tt_fabric::LowLatencyRoutingFields::WRITE_ONLY:
                execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            case tt::tt_fabric::LowLatencyRoutingFields::FORWARD_ONLY:
                forward_payload_to_downstream_edm<SENDER_NUM_BUFFERS, enable_ring_support>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                break;
            case tt::tt_fabric::LowLatencyRoutingFields::WRITE_AND_FORWARD:
                forward_payload_to_downstream_edm<SENDER_NUM_BUFFERS, enable_ring_support>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            default: ASSERT(false);
        }
    }
}

FORCE_INLINE bool connect_is_requested(uint32_t cached) {
    return cached == tt::tt_fabric::EdmToEdmSender<0>::open_connection_value ||
           cached == tt::tt_fabric::EdmToEdmSender<0>::close_connection_request_value;
}

template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE void establish_connection(
    tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>& local_sender_channel_worker_interface) {
    local_sender_channel_worker_interface.cache_producer_noc_addr();
    if constexpr (enable_first_level_ack) {
        local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
            local_sender_channel_worker_interface.local_ackptr.get_ptr());
    } else {
        local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
            local_sender_channel_worker_interface.local_rdptr.get_ptr());
    }
}

template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE void check_worker_connections(
    tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>& local_sender_channel_worker_interface,
    bool& channel_connection_established) {
    if (!channel_connection_established) {
        // Can get rid of one of these two checks if we duplicate the logic above here in the function
        // and depending on which of the two versions we are in (the connected version or disconnected version)
        // We also check if the interface has a teardown request in case worker
        // 1. opened connection
        // 2. sent of all packets (EDM sender channel was sufficiently empty)
        // 3. closed the connection
        //
        // In such a case like that, we still want to formally teardown the connection to keep things clean
        uint32_t cached = *local_sender_channel_worker_interface.connection_live_semaphore;
        if (connect_is_requested(cached)) {
            // if constexpr (enable_fabric_counters) {
            //     sender_channel_counters->add_connection();
            // }
            channel_connection_established = true;
            establish_connection(local_sender_channel_worker_interface);
        }
    } else if (local_sender_channel_worker_interface.has_worker_teardown_request()) {
        channel_connection_established = false;
        local_sender_channel_worker_interface.template teardown_connection<true>(
            local_sender_channel_worker_interface.local_rdptr.get_ptr());
    }
}

////////////////////////////////////
////////////////////////////////////
//  Main Control Loop
////////////////////////////////////
////////////////////////////////////
template <
    bool enable_packet_header_recording,
    bool enable_fabric_counters,
    uint8_t RECEIVER_NUM_BUFFERS,
    uint8_t SENDER_NUM_BUFFERS,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK>
void run_sender_channel_step(
    tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>& local_sender_channel,
    tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>& local_sender_channel_worker_interface,
    OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& remote_receiver_channel,
    volatile tt::tt_fabric::EdmFabricSenderChannelCounters* sender_channel_counters,
    PacketHeaderRecorder& packet_header_recorder,
    bool& channel_connection_established,
    uint8_t sender_channel_index) {
    // If the receiver has space, and we have one or more packets unsent from producer, then send one
    // TODO: convert to loop to send multiple packets back to back (or support sending multiple packets in one shot)
    //       when moving to stream regs to manage rd/wr ptrs
    // TODO: update to be stream reg based. Initialize to space available and simply check for non-zero
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet = local_sender_channel_worker_interface.has_unsent_payload();
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;
    if constexpr (enable_first_level_ack) {
        bool sender_backpressured_from_sender_side =
            !(local_sender_channel_worker_interface.local_rdptr.distance_behind(
                  local_sender_channel_worker_interface.local_wrptr) < SENDER_NUM_BUFFERS);
        can_send = can_send && !sender_backpressured_from_sender_side;
    }
    if (can_send) {
        did_something = true;
        if constexpr (enable_packet_header_recording) {
            auto packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(local_sender_channel.get_buffer_address(
                local_sender_channel_worker_interface.local_wrptr.get_buffer_index()));
            tt::tt_fabric::validate(*packet_header);
            packet_header_recorder.record_packet_header(reinterpret_cast<volatile uint32_t*>(packet_header));
        }
        send_next_data<SENDER_NUM_BUFFERS, RECEIVER_NUM_BUFFERS, to_receiver_pkts_sent_id>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            remote_receiver_channel,
            sender_channel_index);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_packets_completed_streams[sender_channel_index]);
    if (completions_since_last_check) {
        auto& sender_rdptr = local_sender_channel_worker_interface.local_rdptr;
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_rdptr.increment_n(completions_since_last_check);
        increment_local_update_ptr_val(
            to_sender_packets_completed_streams[sender_channel_index], -completions_since_last_check);
        if constexpr (!enable_first_level_ack) {
            if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
                local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
                    sender_rdptr.get_ptr());
            } else {
                if (channel_connection_established) {
                    local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
                        sender_rdptr.get_ptr());
                }
            }
        }
    }

    // Process ACKs from receiver
    // ACKs are processed second to avoid any sort of races. If we process acks second,
    // we are guaranteed to see equal to or greater the number of acks than completions
    if constexpr (enable_first_level_ack) {
        auto acks_since_last_check = get_ptr_val(to_sender_packets_acked_streams[sender_channel_index]);
        auto& sender_ackptr = local_sender_channel_worker_interface.local_ackptr;
        if (acks_since_last_check > 0) {
            sender_ackptr.increment_n(acks_since_last_check);
            if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
                local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
                    sender_ackptr.get_ptr());
            } else {
                if (channel_connection_established) {
                    local_sender_channel_worker_interface.template update_worker_copy_of_read_ptr<enable_ring_support>(
                        sender_ackptr.get_ptr());
                }
            }
            increment_local_update_ptr_val(
                to_sender_packets_acked_streams[sender_channel_index], -acks_since_last_check);
        }
    }

    if constexpr (!SKIP_CONNECTION_LIVENESS_CHECK) {
        auto check_connection_status =
            !channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request();
        if (check_connection_status) {
            check_worker_connections(local_sender_channel_worker_interface, channel_connection_established);
        }
    }
};

template <
    bool enable_packet_header_recording,
    bool enable_fabric_counters,
    uint8_t RECEIVER_NUM_BUFFERS,
    uint8_t SENDER_NUM_BUFFERS,
    size_t NUM_SENDER_CHANNELS,
    uint8_t to_receiver_pkts_sent_id,
    typename WriteTridTracker>
void run_receiver_channel_step(
    tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& local_receiver_channel,
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>& remote_sender_channels,
    tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>& downstream_edm_interface,
    volatile tt::tt_fabric::EdmFabricReceiverChannelCounters* receiver_channel_counters_ptr,
    ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>& receiver_channel_pointers,
    PacketHeaderRecorder& packet_header_recorder,
    WriteTridTracker& receiver_channel_trid_tracker,
    uint8_t rx_channel_id) {
    auto& ack_counter = receiver_channel_pointers.ack_counter;
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();
    if constexpr (enable_first_level_ack) {
        bool pkts_received = pkts_received_since_last_check > 0;
        ASSERT(receiver_channel_pointers.completion_ptr.distance_behind(ack_counter) < RECEIVER_NUM_BUFFERS);
        if (pkts_received) {
            // currently only support processing one packet at a time, so we only decrement by 1
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            receiver_send_received_ack(ack_counter.get_buffer_index(), local_receiver_channel);
            ack_counter.increment();
        }
    } else {
        increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-pkts_received_since_last_check);
        ack_counter.increment_n(pkts_received_since_last_check);
    }

    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = !wr_sent_counter.is_caught_up_to(ack_counter);
    if (unwritten_packets) {
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        ROUTING_FIELDS_TYPE cached_routing_fields = packet_header->routing_fields;
        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
        bool can_send_to_all_local_chip_receivers =
            can_forward_packet_completely(cached_routing_fields, downstream_edm_interface);
        bool trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        if (can_send_to_all_local_chip_receivers && trid_flushed) {
            did_something = true;
            uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                receiver_buffer_index);
            receiver_forward_packet(
                packet_header, cached_routing_fields, downstream_edm_interface, trid, rx_channel_id);
            wr_sent_counter.increment();
        }
    }

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
        if (unsent_completions) {
            // completion ptr incremented in callee
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
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
        if (can_send_completion) {
            receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
            receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
            completion_counter.increment();
        }
    }
};

/* Termination signal handling*/
FORCE_INLINE bool got_immediate_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE;
}
FORCE_INLINE bool got_graceful_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE;
}
FORCE_INLINE bool got_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return got_immediate_termination_signal(termination_signal_ptr) ||
           got_graceful_termination_signal(termination_signal_ptr);
}

template <
    uint8_t RECEIVER_NUM_BUFFERS,
    size_t NUM_RECEIVER_CHANNELS,
    uint8_t SENDER_NUM_BUFFERS,
    size_t NUM_SENDER_CHANNELS>
bool all_channels_drained(
    std::array<tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS>& local_receiver_channels,
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>& local_sender_channels,
    std::array<tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>&
        local_sender_channel_worker_interfaces,
    std::array<ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS>& receiver_channel_pointers) {
    bool eth_buffers_drained = local_sender_channel_worker_interfaces[0].all_eth_packets_completed() &&
                               local_sender_channel_worker_interfaces[1].all_eth_packets_completed() &&
                               !local_sender_channel_worker_interfaces[0].has_unsent_payload() &&
                               !local_sender_channel_worker_interfaces[1].has_unsent_payload() &&
                               get_ptr_val<to_sender_packets_acked_streams[0]>() == 0 &&
                               get_ptr_val<to_sender_packets_acked_streams[1]>() == 0 &&
                               get_ptr_val<to_sender_packets_completed_streams[0]>() == 0 &&
                               get_ptr_val<to_sender_packets_completed_streams[1]>() == 0;
    // Reeiver 0 enabled
    if constexpr (!dateline_connection) {
        eth_buffers_drained =
            eth_buffers_drained &&
            (get_ptr_val<to_receiver_packets_sent_streams[0]>() == 0 &&
             receiver_channel_pointers[0].completion_counter.is_caught_up_to(receiver_channel_pointers[0].ack_counter));
    }
    // Receiver 1 enabled
    if constexpr (enable_ring_support) {
        eth_buffers_drained =
            eth_buffers_drained &&
            (get_ptr_val<to_receiver_packets_sent_streams[1]>() == 0 &&
             receiver_channel_pointers[1].completion_counter.is_caught_up_to(receiver_channel_pointers[1].ack_counter));
    }
    // Sender 2 enabled
    if constexpr (enable_ring_support && !dateline_connection) {
        eth_buffers_drained =
            eth_buffers_drained && (local_sender_channel_worker_interfaces[2].all_eth_packets_completed() &&
                                    !local_sender_channel_worker_interfaces[2].has_unsent_payload() &&
                                    get_ptr_val<to_sender_packets_acked_streams[2]>() == 0 &&
                                    get_ptr_val<to_sender_packets_completed_streams[2]>() == 0);
    }
    return eth_buffers_drained;
}

/*
 * Main control loop for fabric EDM. Run indefinitely until a termination signal is received
 *
 * Every loop iteration visit a sender channel and the receiver channel. Switch between sender
 * channels every iteration unless it is unsafe/undesirable to do so (e.g. for performance reasons).
 */
template <
    bool enable_packet_header_recording,
    bool enable_fabric_counters,
    uint8_t RECEIVER_NUM_BUFFERS,
    uint8_t NUM_RECEIVER_CHANNELS,
    uint8_t SENDER_NUM_BUFFERS,
    size_t NUM_SENDER_CHANNELS,
    size_t MAX_NUM_SENDER_CHANNELS,
    size_t MAX_NUM_RECEIVER_CHANNELS>
void run_fabric_edm_main_loop(
    std::array<tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS>& local_receiver_channels,
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>& local_sender_channels,
    std::array<tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>&
        local_sender_channel_worker_interfaces,
    std::array<tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>, NUM_USED_RECEIVER_CHANNELS>&
        downstream_edm_noc_interfaces,
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>& remote_sender_channels,
    std::array<tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS>& remote_receiver_channels,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    std::array<volatile tt::tt_fabric::EdmFabricReceiverChannelCounters*, MAX_NUM_RECEIVER_CHANNELS>
        receiver_channel_counters_ptrs,
    std::array<volatile tt::tt_fabric::EdmFabricSenderChannelCounters*, MAX_NUM_SENDER_CHANNELS>
        sender_channel_counters_ptrs,
    std::array<PacketHeaderRecorder, MAX_NUM_RECEIVER_CHANNELS>& receiver_channel_packet_recorders,
    std::array<PacketHeaderRecorder, MAX_NUM_SENDER_CHANNELS>& sender_channel_packet_recorders,
    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS, NUM_TRANSACTION_IDS, 0>& receiver_channel_0_trid_tracker,
    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS, NUM_TRANSACTION_IDS, NUM_TRANSACTION_IDS>&
        receiver_channel_1_trid_tracker) {
    size_t did_nothing_count = 0;
    *termination_signal_ptr = tt::tt_fabric::TerminationSignal::KEEP_RUNNING;

    // May want to promote to part of the handshake but for now we just initialize in this standalone way
    // TODO: flatten all of these arrays into a single object (one array lookup) OR
    //       (probably better) pack most of these into single words (e.g. we could hold a read, write, and ackptr in a
    //       single word) this way - especially if power of 2 wraps, we can handle both channels literally at once with
    //       math ops on single individual words (or half words)
    std::array<OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS>
        outbound_to_receiver_channel_pointers;
    std::array<ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS> receiver_channel_pointers;
    std::array<bool, NUM_SENDER_CHANNELS> channel_connection_established =
        initialize_array<NUM_SENDER_CHANNELS, bool, false>();

    // This value defines the number of loop iterations we perform of the main control sequence before exiting
    // to check for termination and context switch. Removing the these checks from the inner loop can drastically
    // improve performance. The value of 32 was chosen somewhat empirically and then raised up slightly.

    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        bool got_graceful_termination = got_graceful_termination_signal(termination_signal_ptr);
        if (got_graceful_termination) {
            DPRINT << "EDM Graceful termination\n";
            bool all_drained = all_channels_drained<
                RECEIVER_NUM_BUFFERS,
                NUM_RECEIVER_CHANNELS,
                SENDER_NUM_BUFFERS,
                NUM_SENDER_CHANNELS>(
                local_receiver_channels,
                local_sender_channels,
                local_sender_channel_worker_interfaces,
                receiver_channel_pointers);

            if (all_drained) {
                return;
            }
        }
        did_something = false;
        for (size_t i = 0; i < DEFAULT_ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS; i++) {
            // Capture these to see if we made progress

            // There are some cases, mainly for performance, where we don't want to switch between sender channels
            // so we interoduce this to provide finer grain control over when we disable the automatic switching
            run_sender_channel_step<
                enable_packet_header_recording,
                enable_fabric_counters,
                RECEIVER_NUM_BUFFERS,
                SENDER_NUM_BUFFERS,
                to_receiver_packets_sent_streams[VC0_RECEIVER_CHANNEL],
                sender_ch_live_check_skip[0]>(
                local_sender_channels[0],
                local_sender_channel_worker_interfaces[0],
                outbound_to_receiver_channel_pointers[VC0_RECEIVER_CHANNEL],
                remote_receiver_channels[VC0_RECEIVER_CHANNEL],
                sender_channel_counters_ptrs[0],
                sender_channel_packet_recorders[0],
                channel_connection_established[0],
                0);
            if constexpr (!dateline_connection) {
                run_receiver_channel_step<
                    enable_packet_header_recording,
                    enable_fabric_counters,
                    RECEIVER_NUM_BUFFERS,
                    SENDER_NUM_BUFFERS,
                    NUM_SENDER_CHANNELS,
                    to_receiver_packets_sent_streams[0]>(
                    local_receiver_channels[0],
                    remote_sender_channels,
                    downstream_edm_noc_interfaces[0],
                    receiver_channel_counters_ptrs[0],
                    receiver_channel_pointers[0],
                    receiver_channel_packet_recorders[0],
                    receiver_channel_0_trid_tracker,
                    0);
            }
            if constexpr (enable_ring_support) {
                run_receiver_channel_step<
                    enable_packet_header_recording,
                    enable_fabric_counters,
                    RECEIVER_NUM_BUFFERS,
                    SENDER_NUM_BUFFERS,
                    NUM_SENDER_CHANNELS,
                    to_receiver_packets_sent_streams[1]>(
                    local_receiver_channels[VC1_RECEIVER_CHANNEL],
                    remote_sender_channels,
                    downstream_edm_noc_interfaces[1],
                    receiver_channel_counters_ptrs[1],
                    receiver_channel_pointers[1],
                    receiver_channel_packet_recorders[1],
                    receiver_channel_1_trid_tracker,
                    1);
            }

            run_sender_channel_step<
                enable_packet_header_recording,
                enable_fabric_counters,
                RECEIVER_NUM_BUFFERS,
                SENDER_NUM_BUFFERS,
                to_receiver_packets_sent_streams[VC0_RECEIVER_CHANNEL],
                sender_ch_live_check_skip[1]>(
                local_sender_channels[1],
                local_sender_channel_worker_interfaces[1],
                outbound_to_receiver_channel_pointers[VC0_RECEIVER_CHANNEL],
                remote_receiver_channels[VC0_RECEIVER_CHANNEL],
                sender_channel_counters_ptrs[1],
                sender_channel_packet_recorders[1],
                channel_connection_established[1],
                1);
            if constexpr (enable_ring_support && !dateline_connection) {
                run_sender_channel_step<
                    enable_packet_header_recording,
                    enable_fabric_counters,
                    RECEIVER_NUM_BUFFERS,
                    SENDER_NUM_BUFFERS,
                    to_receiver_packets_sent_streams[VC1_RECEIVER_CHANNEL],
                    sender_ch_live_check_skip[2]>(
                    local_sender_channels[2],
                    local_sender_channel_worker_interfaces[2],
                    outbound_to_receiver_channel_pointers[VC1_RECEIVER_CHANNEL],
                    remote_receiver_channels[VC1_RECEIVER_CHANNEL],
                    sender_channel_counters_ptrs[2],
                    sender_channel_packet_recorders[2],
                    channel_connection_established[2],
                    2);
            }

        }

        if (did_something) {
            did_nothing_count = 0;
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                // shouldn't do noc counter sync since we are not incrementing them
                run_routing_without_noc_sync();
            }
        }
    }
    DPRINT << "EDM Terminating\n";
}

template <size_t NUM_SENDER_CHANNELS, uint8_t SENDER_NUM_BUFFERS>
void __attribute__((noinline)) wait_for_static_connection_to_ready(
    std::array<tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>&
        local_sender_channel_worker_interfaces) {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        if (sender_ch_live_check_skip[i]) {
            while (!connect_is_requested(*local_sender_channel_worker_interfaces[i].connection_live_semaphore));
            establish_connection(local_sender_channel_worker_interfaces[i]);
        }
    }
}

template <size_t NUM_SENDER_CHANNELS, uint8_t SENDER_NUM_BUFFERS>
void __attribute__((noinline)) init_local_sender_channel_worker_interfaces(
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_live_semaphore_addresses,
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_info_addresses,
    std::array<tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>&
        local_sender_channel_worker_interfaces,
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_flow_control_semaphores) {
    // manual unrol because previously, going from having this in a loop to unrolling this would
    // lead to a performance regression. Having these unrolled is needed to enable some performance optimizations
    // because setup will differ in that each will be a different type. Keeping them unrolled here let's us
    // stay safe from perf regression due to weirdness of codegen.
    {
        auto connection_live_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_connection_live_semaphore_addresses[0]);
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[0]);
        connection_worker_info_ptr->edm_rdptr = 0;
        new (&local_sender_channel_worker_interfaces[0]) tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>(
            connection_worker_info_ptr,
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_flow_control_semaphores[0]),
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_live_semaphore_ptr),
            sender_channel_ack_cmd_buf_ids[0]);
    }
    {
        auto connection_live_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_connection_live_semaphore_addresses[1]);
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[1]);
        connection_worker_info_ptr->edm_rdptr = 0;
        new (&local_sender_channel_worker_interfaces[1]) tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>(
            connection_worker_info_ptr,
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_flow_control_semaphores[1]),
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_live_semaphore_ptr),
            sender_channel_ack_cmd_buf_ids[1]);
    }
    if constexpr (NUM_SENDER_CHANNELS > 2) {
        {
            auto connection_live_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(
                local_sender_connection_live_semaphore_addresses[2]);
            auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
                local_sender_connection_info_addresses[2]);
            connection_worker_info_ptr->edm_rdptr = 0;
            new (&local_sender_channel_worker_interfaces[2])
                tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>(
                    connection_worker_info_ptr,
                    reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_flow_control_semaphores[2]),
                    reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_live_semaphore_ptr),
                    sender_channel_ack_cmd_buf_ids[2]);
        }
    }
}

void kernel_main() {
    eth_txq_reg_write(DEFAULT_ETH_TXQ, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    //
    // COMMON CT ARGS (not specific to sender or receiver)
    //
    *reinterpret_cast<volatile uint32_t*>(handshake_addr) = 0;
    auto eth_transaction_ack_word_addr = handshake_addr + sizeof(eth_channel_sync_t);

    // Initialize stream register state for credit management across the Ethernet link.
    // We make sure to do this before we handshake to guarantee that the registers are
    // initialized before the other side has any possibility of modifying them.
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_ptr_val<to_receiver_packets_sent_streams[1]>(0);
    init_ptr_val<to_sender_packets_acked_streams[0]>(0);
    init_ptr_val<to_sender_packets_acked_streams[1]>(0);
    init_ptr_val<to_sender_packets_acked_streams[2]>(0);
    init_ptr_val<to_sender_packets_completed_streams[0]>(0);
    init_ptr_val<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<to_sender_packets_completed_streams[2]>(0);

    if constexpr (is_handshake_sender) {
        erisc::datamover::handshake::sender_side_start(handshake_addr, DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
    } else {
        erisc::datamover::handshake::receiver_side_start(handshake_addr);
    }

    DPRINT << "SENDER_NUM_BUFFERS: " << (uint32_t)SENDER_NUM_BUFFERS << "\n";
    DPRINT << "RECEIVER_NUM_BUFFERS: " << (uint32_t)RECEIVER_NUM_BUFFERS << "\n";
    DPRINT << "local_sender_0_channel_address: " << (uint32_t)local_sender_0_channel_address << "\n";
    DPRINT << "local_sender_channel_0_connection_info_addr: " << (uint32_t)local_sender_channel_0_connection_info_addr
           << "\n";
    DPRINT << "local_sender_1_channel_address: " << (uint32_t)local_sender_1_channel_address << "\n";
    DPRINT << "local_sender_channel_1_connection_info_addr: " << (uint32_t)local_sender_channel_1_connection_info_addr
           << "\n";
    DPRINT << "local_sender_2_channel_address: " << (uint32_t)local_sender_2_channel_address << "\n";
    DPRINT << "local_sender_channel_2_connection_info_addr: " << (uint32_t)local_sender_channel_2_connection_info_addr
           << "\n";
    DPRINT << "local_receiver_0_channel_buffer_address: " << (uint32_t)local_receiver_0_channel_buffer_address << "\n";
    DPRINT << "remote_receiver_0_channel_buffer_address: " << (uint32_t)remote_receiver_0_channel_buffer_address
           << "\n";
    DPRINT << "local_receiver_1_channel_buffer_address: " << (uint32_t)local_receiver_1_channel_buffer_address << "\n";
    DPRINT << "remote_receiver_1_channel_buffer_address: " << (uint32_t)remote_receiver_1_channel_buffer_address
           << "\n";
    DPRINT << "remote_sender_0_channel_address: " << (uint32_t)remote_sender_0_channel_address << "\n";
    DPRINT << "remote_sender_1_channel_address: " << (uint32_t)remote_sender_1_channel_address << "\n";
    DPRINT << "remote_sender_2_channel_address: " << (uint32_t)remote_sender_2_channel_address << "\n";
    DPRINT << "forward_and_local_write_noc_vc: " << (uint32_t)tt::tt_fabric::forward_and_local_write_noc_vc << ENDL();

    // TODO: CONVERT TO SEMAPHORE
    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    volatile auto edm_local_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(edm_local_sync_ptr_addr);
    volatile auto edm_status_ptr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);

    // In persistent mode, we must rely on static addresses for our local semaphores that are locally
    // initialized, rather than metal device APIs. This way different subdevice programs can reliably
    // resolve the semaphore addresses on the EDM core

    std::array<PacketHeaderRecorder, MAX_NUM_SENDER_CHANNELS> sender_channel_packet_recorders{
        PacketHeaderRecorder(
            reinterpret_cast<volatile uint32_t*>(sender_0_completed_packet_header_cb_address),
            sender_0_completed_packet_header_cb_size_headers),
        PacketHeaderRecorder(
            reinterpret_cast<volatile uint32_t*>(sender_1_completed_packet_header_cb_address),
            sender_1_completed_packet_header_cb_size_headers),
        PacketHeaderRecorder(
            reinterpret_cast<volatile uint32_t*>(sender_2_completed_packet_header_cb_address),
            sender_2_completed_packet_header_cb_size_headers)};
    std::array<PacketHeaderRecorder, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_packet_recorders{
        PacketHeaderRecorder(
            reinterpret_cast<volatile uint32_t*>(receiver_0_completed_packet_header_cb_address),
            receiver_0_completed_packet_header_cb_size_headers),
        PacketHeaderRecorder(
            reinterpret_cast<volatile uint32_t*>(receiver_1_completed_packet_header_cb_address),
            receiver_1_completed_packet_header_cb_size_headers)};

    static_assert(SENDER_NUM_BUFFERS > 0, "compile time argument [1]: SENDER_NUM_BUFFERS must be > 0");
    static_assert(RECEIVER_NUM_BUFFERS > 0, "compile time argument [2]: RECEIVER_NUM_BUFFERS must be > 0");

    volatile tt::tt_fabric::EdmFabricReceiverChannelCounters* receiver_0_channel_counters_ptr = nullptr;
    volatile tt::tt_fabric::EdmFabricReceiverChannelCounters* receiver_1_channel_counters_ptr = nullptr;
    volatile tt::tt_fabric::EdmFabricSenderChannelCounters* sender_channel_0_counters_ptr = nullptr;
    volatile tt::tt_fabric::EdmFabricSenderChannelCounters* sender_channel_1_counters_ptr = nullptr;
    volatile tt::tt_fabric::EdmFabricSenderChannelCounters* sender_channel_2_counters_ptr = nullptr;

    if constexpr (enable_fabric_counters) {
        new (const_cast<tt::tt_fabric::EdmFabricReceiverChannelCounters*>(receiver_0_channel_counters_ptr))
            tt::tt_fabric::EdmFabricReceiverChannelCounters();
        new (const_cast<tt::tt_fabric::EdmFabricReceiverChannelCounters*>(receiver_1_channel_counters_ptr))
            tt::tt_fabric::EdmFabricReceiverChannelCounters();
        new (const_cast<tt::tt_fabric::EdmFabricSenderChannelCounters*>(sender_channel_0_counters_ptr))
            tt::tt_fabric::EdmFabricSenderChannelCounters();
        new (const_cast<tt::tt_fabric::EdmFabricSenderChannelCounters*>(sender_channel_1_counters_ptr))
            tt::tt_fabric::EdmFabricSenderChannelCounters();
        new (const_cast<tt::tt_fabric::EdmFabricSenderChannelCounters*>(sender_channel_2_counters_ptr))
            tt::tt_fabric::EdmFabricSenderChannelCounters();
    }

    size_t arg_idx = 0;
    ///////////////////////
    // Common runtime args:
    ///////////////////////

    const size_t local_sender_channel_0_connection_semaphore_addr =
        persistent_mode ? get_arg_val<uint32_t>(arg_idx++)
                        : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const size_t local_sender_channel_1_connection_semaphore_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const size_t local_sender_channel_2_connection_semaphore_addr =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    // unused - can later remove
    const size_t local_sender_channel_0_connection_buffer_index_addr =
        persistent_mode ? get_arg_val<uint32_t>(arg_idx++)
                        : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    const size_t local_sender_channel_1_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
    const size_t local_sender_channel_2_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);

    // downstream EDM semaphore location
    const bool has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    const auto downstream_edm_vc0_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // remote address for flow control
    const auto downstream_edm_vc0_semaphore_id = get_arg_val<uint32_t>(arg_idx++);  // TODO: Convert to semaphore ID
    const auto downstream_edm_vc0_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_vc0_noc_interface_buffer_index_local_addr = get_arg_val<uint32_t>(arg_idx++);

    // Receiver channels local semaphore for managing flow control with the downstream EDM.
    // The downstream EDM should be sending semaphore updates to this address any time it can
    // accept a new message
    const auto edm_vc0_forwarding_semaphore_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const auto edm_vc0_teardown_semaphore_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    // downstream EDM semaphore location
    const bool has_downstream_edm_vc1_buffer_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    const auto downstream_edm_vc1_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc1_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc1_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // remote address for flow control
    const auto downstream_edm_vc1_semaphore_id = get_arg_val<uint32_t>(arg_idx++);  // TODO: Convert to semaphore ID
    const auto downstream_edm_vc1_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc1_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_vc1_noc_interface_buffer_index_local_addr = get_arg_val<uint32_t>(arg_idx++);

    // Receiver channels local semaphore for managing flow control with the downstream EDM.
    // The downstream EDM should be sending semaphore updates to this address any time it can
    // accept a new message
    const auto edm_vc1_forwarding_semaphore_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));
    const auto edm_vc1_teardown_semaphore_address =
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++));

    ////////////////////////
    // Sender runtime args
    ////////////////////////
    auto sender0_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(
        persistent_mode ? get_arg_val<uint32_t>(arg_idx++)
                        : get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++)));
    auto sender1_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++)));
    auto sender_2_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(
        get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(get_arg_val<uint32_t>(arg_idx++)));

    if constexpr (persistent_mode) {
        // initialize the statically allocated "semaphores"
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_semaphore_addr) = 0;
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_buffer_index_addr) = 0;
        *sender0_worker_semaphore_ptr = 0;
    }

    *edm_status_ptr = tt::tt_fabric::EDMStatus::STARTED;

    //////////////////////////////
    //////////////////////////////
    //        Object Setup
    //////////////////////////////
    //////////////////////////////

    const auto& local_sender_buffer_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_0_channel_address, local_sender_1_channel_address, local_sender_2_channel_address});
    const auto& remote_sender_buffer_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                remote_sender_0_channel_address, remote_sender_1_channel_address, remote_sender_2_channel_address});
    const auto& local_receiver_buffer_addresses =
        take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_RECEIVER_CHANNELS>{
                local_receiver_0_channel_buffer_address, local_receiver_1_channel_buffer_address});
    const auto& remote_receiver_buffer_addresses =
        take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_RECEIVER_CHANNELS>{
                remote_receiver_0_channel_buffer_address, remote_receiver_1_channel_buffer_address});

    std::array<tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS> remote_receiver_channels;
    std::array<tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>, NUM_RECEIVER_CHANNELS> local_receiver_channels;
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> remote_sender_channels;
    std::array<tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS> local_sender_channels;
    std::array<tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>, NUM_SENDER_CHANNELS>
        local_sender_channel_worker_interfaces;
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_flow_control_semaphores =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                reinterpret_cast<size_t>(sender0_worker_semaphore_ptr),
                reinterpret_cast<size_t>(sender1_worker_semaphore_ptr),
                reinterpret_cast<size_t>(sender_2_worker_semaphore_ptr)});
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_channel_0_connection_semaphore_addr,
                local_sender_channel_1_connection_semaphore_addr,
                local_sender_channel_2_connection_semaphore_addr});
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_channel_0_connection_info_addr,
                local_sender_channel_1_connection_info_addr,
                local_sender_channel_2_connection_info_addr});
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_rdptr = 0;
    }

    std::array<tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>, NUM_USED_RECEIVER_CHANNELS>
        downstream_edm_noc_interfaces;
    if (has_downstream_edm_vc0_buffer_connection) {
        new (&downstream_edm_noc_interfaces[0]) tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>(
            // persistent_mode -> hardcode to false because for EDM -> EDM
            //  connections we must always use semaphore lookup
            false,
            downstream_edm_vc0_noc_x,
            downstream_edm_vc0_noc_y,
            downstream_edm_vc0_buffer_base_address,
            SENDER_NUM_BUFFERS,
            downstream_edm_vc0_semaphore_id,
            downstream_edm_vc0_worker_registration_id,
            downstream_edm_vc0_worker_location_info_address,
            channel_buffer_size,
            local_sender_channel_1_connection_buffer_index_id,
            reinterpret_cast<volatile uint32_t* const>(edm_vc0_forwarding_semaphore_address),
            reinterpret_cast<volatile uint32_t* const>(edm_vc0_teardown_semaphore_address),
            downstream_vc0_noc_interface_buffer_index_local_addr,
            receiver_channel_forwarding_data_cmd_buf_ids[0],
            receiver_channel_forwarding_sync_cmd_buf_ids[0]);
        downstream_edm_noc_interfaces[0]
            .template setup_edm_noc_cmd_buf<
                tt::tt_fabric::edm_to_downstream_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc>();
    }
    if constexpr (enable_ring_support) {
        if (has_downstream_edm_vc1_buffer_connection) {
            new (&downstream_edm_noc_interfaces[1]) tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>(
                // persistent_mode -> hardcode to false because for EDM -> EDM
                //  connections we must always use semaphore lookup
                false,
                downstream_edm_vc1_noc_x,
                downstream_edm_vc1_noc_y,
                downstream_edm_vc1_buffer_base_address,
                SENDER_NUM_BUFFERS,
                downstream_edm_vc1_semaphore_id,
                downstream_edm_vc1_worker_registration_id,
                downstream_edm_vc1_worker_location_info_address,
                channel_buffer_size,
                local_sender_channel_2_connection_buffer_index_id,
                reinterpret_cast<volatile uint32_t* const>(edm_vc1_forwarding_semaphore_address),
                reinterpret_cast<volatile uint32_t* const>(edm_vc1_teardown_semaphore_address),
                downstream_vc1_noc_interface_buffer_index_local_addr,
                receiver_channel_forwarding_data_cmd_buf_ids[1],
                receiver_channel_forwarding_sync_cmd_buf_ids[1]);
            downstream_edm_noc_interfaces[1]
                .template setup_edm_noc_cmd_buf<
                    tt::tt_fabric::edm_to_downstream_noc,
                    tt::tt_fabric::forward_and_local_write_noc_vc>();
        }
    }
    for (uint8_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        new (&local_receiver_channels[i]) tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>(
            local_receiver_buffer_addresses[i],
            channel_buffer_size,
            sizeof(PACKET_HEADER_TYPE),
            eth_transaction_ack_word_addr,  // Unused, otherwise probably need to have unique ack word per channel
            receiver_channel_base_id + i);
        new (&remote_receiver_channels[i]) tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>(
            remote_receiver_buffer_addresses[i],
            channel_buffer_size,
            sizeof(PACKET_HEADER_TYPE),
            eth_transaction_ack_word_addr,  // Unused, otherwise probably need to have unique ack word per channel
            receiver_channel_base_id + i);
    }

    for (uint8_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        new (&local_sender_channels[i]) tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>(
            local_sender_buffer_addresses[i],
            channel_buffer_size,
            sizeof(PACKET_HEADER_TYPE),
            0,  // For sender channels there is no eth_transaction_ack_word_addr because they don't send acks
            i);
        new (&remote_sender_channels[i]) tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>(
            remote_sender_buffer_addresses[i],
            channel_buffer_size,
            sizeof(PACKET_HEADER_TYPE),
            0,  // For sender channels there is no eth_transaction_ack_word_addr because they don't send acks
            i);
    }
    init_local_sender_channel_worker_interfaces(
        local_sender_connection_live_semaphore_addresses,
        local_sender_connection_info_addresses,
        local_sender_channel_worker_interfaces,
        local_sender_flow_control_semaphores);

    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS, NUM_TRANSACTION_IDS, 0> receiver_channel_0_trid_tracker;
    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS, NUM_TRANSACTION_IDS, NUM_TRANSACTION_IDS>
        receiver_channel_1_trid_tracker;

    if (has_downstream_edm_vc0_buffer_connection) {
        for (auto& downstream_edm_noc_interface : downstream_edm_noc_interfaces) {
            downstream_edm_noc_interface.template open<true, tt::tt_fabric::worker_handshake_noc>();
            *downstream_edm_noc_interface.from_remote_buffer_slot_rdptr_ptr = 0;
            ASSERT(*downstream_edm_noc_interface.from_remote_buffer_slot_rdptr_ptr == 0);
        }
    }

    if constexpr (is_handshake_sender) {
        erisc::datamover::handshake::sender_side_finish(handshake_addr, DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
    } else {
        erisc::datamover::handshake::receiver_side_finish(handshake_addr, DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
    }

    *edm_status_ptr = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;

    if constexpr (wait_for_host_signal) {
        if constexpr (is_local_handshake_master) {
            wait_for_notification((uint32_t)edm_local_sync_ptr, num_local_edms - 1);
            notify_slave_routers(
                edm_channels_mask, local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr, num_local_edms);
        } else {
            notify_master_router(local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr);
            wait_for_notification((uint32_t)edm_local_sync_ptr, num_local_edms);
        }

        *edm_status_ptr = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;

        wait_for_notification((uint32_t)edm_status_ptr, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);

        if constexpr (is_local_handshake_master) {
            notify_slave_routers(
                edm_channels_mask,
                local_handshake_master_eth_chan,
                (uint32_t)edm_status_ptr,
                tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        }
    }

    wait_for_static_connection_to_ready(local_sender_channel_worker_interfaces);

    //////////////////////////////
    //////////////////////////////
    //        MAIN LOOP
    //////////////////////////////
    //////////////////////////////
    run_fabric_edm_main_loop<
        enable_packet_header_recording,
        enable_fabric_counters,
        RECEIVER_NUM_BUFFERS,
        NUM_RECEIVER_CHANNELS,
        SENDER_NUM_BUFFERS,
        NUM_SENDER_CHANNELS,
        MAX_NUM_SENDER_CHANNELS,
        MAX_NUM_RECEIVER_CHANNELS>(
        local_receiver_channels,
        local_sender_channels,
        local_sender_channel_worker_interfaces,
        downstream_edm_noc_interfaces,
        remote_sender_channels,
        remote_receiver_channels,
        termination_signal_ptr,
        {receiver_0_channel_counters_ptr, receiver_1_channel_counters_ptr},
        {sender_channel_0_counters_ptr, sender_channel_1_counters_ptr, sender_channel_2_counters_ptr},
        receiver_channel_packet_recorders,
        sender_channel_packet_recorders,
        receiver_channel_0_trid_tracker,
        receiver_channel_1_trid_tracker);

    if constexpr (persistent_mode) {
        // we force these values to a non-zero value so that if we run the fabric back to back,
        // and we can reliably probe from host that this kernel has initialized properly.
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_semaphore_addr) = 99;
        *reinterpret_cast<volatile uint32_t*>(local_sender_channel_0_connection_buffer_index_addr) = 99;
        *sender0_worker_semaphore_ptr = 99;
    }

    // make sure all the noc transactions are acked before re-init the noc counters
    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();
    receiver_channel_1_trid_tracker.all_buffer_slot_transactions_acked();

    // re-init the noc counters as the noc api used is not incrementing them
    ncrisc_noc_counters_init();

    if constexpr (wait_for_host_signal) {
        if constexpr (is_local_handshake_master) {
            notify_slave_routers(
                edm_channels_mask,
                local_handshake_master_eth_chan,
                (uint32_t)termination_signal_ptr,
                *termination_signal_ptr);
            noc_async_write_barrier();
        }
    }

    *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;

    DPRINT << "EDM DONE\n";
    WAYPOINT("DONE");
}
