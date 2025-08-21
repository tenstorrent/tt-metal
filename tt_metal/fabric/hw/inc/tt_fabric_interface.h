// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_l1_address_map.h"
#include "noc/noc_parameters.h"
#include "hostdevcommon/fabric_common.h"

#if not(defined(KERNEL_BUILD) || defined(FW_BUILD))
static_assert(false, "tt_fabric_interface.h should only be included in kernel or firmware build");
#endif

namespace tt::tt_fabric {

struct endpoint_sync_t {
    uint32_t sync_addr : 24;
    uint32_t endpoint_type : 8;
};

static_assert(sizeof(endpoint_sync_t) == 4);

constexpr uint32_t CLIENT_HEADER_BUFFER_ENTRIES = 4;
constexpr uint32_t PUSH_CLIENT_INTERFACE_SIZE = 288;
constexpr uint32_t NUM_WR_CMD_BUFS = 4;
constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES) / PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2 * 1024;
constexpr uint32_t FVC_SYNC_THRESHOLD = 256;

#define INVALID 0x0
#define MCAST_ACTIVE 0x1
#define MCAST_DATA 0x2
#define SYNC 0x4
#define FORWARD 0x8
#define INLINE_FORWARD 0x10
#define PACK_N_FORWARD 0x20
#define TERMINATE 0x40
#define NOP 0xFF

struct tt_routing {
    uint32_t packet_size_bytes;
    uint16_t dst_mesh_id;  // Remote mesh
    uint16_t dst_dev_id;   // Remote device
    uint16_t src_mesh_id;  // Source mesh
    uint16_t src_dev_id;   // Source device
    uint16_t ttl;
    uint8_t version;
    uint8_t flags;
};

static_assert(sizeof(tt_routing) == 16);

struct tt_low_latency_routing_vector {
    static constexpr uint32_t FIELD_WIDTH = 8;
    static constexpr uint32_t FIELD_MASK = 0b1111;
    static constexpr uint32_t NOOP = 0b0000;
    static constexpr uint32_t FORWARD_EAST = 0b0001;
    static constexpr uint32_t FORWARD_WEST = 0b0010;
    static constexpr uint32_t FORWARD_NORTH = 0b0100;
    static constexpr uint32_t FORWARD_SOUTH = 0b1000;

    static constexpr uint32_t FORWARD_ONLY = 0b10;
    static constexpr uint32_t WRITE_AND_FORWARD = 0b11;
    static constexpr uint32_t MAX_NUM_ENCODINGS = sizeof(uint32_t) * CHAR_BIT / FIELD_WIDTH;
    static constexpr uint32_t FWD_ONLY_FIELD = 0xAAAAAAAA;
    static constexpr uint32_t WR_ONLY_FIELD = 0x55555555;
    uint32_t hop_index;
    uint32_t value[4];
};

struct tt_low_latency_routing {
    uint32_t packet_size_bytes;
    uint32_t target_offset_l;
    uint32_t target_offset_h;
    uint32_t command;
    tt_low_latency_routing_vector route_vector;
    uint32_t atomic_offset_l;
    uint32_t atomic_offset_h;
    uint16_t atomic_increment;
    uint16_t atomic_wrap;
};

static_assert(sizeof(tt_low_latency_routing) == PACKET_HEADER_SIZE_BYTES);

struct tt_session {
    uint32_t command;
    uint32_t target_offset_l;  // RDMA address
    uint32_t target_offset_h;
    uint32_t ack_offset_l;  // fabric client local address for session command acknowledgement.
                            // This is complete end-to-end acknowledgement of sessoin command completion at the remote
                            // device.
    uint32_t ack_offset_h;
};

static_assert(sizeof(tt_session) == 20);

struct mcast_params {
    uint32_t socket_id;  // Socket Id for DSocket Multicast. Ignored for ASYNC multicast.
    uint16_t east;
    uint16_t west;
    uint16_t north;
    uint16_t south;
};

struct socket_params {
    uint32_t padding1;
    uint16_t socket_id;
    uint16_t epoch_id;
    uint8_t socket_type;
    uint8_t socket_direction;
    uint8_t routing_plane;
    uint8_t padding;
};

struct atomic_params {
    uint32_t padding;
    uint32_t
        return_offset;  // L1 offset where atomic read should be returned. Noc X/Y is taken from tt_session.ack_offset
    uint32_t increment : 24;  // NOC atomic increment wrapping value.
    uint32_t wrap_boundary : 8;
};

struct async_wr_atomic_params {
    uint32_t padding;
    uint32_t l1_offset;
    uint32_t noc_xy : 24;
    uint32_t increment : 8;
};

struct read_params {
    uint32_t return_offset_l;  // address where read data should be copied
    uint32_t return_offset_h;
    uint32_t size;  // number of bytes to read
};

struct misc_params {
    uint32_t words[3];
};

union packet_params {
    mcast_params mcast_parameters;
    socket_params socket_parameters;
    atomic_params atomic_parameters;
    async_wr_atomic_params async_wr_atomic_parameters;
    read_params read_parameters;
    misc_params misc_parameters;
    uint8_t bytes[12];
};

struct packet_header_t {
    tt_routing routing;
    tt_session session;
    packet_params packet_parameters;
};

struct low_latency_packet_header_t {
    tt_low_latency_routing routing;
};

static_assert(sizeof(low_latency_packet_header_t) == PACKET_HEADER_SIZE_BYTES);

static_assert(sizeof(packet_header_t) == PACKET_HEADER_SIZE_BYTES);

static_assert(offsetof(packet_header_t, routing) % 4 == 0);

constexpr uint32_t packet_header_routing_offset_dwords = offsetof(packet_header_t, routing) / 4;

void tt_fabric_add_header_checksum(packet_header_t* p_header) {
    uint16_t* ptr = (uint16_t*)p_header;
    uint32_t sum = 0;
    for (uint32_t i = 2; i < sizeof(packet_header_t) / 2; i++) {
        sum += ptr[i];
    }
    sum = ~sum;
    sum += sum;
    p_header->packet_parameters.misc_parameters.words[0] = sum;
}

bool tt_fabric_is_header_valid(packet_header_t* p_header) {
#ifdef TT_FABRIC_DEBUG
    uint16_t* ptr = (uint16_t*)p_header;
    uint32_t sum = 0;
    for (uint32_t i = 2; i < sizeof(packet_header_t) / 2; i++) {
        sum += ptr[i];
    }
    sum = ~sum;
    sum += sum;
    return (p_header->packet_parameters.misc_parameters.words[0] == sum);
#else
    return true;
#endif
}

constexpr uint32_t CHAN_PTR_SIZE_BYTES = 16;
struct chan_ptr {
    uint32_t ptr;
    uint32_t pad[3];
};
static_assert(sizeof(chan_ptr) == CHAN_PTR_SIZE_BYTES);

constexpr uint32_t CHAN_REQ_BUF_LOG_SIZE = 4;  // must be 2^N
constexpr uint32_t CHAN_REQ_BUF_SIZE = 16;     // must be 2^N
constexpr uint32_t CHAN_REQ_BUF_SIZE_MASK = (CHAN_REQ_BUF_SIZE - 1);
constexpr uint32_t CHAN_REQ_BUF_PTR_MASK = ((CHAN_REQ_BUF_SIZE << 1) - 1);

struct chan_payload_ptr {
    uint32_t ptr;
    uint32_t pad[2];
    uint32_t ptr_cleared;
};

static_assert(sizeof(chan_payload_ptr) == CHAN_PTR_SIZE_BYTES);

constexpr uint32_t FABRIC_ROUTER_CLIENT_QUEUE_SIZE = 48;
struct fabric_push_client_queue_t {
    chan_ptr client_idx_counter;
    chan_ptr curr_client_idx;
    chan_ptr router_wr_ptr;
};
static_assert(sizeof(fabric_push_client_queue_t) % 16 == 0);
static_assert(sizeof(fabric_push_client_queue_t) == FABRIC_ROUTER_CLIENT_QUEUE_SIZE);

constexpr uint32_t FABRIC_ROUTER_CLIENT_QUEUE_LOCAL_SIZE = 48;
struct fabric_push_client_queue_local_t {
    chan_ptr my_client_idx;
    chan_ptr remote_curr_client_idx;
    chan_ptr remote_router_wr_ptr;
};
static_assert(sizeof(fabric_push_client_queue_local_t) % 16 == 0);
static_assert(sizeof(fabric_push_client_queue_local_t) == FABRIC_ROUTER_CLIENT_QUEUE_LOCAL_SIZE);

struct fabric_push_client_interface_t {
    uint32_t num_routing_planes;
    uint32_t routing_tables_l1_offset;
    uint32_t router_addr_h;
    uint32_t buffer_start;
    uint32_t buffer_size;
    uint32_t wr_ptr;
    uint32_t router_push_addr;
    uint32_t router_space;
    uint32_t update_router_space;
    uint32_t reserved[3];
    fabric_push_client_queue_local_t local_client_req_entry;
    packet_header_t header_buffer[CLIENT_HEADER_BUFFER_ENTRIES];
};

static_assert(sizeof(fabric_push_client_interface_t) % 16 == 0);
static_assert(sizeof(fabric_push_client_interface_t) == PUSH_CLIENT_INTERFACE_SIZE);

constexpr uint32_t FABRIC_ROUTER_MISC_START = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
constexpr uint32_t FABRIC_ROUTER_MISC_SIZE = 256;
constexpr uint32_t FABRIC_ROUTER_SYNC_SEM = FABRIC_ROUTER_MISC_START;
constexpr uint32_t FABRIC_ROUTER_SYNC_SEM_SIZE = 16;
static_assert(FABRIC_ROUTER_SYNC_SEM == eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);

// Fabric Virtual Channel start/size
constexpr uint32_t FABRIC_ROUTER_CLIENT_QUEUE_START = FABRIC_ROUTER_MISC_START + FABRIC_ROUTER_MISC_SIZE;
constexpr uint32_t FABRIC_ROUTER_REQ_QUEUE_START = FABRIC_ROUTER_CLIENT_QUEUE_START + FABRIC_ROUTER_CLIENT_QUEUE_SIZE;
constexpr uint32_t FABRIC_ROUTER_DATA_BUF_START = FABRIC_ROUTER_REQ_QUEUE_START;
constexpr uint32_t FABRIC_ROUTER_BUF_SLOT_SIZE = 0x1000 + PACKET_HEADER_SIZE_BYTES;
constexpr uint32_t FABRIC_ROUTER_OUTBOUND_BUF_SIZE = 4 * FABRIC_ROUTER_BUF_SLOT_SIZE;
constexpr uint32_t FABRIC_ROUTER_INBOUND_BUF_SIZE = 8 * FABRIC_ROUTER_BUF_SLOT_SIZE;
constexpr uint32_t FABRIC_ROUTER_OUTBOUND_BUF_SLOTS = FABRIC_ROUTER_OUTBOUND_BUF_SIZE / FABRIC_ROUTER_BUF_SLOT_SIZE;
constexpr uint32_t FABRIC_ROUTER_INBOUND_BUF_SLOTS = FABRIC_ROUTER_INBOUND_BUF_SIZE / FABRIC_ROUTER_BUF_SLOT_SIZE;

}  // namespace tt::tt_fabric
