// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_l1_address_map.h"

typedef struct tt_fabric_endpoint_sync {
    uint32_t sync_addr : 24;
    uint32_t endpoint_type : 8;
} tt_fabric_endpoint_sync_t;

static_assert(sizeof(tt_fabric_endpoint_sync_t) == 4);

constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t NUM_WR_CMD_BUFS = 4;
constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS =
    (NUM_WR_CMD_BUFS - 1) * (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES) / PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2 * 1024;
constexpr uint32_t FVC_SYNC_THRESHOLD = 256;

enum SessionCommand : uint32_t {
    ASYNC_WR = (0x1 << 0),
    ASYNC_WR_RESP = (0x1 << 1),
    ASYNC_RD = (0x1 << 2),
    ASYNC_RD_RESP = (0x1 << 3),
    DSOCKET_WR = (0x1 << 4),
    SSOCKET_WR = (0x1 << 5),
    ATOMIC_INC = (0x1 << 6),
    ATOMIC_READ_INC = (0x1 << 7),
};

#define INVALID 0x0
#define DATA 0x1
#define MCAST_DATA 0x2
#define SYNC 0x4
#define FORWARD 0x8
#define INLINE_FORWARD 0x10
#define PACK_N_FORWARD 0x20
#define TERMINATE 0x40
#define NOP 0xFF

typedef struct tt_routing {
    uint32_t packet_size_bytes;
    uint16_t dst_mesh_id;  // Remote mesh
    uint16_t dst_dev_id;   // Remote device
    uint16_t src_mesh_id;  // Source mesh
    uint16_t src_dev_id;   // Source device
    uint16_t ttl;
    uint8_t version;
    uint8_t flags;
} tt_routing;

static_assert(sizeof(tt_routing) == 16);

typedef struct tt_session {
    SessionCommand command;
    uint32_t target_offset_l;  // RDMA address
    uint32_t target_offset_h;
    uint32_t ack_offset_l;  // fabric client local address for session command acknowledgement.
                            // This is complete end-to-end acknowledgement of sessoin command completion at the remote
                            // device.
    uint32_t ack_offset_h;
} tt_session;

static_assert(sizeof(tt_session) == 20);

typedef struct mcast_params {
    uint16_t east;
    uint16_t west;
    uint16_t north;
    uint16_t south;
    uint32_t socket_id;  // Socket Id for DSocket Multicast. Ignored for ASYNC multicast.
} mcast_params;

typedef struct socket_params {
    uint32_t socket_id;
} socket_params;

typedef struct atomic_params {
    uint32_t padding;
    uint32_t
        return_offset;  // L1 offset where atomic read should be returned. Noc X/Y is taken from tt_session.ack_offset
    uint32_t increment : 24;  // NOC atomic increment wrapping value.
    uint32_t wrap_boundary : 8;
} atomic_params;

typedef struct read_params {
    uint32_t return_offset_l;  // address where read data should be copied
    uint32_t return_offset_h;
    uint32_t size;  // number of bytes to read
} read_params;

typedef struct misc_params {
    uint32_t words[3];
} misc_params;

typedef union packet_params {
    mcast_params mcast_parameters;
    socket_params socket_parameters;
    atomic_params atomic_parameters;
    read_params read_parameters;
    misc_params misc_parameters;
    uint8_t bytes[12];
} packet_params;

typedef struct packet_header {
    packet_params packet_parameters;
    tt_session session;
    tt_routing routing;
} packet_header_t;

const uint32_t PACKET_HEADER_SIZE_BYTES = 48;
const uint32_t PACKET_HEADER_SIZE_WORDS = PACKET_HEADER_SIZE_BYTES / PACKET_WORD_SIZE_BYTES;

static_assert(sizeof(packet_header) == PACKET_HEADER_SIZE_BYTES);

// This is a pull request entry for a fabric router.
// Pull request issuer populates these entries to identify
// the data that fabric router needs to pull from requestor.
// This data is the forwarded by router over ethernet.
// A pull request can be for packetized data or raw data, as specified by flags field.
//   - When registering a pull request for raw data, the requestor pushes two entries to router request queue.
//     First entry is packet_header, second entry is pull_request. This is typical of OP/Endpoint issuing read/writes
//     over tt-fabric.
//   - When registering a pull request for packetized data, the requetor only pushed pull_request entry to router
//   request queue.
//     This is typical of fabric routers forwarding data over noc/ethernet hops.
//
typedef struct pull_request {
    uint32_t wr_ptr;        // Current value of write pointer.
    uint32_t rd_ptr;        // Current value of read pointer. Points to first byte of pull data.
    uint32_t size;          // Total number of bytes that need to be forwarded.
    uint32_t buffer_size;   // Producer local buffer size. Used for flow control when total data to send does not fit in
                            // local buffer.
    uint64_t buffer_start;  // Producer local buffer start. Used for wrapping rd/wr_ptr at the end of buffer.
    uint64_t ack_addr;  // Producer local address to send rd_ptr updates. fabric router pushes its rd_ptr to requestor
                        // at this address.
    uint8_t padding[15];
    uint8_t flags;  // Router command.
} pull_request_t;

const uint32_t PULL_REQ_SIZE_BYTES = 48;

static_assert(sizeof(pull_request) == PULL_REQ_SIZE_BYTES);
static_assert(sizeof(pull_request) == sizeof(packet_header));

typedef union chan_request_entry {
    pull_request_t pull_request;
    packet_header_t packet_header;
    uint8_t bytes[48];
} chan_request_entry_t;

const uint32_t CHAN_PTR_SIZE_BYTES = 16;
typedef struct chan_ptr {
    uint32_t ptr;
    uint32_t pad[3];
} chan_ptr;
static_assert(sizeof(chan_ptr) == CHAN_PTR_SIZE_BYTES);

const uint32_t CHAN_REQ_BUF_LOG_SIZE = 4;  // must be 2^N
const uint32_t CHAN_REQ_BUF_SIZE = 16;     // must be 2^N
const uint32_t CHAN_REQ_BUF_SIZE_MASK = (CHAN_REQ_BUF_SIZE - 1);
const uint32_t CHAN_REQ_BUF_PTR_MASK = ((CHAN_REQ_BUF_SIZE << 1) - 1);
const uint32_t CHAN_REQ_BUF_SIZE_BYTES = 2 * CHAN_PTR_SIZE_BYTES + CHAN_REQ_BUF_SIZE * PULL_REQ_SIZE_BYTES;

typedef struct chan_req_buf {
    chan_ptr wrptr;
    chan_ptr rdptr;
    chan_request_entry_t chan_req[CHAN_REQ_BUF_SIZE];
} chan_req_buf;

static_assert(sizeof(chan_req_buf) == CHAN_REQ_BUF_SIZE_BYTES);

typedef struct local_pull_request {
    chan_ptr wrptr;
    chan_ptr rdptr;
    pull_request_t pull_request;
} local_pull_request_t;

typedef struct chan_payload_ptr {
    uint32_t ptr;
    uint32_t pad[2];
    uint32_t ptr_cleared;
} chan_payload_ptr;

static_assert(sizeof(chan_payload_ptr) == CHAN_PTR_SIZE_BYTES);

constexpr uint32_t FABRIC_ROUTER_REQ_QUEUE_START = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
constexpr uint32_t FABRIC_ROUTER_REQ_QUEUE_SIZE = sizeof(chan_req_buf);
constexpr uint32_t FABRIC_ROUTER_DATA_BUF_START = FABRIC_ROUTER_REQ_QUEUE_START + FABRIC_ROUTER_REQ_QUEUE_SIZE;
