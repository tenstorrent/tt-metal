/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include "noc/noc_parameters.h"

const uint32_t NOC_SIZE_X = 10;
const uint32_t NOC_SIZE_Y = 12;

const uint32_t NUM_ETH_INST = 16;

const uint32_t MAX_CHIP_GRID_SIZE = 128;

const uint32_t CMD_BUF_SIZE = 4;  // we assume must be 2^N

const uint32_t CMD_BUF_SIZE_MASK = (CMD_BUF_SIZE - 1);
const uint32_t CMD_BUF_PTR_MASK  = ((CMD_BUF_SIZE << 1) - 1);

const uint32_t ETH_ROUTING_STRUCT_ADDR = 0x11000;
const uint32_t ETH_ROUTING_DATA_BUFFER_ADDR = 0x12000;
//const uint32_t ETH_ROUTING_STRUCT_ADDR = eth_l1_mem::address_map::COMMAND_Q_BASE;
//const uint32_t ETH_ROUTING_DATA_BUFFER_ADDR = eth_l1_mem::address_map::DATA_BUFFER_BASE;

const uint32_t ROOT_NODE_NOC_X = 9;
const uint32_t ROOT_NODE_NOC_Y = 0;

const uint32_t CMD_WR_REQ  = (0x1 << 0);
const uint32_t CMD_WR_ACK  = (0x1 << 1);
const uint32_t CMD_RD_REQ  = (0x1 << 2);
const uint32_t CMD_RD_DATA = (0x1 << 3);
const uint32_t CMD_DATA_BLOCK_DRAM = (0x1 << 4);
const uint32_t CMD_LAST_DATA_BLOCK_DRAM = (0x1 << 5);
const uint32_t CMD_DATA_BLOCK = (0x1 << 6);
const uint32_t CMD_NOC_ID_SHIFT = 9;
const uint32_t CMD_NOC_ID = (0x1 << CMD_NOC_ID_SHIFT);
const uint32_t CMD_TIMESTAMP_SHIFT = 10;
const uint32_t CMD_TIMESTAMP = (0x1 << CMD_TIMESTAMP_SHIFT);

const uint32_t CMD_DATA_BLOCK_UNAVAILABLE = (0x1 << 30);
const uint32_t CMD_DEST_UNREACHABLE = (0x1 << 31);
const uint32_t MAX_BLOCK_SIZE = (0x1 << 10); // Max 1024 bytes

const uint32_t CMD_SIZE_BYTES = 32;
const uint32_t ETH_RACK_COORD_WIDTH = 8;

typedef struct {
  uint64_t sys_addr;
  uint32_t data;
  uint32_t flags;
  uint16_t rack;
  uint16_t src_resp_buf_index;
  uint32_t local_buf_index;
  uint8_t  src_resp_q_id;
  uint8_t  host_mem_txn_id;
  uint16_t padding;
  uint32_t src_addr_tag; //upper 32-bits of request source address.
} routing_cmd_t;
static_assert(sizeof(routing_cmd_t) == CMD_SIZE_BYTES);


const uint32_t REMOTE_UPDATE_PTR_SIZE_BYTES = 16;
typedef struct {
  uint32_t ptr;
  uint32_t pad[3];
} remote_update_ptr_t;
static_assert(sizeof(remote_update_ptr_t) == REMOTE_UPDATE_PTR_SIZE_BYTES);

const uint32_t CMD_COUNTERS_SIZE_BYTES = 32;
typedef struct {
  uint32_t wr_req;
  uint32_t wr_resp;
  uint32_t rd_req;
  uint32_t rd_resp;
  uint32_t error;
  uint32_t pad[3];
} cmd_counters_t;
static_assert(sizeof(cmd_counters_t) == CMD_COUNTERS_SIZE_BYTES);

const uint32_t CMD_Q_SIZE_BYTES = 2*REMOTE_UPDATE_PTR_SIZE_BYTES + CMD_COUNTERS_SIZE_BYTES + CMD_BUF_SIZE*CMD_SIZE_BYTES;
typedef struct {
  cmd_counters_t cmd_counters;
  remote_update_ptr_t wrptr;
  remote_update_ptr_t rdptr;
  routing_cmd_t cmd[CMD_BUF_SIZE];
} cmd_q_t;
static_assert(sizeof(cmd_q_t) == CMD_Q_SIZE_BYTES);


//there are 16 64-bit latency counters at the beginning of command q region.
constexpr uint32_t REQUEST_CMD_QUEUE_BASE = ETH_ROUTING_STRUCT_ADDR + 128;
constexpr uint32_t REQUEST_ROUTING_CMD_QUEUE_BASE = REQUEST_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t) + sizeof(remote_update_ptr_t) + sizeof(cmd_counters_t);
constexpr uint32_t RESPONSE_CMD_QUEUE_BASE = REQUEST_CMD_QUEUE_BASE + 2*CMD_Q_SIZE_BYTES;
constexpr uint32_t RESPONSE_ROUTING_CMD_QUEUE_BASE = RESPONSE_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t) + sizeof(remote_update_ptr_t) + sizeof(cmd_counters_t);
