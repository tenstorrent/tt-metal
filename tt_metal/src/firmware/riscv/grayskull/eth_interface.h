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

const uint32_t ETH_ROUTING_STRUCT_ADDR = 0x10000;
const uint32_t ETH_ROUTING_DATA_BUFFER_ADDR = 0x11000;

const uint32_t ROOT_NODE_CHIP_X = 0;
const uint32_t ROOT_NODE_CHIP_Y = 0;
const uint32_t ROOT_NODE_NOC_X = 9;
const uint32_t ROOT_NODE_NOC_Y = 0;

const uint32_t CMD_WR_REQ  = (0x1 << 0);
const uint32_t CMD_WR_ACK  = (0x1 << 1);
const uint32_t CMD_RD_REQ  = (0x1 << 2);
const uint32_t CMD_RD_DATA = (0x1 << 3);
const uint32_t CMD_ETH_FWD_STARTED = (0x1 << 4);
const uint32_t CMD_ETH_FWD_ACKED = (0x1 << 5);
const uint32_t CMD_DATA_BLOCK = (0x1 << 6);
const uint32_t MAX_BLOCK_SIZE = (0x1 << 10); // Max 1024 bytes

const uint32_t CMD_SIZE_BYTES = 32;
typedef struct {
  uint64_t sys_addr;
  uint32_t data;
  uint32_t flags;
  uint32_t src_resp_buf_index;
  uint32_t local_buf_index;
  uint32_t pad[2];
} routing_cmd_t;
static_assert(sizeof(routing_cmd_t) == CMD_SIZE_BYTES);


const uint32_t REMOTE_UPDATE_PTR_SIZE_BYTES = 16;
typedef struct {
  uint32_t ptr;
  uint32_t pad[3];
} remote_update_ptr_t;
static_assert(sizeof(remote_update_ptr_t) == REMOTE_UPDATE_PTR_SIZE_BYTES);

const uint32_t CMD_Q_SIZE_BYTES = 2*REMOTE_UPDATE_PTR_SIZE_BYTES + CMD_BUF_SIZE*CMD_SIZE_BYTES;
typedef struct {
  remote_update_ptr_t wrptr;
  remote_update_ptr_t rdptr;
  routing_cmd_t cmd[CMD_BUF_SIZE];
} cmd_q_t;
static_assert(sizeof(cmd_q_t) == CMD_Q_SIZE_BYTES);
