// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "risc_common.h"
#include "noc_nonblocking_api.h"
#include "stream_interface.h"

void risc_init() {
  for (uint32_t n = 0; n < NUM_NOCS; n++) {
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
    my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
    my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
  }
}

void replicate(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate) {
  const uint32_t REPLICATE_VC = 0;
  for (uint32_t j = 0; j < times_to_replicate; j++) {
    while (!noc_cmd_buf_ready(noc_id, NCRISC_WR_CMD_BUF));
    ncrisc_noc_fast_write(noc_id, NCRISC_WR_CMD_BUF,
                          src_addr,
                          dest_addr,
                          chunk_size_bytes,
                          REPLICATE_VC, false, false, 1);
    dest_addr += chunk_size_bytes;
  }
}

void replicate_l1(uint32_t noc_id, uint32_t src_addr, uint64_t dest_addr, uint32_t chunk_size_bytes, uint32_t times_to_replicate) {
  const uint32_t REPLICATE_VC = 0;
  for (uint32_t j = 0; j < times_to_replicate; j++) {
    while (!noc_cmd_buf_ready_l1(noc_id, NCRISC_WR_CMD_BUF));
    ncrisc_noc_fast_write_l1(noc_id, NCRISC_WR_CMD_BUF,
                          src_addr,
                          dest_addr,
                          chunk_size_bytes,
                          REPLICATE_VC, false, false, 1);
    dest_addr += chunk_size_bytes;
  }
}
