// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// TDMA XMOV functions for the firmware
//

#include "tensix.h"
#include "tensix_types.h"
#include "tensix_functions.h"
#include "noc.h"

#include "tdma_xmov.h"

#define CMD_TDMA_XMOV      0x40
#define CMD_TDMA_MOV_FLUSH 0x46
#define CMD_TDMA_MEM_WRITE 0x66

// Write TDMA registers and initiate a move
void tdma_xmov(uint mover_number, uint source_addr, uint dest_addr, uint size, xmov_direction_t direction)
{
  memory_write(RISCV_TDMA_REG_XMOV_SRC_ADDR, source_addr);
  memory_write(RISCV_TDMA_REG_XMOV_DST_ADDR, dest_addr);
  memory_write(RISCV_TDMA_REG_XMOV_SIZE, size);
  memory_write(RISCV_TDMA_REG_XMOV_DIRECTION, (uint)direction);

  uint cmd = CMD_TDMA_XMOV;
  uint args = mover_number;

  cmd = cmd | (args << 8);
  memory_write(RISCV_TDMA_REG_COMMAND_ADDR, cmd);
}

void wait_tdma_movers_done(uint mover_busy_mask)
{
  uint volatile tdma_mover_status;
  tdma_mover_status = memory_read(RISCV_TDMA_REG_STATUS); // Dummy read to flush the pipe

  do {
    tdma_mover_status = memory_read(RISCV_TDMA_REG_STATUS);
  // Wait until both movers are not busy, and fifo is empty
  } while ( (tdma_mover_status & (mover_busy_mask | RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK))
              != RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK);
}
