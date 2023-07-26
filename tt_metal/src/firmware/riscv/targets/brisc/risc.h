#pragma once

#include <stdint.h>

#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"


typedef struct active_stream_info_t {
  uint8_t  stream_id;
  uint8_t  active_streams_idx;
  uint8_t  start_phase_num_cfg_regs;
  uint8_t  dram_prefetch_stream_need_restart;
  uint16_t blob_start_offset;
  uint16_t flags;
  uint32_t epoch_iters_remaining;
} active_stream_info_t;


inline void RISC_POST_STATUS(uint32_t status) {
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)(NOC_CFG(ROUTER_CFG_2) + NOC_INSTANCE_OFFSET);
  ptr[0] = status;
}

void risc_reset_check();

extern uint8_t loading_noc;

#define RISC_EPOCH_INFO_PTR EPOCH_INFO_PTR
