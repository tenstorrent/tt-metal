// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _ERISC_H_
#define _ERISC_H_

#include <stdint.h>

#include "noc_parameters.h"

extern uint8_t my_x[NUM_NOCS];
extern uint8_t my_y[NUM_NOCS];
extern uint8_t my_logical_x[NUM_NOCS];
extern uint8_t my_logical_y[NUM_NOCS];
extern uint8_t loading_noc;
extern uint8_t noc_size_x;
extern uint8_t noc_size_y;
extern uint8_t noc_trans_table_en;

extern int post_index;

inline void RISC_POST_FW_VERSION(uint32_t status) {
  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2) + NOC_INSTANCE_OFFSET);
  ptr[0] = status;
}

inline void RISC_POST_STATUS(uint32_t status) {
  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
  ptr[0] = status;
}

inline void RISC_POST_DEBUG(uint32_t info) {
#ifdef ENABLE_ERISC_DEBUG_POST_CODES
  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
  ptr[0] = info;
#endif
}

typedef struct active_stream_info_t {
  uint8_t  stream_id;
  uint8_t  active_streams_idx;
  uint8_t  start_phase_num_cfg_regs;
  uint8_t  dram_prefetch_stream_need_restart;
  uint16_t unused0;
  uint16_t flags;
  uint32_t epoch_iters_remaining;
#ifdef DRAM_DECOUPLE
  uint32_t dram_decoupled;
#endif
} active_stream_info_t;


void set_risc_reset_vector();
uint8_t risc_streams_kernels_done();
void risc_context_switch();
void risc_initialize_tile_counters(uint32_t num_kernel_inputs, uint32_t num_kernel_outputs);

#define RISC_EPOCH_INFO_PTR ETH_EPOCH_INFO_PTR
#define RISC_L1_EPOCH_Q_PTR  ETH_L1_EPOCH_Q_PTR
#define RISC_DRAM_POLLING_CTRL_PTR ETH_DRAM_POLLING_CTRL_PTR
#define RISC_EPOCH_RUNTIME_CONFIG_PTR ETH_EPOCH_RUNTIME_CONFIG_PTR

void ApplicationTask(void);

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif


#endif
