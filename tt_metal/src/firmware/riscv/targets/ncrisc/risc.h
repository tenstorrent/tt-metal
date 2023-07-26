#ifndef _NCRISC_H_
#define _NCRISC_H_

#include <stdint.h>

#include "noc_parameters.h"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"


typedef struct active_stream_info_t {
  uint8_t  stream_id;
  uint8_t  unused0;
  uint16_t flags;
} active_stream_info_t;

inline __attribute__((always_inline)) void RISC_POST_STATUS(uint32_t status) {
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)(NOC_CFG(ROUTER_CFG_2));
  ptr[0] = status;
}

inline __attribute__((always_inline)) void RISC_POST_DEBUG(uint32_t info) {
#ifdef ENABLE_NCRISC_DEBUG_POST_CODES
  volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)(NOC_CFG(ROUTER_CFG_2));
  ptr[0] = info;
#endif
}

uint8_t risc_streams_kernels_done();

#define RISC_EPOCH_INFO_PTR EPOCH_INFO_PTR

#endif
