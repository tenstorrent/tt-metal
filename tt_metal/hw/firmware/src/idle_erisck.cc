// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <cstdint>

#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "dataflow_api.h"
#include "noc_addr_ranges_gen.h"

#include <kernel_includes.hpp>

uint8_t noc_index = NOC_INDEX;
//inline void RISC_POST_STATUS(uint32_t status) {
//  volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
//  ptr[0] = status;
//}
void kernel_launch() {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");
    firmware_kernel_common_init((void tt_l1_ptr *)MEM_IERISC_INIT_LOCAL_L1_BASE);

    noc_local_state_init(noc_index);

    kernel_main();
}
