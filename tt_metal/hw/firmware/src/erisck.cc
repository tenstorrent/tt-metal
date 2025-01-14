// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "noc_parameters.h"
#include "ethernet/dataflow_api.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "risc_attribs.h"
#include "tensix.h"
#include "tensix_types.h"
#include "tt_eth_api.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "stream_io_map.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#include <stdint.h>

extern "C" void wzerorange(uint32_t *start, uint32_t *end);

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

bool skip_kernel() {
#ifdef SKIP_KERNEL
    volatile tt_l1_ptr uint32_t* p_tensor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(P_TENSOR_ADDR);
    uint32_t p_tensor_data = *p_tensor;
    DPRINT << "ADDR: " << P_TENSOR_ADDR << " ERISC: " << p_tensor_data << ENDL();

    if (p_tensor_data == 0) {
        DPRINT << "Skipping BRISC kernel" << ENDL();
        return true;
    }
    return false;
#else
    return false;
#endif
}

extern "C" [[gnu::section(".start")]] void _start(uint32_t) {
    DeviceZoneScopedMainChildN("ERISC-KERNEL");

    // Clear bss, we write to rtos_context_switch_ptr just below.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    if (!skip_kernel()) {
        kernel_main();
    }
}
