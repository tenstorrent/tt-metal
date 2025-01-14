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
#include "noc_nonblocking_api.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <kernel_includes.hpp>
#if defined ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#include "remote_circular_buffer_api.h"
#endif
#include "debug/dprint.h"

bool skip_kernel() {
#ifdef SKIP_KERNEL
    volatile tt_l1_ptr uint32_t* p_tensor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(P_TENSOR_ADDR);
    uint32_t p_tensor_data = *p_tensor;
    DPRINT << "ADDR: " << P_TENSOR_ADDR << " BRISC: " << p_tensor_data << ENDL();

    if (p_tensor_data == 0) {
        DPRINT << "Skipping BRISC kernel" << ENDL();
        return true;
    }
    return false;
#else
    return false;
#endif
}

void kernel_launch(uint32_t kernel_base_addr) {
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
    wait_for_go_message();
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < end_time);
#endif
#else
    extern uint32_t __kernel_init_local_l1_base[];
    extern uint32_t __fw_export_end_text[];
    do_crt1((uint32_t tt_l1_ptr
                 *)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base - (uint32_t)__fw_export_end_text));

    if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
        noc_local_state_init(NOC_INDEX);
    }
#ifdef ALIGN_LOCAL_CBS_TO_REMOTE_CBS
    ALIGN_LOCAL_CBS_TO_REMOTE_CBS
#endif
    wait_for_go_message();
    if (!skip_kernel()) {
        DeviceZoneScopedMainChildN("BRISC-KERNEL");
        kernel_main();
    }
#endif
}
