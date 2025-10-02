// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"

#if !defined(COOPERATIVE_ERISC)
#include "tt_metal/lite_fabric/hw/inc/kernel_api.hpp"
#endif

#include "eth_fw_api.h"

inline void (*rtos_context_switch_ptr)();
inline void (*toggle_macpcs_ptr)(uint32_t);
#if defined(ARCH_BLACKHOLE)
volatile inline uint32_t* flag_disable = (uint32_t*)(GET_MAILBOX_ADDRESS_DEV(aerisc_run_flag));
#else
volatile inline uint32_t* flag_disable = (uint32_t*)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
#endif

namespace internal_ {
inline __attribute__((always_inline)) void risc_context_switch() {
#if defined(COMPILE_FOR_ERISC)
#if defined(COOPERATIVE_ERISC)
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
#elif defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    // Only NoC0
    ncrisc_noc_full_sync<1>();
    service_eth_msg();
    update_boot_results_eth_link_status_check();
    ncrisc_noc_counters_init<1>();
#endif
#endif
}

inline __attribute__((always_inline)) void risc_context_switch_without_noc_sync() {
#if defined(COMPILE_FOR_ERISC)
#if defined(COOPERATIVE_ERISC)
    rtos_context_switch_ptr();
#elif defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    update_boot_results_eth_link_status_check();
#elif defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 1
    lite_fabric::service_lite_fabric_channels();
#endif
#endif
}

inline __attribute__((always_inline)) void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
