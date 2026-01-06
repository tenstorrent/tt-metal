// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"

#include "eth_fw_api.h"

inline void (*rtos_context_switch_ptr)();
inline void (*toggle_macpcs_ptr)(uint32_t);
#if defined(ARCH_BLACKHOLE)
volatile inline uint32_t* flag_disable = (uint32_t*)(GET_MAILBOX_ADDRESS_DEV(aerisc_run_flag));
#else
volatile inline uint32_t* flag_disable = (uint32_t*)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
#endif

namespace internal_ {
inline __attribute__((always_inline)) void risc_context_switch([[maybe_unused]] bool skip_sync = false) {
#if defined(COMPILE_FOR_ERISC)
#if defined(COOPERATIVE_ERISC)
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
#elif defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    // Skip sync if kernel group is using Dynamic NOC. Firmware doesn't define NOC_MODE so
    // this needs to be checked at runtime.
    // Note, <1> means only sync NOC0. NOC1 can be skipped because base firmware only uses NOC0
    if (!skip_sync) {
        ncrisc_noc_full_sync<1>();
    }
    service_eth_msg();
    update_boot_results_eth_link_status_check();

    // Not harmful to initialize the counters again
    ncrisc_noc_counters_init<1>();

#if defined(NOC_MODE) && (NOC_MODE == DM_DYNAMIC_NOC)
    // Reprogram cmd bufs for dynamic NOC
    dynamic_noc_init();
    // Base firmware counters are shared in L1. No need
    // to reinit them using dynamic_noc_local_state_init()
#endif  // NOC_MODE == DM_DYNAMIC_NOC

#endif  // (COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)

#endif  // COMPILE_FOR_ERISC
}

inline __attribute__((always_inline)) void risc_context_switch_without_noc_sync() {
#if defined(COMPILE_FOR_ERISC)
#if defined(COOPERATIVE_ERISC)
    rtos_context_switch_ptr();
#elif defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    update_boot_results_eth_link_status_check();
#endif
#endif
}

inline __attribute__((always_inline)) void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
