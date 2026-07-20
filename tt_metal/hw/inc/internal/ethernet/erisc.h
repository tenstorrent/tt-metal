// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
volatile inline uint32_t* flag_disable = (uint32_t*)(uintptr_t)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
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
    aerisc_context_switch();

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
    // Drain the eth mailbox so FW-level messages (e.g. the port-down request injected by
    // run_link_control) are processed even while the fabric router is the one yielding through this
    // path. The full risc_context_switch() services the mailbox via aerisc_context_switch(), but the
    // router's steady-state loop uses this without_noc_sync variant, which otherwise never would.
    //
    // service_eth_msg() can dispatch FW handlers (port action, MAC/PCS reinit, link recovery) that use
    // NOC0. The fabric router runs in dedicated-NOC mode, so it keeps PRIVATE software shadow counters:
    // base-FW NOC0 use here would desync them and hang the router on resume. Bracket the yield exactly
    // like the full risc_context_switch() does -- flush the router's in-flight NOC0 first, then realign
    // its shadow counters after. (TEST-ONLY: this makes the "without_noc_sync" path do a full sync.)
    update_boot_results_eth_link_status_check();
    recover_eth_link_if_down();
    // [PKTMODE-PROBE] TX/RX counter push DISABLED -- the ring buffer is used for the 7-point config-register
    // snapshots instead (they would be flooded/evicted by a per-context-switch TX/RX push). Re-enable this
    // (and disable the snapshots) to go back to live TX/RX counter tracking.
    // fabric_dbg_ringbuf_push_txrx_counts();
#endif
#endif
}

inline __attribute__((always_inline)) void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
