// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"

#if !defined(COOPERATIVE_ERISC)
#include "tt_metal/lite_fabric/hw/inc/kernel_api.hpp"
#endif

inline void (*rtos_context_switch_ptr)();
inline void (*toggle_macpcs_ptr)(uint32_t);
volatile inline uint32_t* flag_disable = (uint32_t*)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

namespace internal_ {
inline __attribute__((always_inline)) void risc_context_switch() {
#ifdef COOPERATIVE_ERISC
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
#endif
}

inline __attribute__((always_inline)) void risc_context_switch_without_noc_sync() {
#if defined(COOPERATIVE_ERISC)
    rtos_context_switch_ptr();
#else
    lite_fabric::service_lite_fabric_channels();
#endif
}

inline __attribute__((always_inline)) void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
