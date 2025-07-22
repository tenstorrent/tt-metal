// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"
#include <cstdint>

inline void (*rtos_context_switch_ptr)();
inline void (*toggle_macpcs_ptr)(uint32_t);

#if defined(COMPILE_FOR_AERISC)
inline volatile std::uint32_t* gEnableFwFlag = (volatile uint32_t*)(MEM_AERISC_RUN_FW_FLAG);
#else
inline volatile std::uint32_t* gEnableFwFlag = (uint32_t*)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
#endif

namespace internal_ {
inline __attribute__((always_inline)) void risc_context_switch() {
#ifdef COOPERATIVE_ERISC
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
#endif
}

inline __attribute__((always_inline)) void risc_context_switch_without_noc_sync() { rtos_context_switch_ptr(); }

inline __attribute__((always_inline)) void enable_erisc_app() { gEnableFwFlag[0] = 1; }

inline __attribute__((always_inline)) void disable_erisc_app() { gEnableFwFlag[0] = 0; }
}  // namespace internal_
