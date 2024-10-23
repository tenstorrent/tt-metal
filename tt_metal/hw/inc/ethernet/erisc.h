// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "noc_nonblocking_api.h"

void (*rtos_context_switch_ptr)();
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

namespace internal_ {
inline __attribute__((always_inline))
void risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

inline __attribute__((always_inline))
void disable_erisc_app() { flag_disable[0] = 0; }
}  // namespace internal_
