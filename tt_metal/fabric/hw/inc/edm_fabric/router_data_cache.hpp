// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void router_invalidate_l1_cache() {
    if constexpr (RISC_CPU_DATA_CACHE_ENABLED) {
        invalidate_l1_cache();
    }
}
