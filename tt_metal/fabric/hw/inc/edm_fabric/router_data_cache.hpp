// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"

template <bool ENABLE_RISC_CPU_DATA_CACHE>
FORCE_INLINE void router_invalidate_l1_cache() {
    if constexpr (ENABLE_RISC_CPU_DATA_CACHE) {
        invalidate_l1_cache();
    }
}
