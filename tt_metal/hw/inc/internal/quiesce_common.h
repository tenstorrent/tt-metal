// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"
#include "dev_mem_map.h"
#include "hostdev/dev_msgs.h"

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC) && !defined(HAL_BUILD) && \
    __has_include("registers/overlay_reg.h")
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"

FORCE_INLINE volatile tt_l1_ptr quiesce_msg_t* quiesce_slot() {
    return reinterpret_cast<volatile tt_l1_ptr quiesce_msg_t*>(
        static_cast<std::uintptr_t>(MEM_L1_UNCACHED_BASE) + static_cast<std::uintptr_t>(MEM_QUIESCE_BASE));
}

FORCE_INLINE bool quiesce_requested() { return quiesce_slot()->control != QUIESCE_CTRL_NONE; }
#else
constexpr FORCE_INLINE bool quiesce_requested() { return false; }
#endif
