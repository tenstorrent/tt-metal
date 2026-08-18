// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdev/realtime_profiler_msgs.h"

template <uint32_t Width>
constexpr bool realtime_profiler_modular_ge(uint32_t value, uint32_t target) {
    static_assert(Width > 1 && Width < 32);
    constexpr uint32_t shift = 32 - Width;
    const uint32_t shifted_difference = (value - target) << shift;
    return static_cast<int32_t>(shifted_difference) >= 0;
}

static_assert(
    offsetof(realtime_profiler_msg_t, loss_device_ring) + sizeof(uint32_t) -
        offsetof(realtime_profiler_msg_t, loss_descriptor_full) ==
    10 * sizeof(uint32_t));
