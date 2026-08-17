// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR)

#include "dest_order.h"

#else

namespace dest_order {

inline __attribute__((always_inline)) void touch_unpack() {}

inline __attribute__((always_inline)) void touch_fpu() {}

inline __attribute__((always_inline)) void touch_sfpu() {}

inline __attribute__((always_inline)) void touch_pack() {}

}  // namespace dest_order

#endif
