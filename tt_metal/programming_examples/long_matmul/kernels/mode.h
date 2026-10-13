// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// LONG_MATMUL_MODE, passed by the host as JIT defines. "legacy" reproduces how the published
// tt-ember results were measured; "updated" (the default) fixes the effects found since. The host
// README lists every difference.
#if defined(LONG_MATMUL_LEGACY)
constexpr bool kLegacyMode = true;
#else
constexpr bool kLegacyMode = false;
#endif

// In updated mode an idle kernel still does its real work for the first two steps, so both halves
// of each double-buffered circular buffer hold real data, which the idle loop then recycles. In
// legacy mode an idle kernel never does real work and the buffers keep whatever L1 held before.
constexpr uint32_t kRealStepsWhenIdle = kLegacyMode ? 0 : 2;
