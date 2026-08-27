// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared host<->kernel definitions for the Quasar scoped-lock cache-op tests. Included by both the host
// test and the cache kernels.
#pragma once

#include <cstdint>

enum class DfbCacheTestMode : uint32_t {
    FlushOnRelease = 0,       // the lock flushes the held entries on release (write lock only)
    InvalidateOnAcquire = 1,  // the lock invalidates the held entries on acquire (both read and write locks)
};
