// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared host<->kernel definitions for the Quasar scoped-lock cache-op tests, covering every type that
// exposes a scoped lock.
#pragma once

#include <cstdint>

enum class ScopedLockCacheMode : uint32_t {
    InvalidateOnAcquire = 0,  // the lock invalidates the held entries on acquire (both read and write locks)
    FlushOnRelease = 1,       // the lock flushes the held entries on release (write lock only)
};

// One probe word per 64 B cache line, so a per-line claim is a per-line measurement.
constexpr uint32_t SCOPED_LOCK_CACHE_WORDS_PER_LINE = 16;  // 64 B / sizeof(uint32_t)
constexpr uint32_t SCOPED_LOCK_CACHE_NUM_LINES = 8;
constexpr uint32_t SCOPED_LOCK_CACHE_REGION_WORDS = SCOPED_LOCK_CACHE_NUM_LINES * SCOPED_LOCK_CACHE_WORDS_PER_LINE;
constexpr uint32_t SCOPED_LOCK_CACHE_REGION_BYTES = SCOPED_LOCK_CACHE_REGION_WORDS * sizeof(uint32_t);

// Sentinels: line l is seeded to OLD_BASE + l in TL1 and later overwritten with NEW_BASE + l.
constexpr uint32_t SCOPED_LOCK_CACHE_OLD_BASE = 0xBB00;
constexpr uint32_t SCOPED_LOCK_CACHE_NEW_BASE = 0xAA00;
