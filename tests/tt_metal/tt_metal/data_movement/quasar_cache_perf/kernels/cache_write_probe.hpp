// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// Progress record written by cache_write_perf into the watcher ring buffer.
// Watcher must be off; that buffer is otherwise unused. phase is stored last.
inline constexpr std::uint32_t kCacheWriteProbeMagic = 0xCace0124u;

enum class CacheWriteProbePhase : std::uint32_t {
    Entered = 1,
    BeforeStores = 2,
    BeforeFlush = 3,
    AfterFlush = 4,
    Done = 5,
};

struct CacheWriteProbe {
    std::uint32_t magic;
    std::uint32_t probe_id;
    std::uint32_t mode;
    std::uint32_t size_bytes;
    std::uint32_t num_iterations;
    std::uint32_t iteration;
    std::uint32_t flush_addr;
    std::uint32_t phase;
};

static_assert(sizeof(CacheWriteProbe) == 32);
static_assert(offsetof(CacheWriteProbe, phase) == 28);
