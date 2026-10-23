// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The record the external test's kernel publishes to each destination, compiled by the device compiler for the kernel
// and by the host compiler for the profiled process's callback and the external process's reader.
#pragma once

#include <cstdint>

#include <hostdevcommon/streaming_profiler_types.hpp>

namespace external_test {

// Which of the test's two call sites sent it, the iteration that sent it, and a value derived from both.
struct Reading {
    uint32_t destination;  // 0 this process alone, 1 the external process too
    uint32_t iteration;
    uint32_t value;
};
struct Name {
    tt::tt_metal::experimental::streaming_profiler::String name;
};

constexpr uint32_t reading_value(uint32_t destination, uint32_t iteration) { return iteration * 7 + destination; }

}  // namespace external_test
