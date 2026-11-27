// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;
inline constexpr std::array<const char*, kProcessorCount> kProcessorNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC_0", "ERISC_1"};

constexpr bool is_ethernet(experimental::streaming_profiler::Processor processor) {
    return processor == experimental::streaming_profiler::Processor::ERISC0 ||
           processor == experimental::streaming_profiler::Processor::ERISC1;
}

struct FileClose {
    void operator()(FILE* file) const { std::fclose(file); }
};

}  // namespace tt::tt_metal::streaming_profiler
