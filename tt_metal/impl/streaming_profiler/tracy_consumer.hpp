// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

#if defined(TRACY_ENABLE)
// Called from one consumer thread only. Never destroyed (the Service holding it is indestructible), so its per-core
// Tracy contexts are never torn down after Tracy's own shutdown.
class TracyConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::All>;

    TracyConsumer();
    TracyConsumer(const TracyConsumer&) = delete;
    TracyConsumer& operator=(const TracyConsumer&) = delete;
    void operator()(const Batch& batch);

private:
    using Core = experimental::streaming_profiler::Core;
    struct Lane {
        TracyTTCtx ctx = nullptr;
        uint32_t thread = 0;
        uint32_t processor = 0;
    };
    struct CoreEntry {
        TracyTTCtx ctx = nullptr;
        std::array<uint32_t, kProcessorCount> thread{};
        uint8_t named = 0;
    };
    struct SrclocEntry {
        const char* name = nullptr;
        uint64_t key = 0;
        const tracy::SourceLocationData* srcloc = nullptr;
    };

    int64_t timeline_ns(int64_t tsc) const;
    Lane lane(const Core& core);
    const tracy::SourceLocationData* srcloc(std::string_view name, uint32_t processor);
    const tracy::SourceLocationData* srcloc_slow(std::string_view name, uint32_t processor);
    void push_zone(const Core& core, std::string_view name, int64_t start_tsc, int64_t end_tsc);
    void push_marker(
        const Core& core, std::string_view name, int64_t tsc, uint32_t runtime_id, std::span<const uint64_t> values);

    int64_t anchor_tracy_ = 0;
    uint64_t lane_key_ = ~uint64_t{0};
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // Keyed by the name's address, which a callback's name strings keep while it lives, and probed in place: a
    // formatted-string map lookup per record was most of the sink's cost.
    std::vector<SrclocEntry> srcloc_table_;
    size_t srcloc_count_ = 0;
    std::unordered_map<std::string, const tracy::SourceLocationData*> srclocs_;
    // Tracy keeps the pointers for the rest of the process, and a deque never moves its elements.
    std::deque<std::string> srcloc_names_;
    std::deque<tracy::SourceLocationData> srcloc_data_;
};
#endif

}  // namespace tt::tt_metal::streaming_profiler
