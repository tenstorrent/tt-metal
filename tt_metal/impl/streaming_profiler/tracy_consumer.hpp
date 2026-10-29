// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

#if defined(TRACY_ENABLE)
// Pushes each batch's zones, events and timestamped data to Tracy, with one context per core and one row per RISC.
// It must be used from one thread only.
class TracyConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<
        experimental::streaming_profiler::Zone,
        experimental::streaming_profiler::TimestampedData,
        experimental::streaming_profiler::Event>;

    TracyConsumer();
    TracyConsumer(const TracyConsumer&) = delete;
    TracyConsumer& operator=(const TracyConsumer&) = delete;
    void operator()(const Batch& batch);

private:
    using Core = experimental::streaming_profiler::Core;
    struct Lane {
        TracyTTCtx ctx = nullptr;
        uint32_t thread = 0;
    };
    struct CoreEntry {
        TracyTTCtx ctx = nullptr;
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
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // An open-addressing table from a zone name and processor to its source location in srclocs_. It is keyed by the
    // name's address, which the zone-name registry keeps valid for the process, and probed in place.
    std::vector<SrclocEntry> srcloc_table_;
    size_t srcloc_count_ = 0;
    // The source locations handed to Tracy, keyed by name and color. Tracy keeps the pointers for the rest of the
    // process, and a map never moves its nodes.
    std::map<std::pair<std::string, uint32_t>, tracy::SourceLocationData> srclocs_;
};
#endif

}  // namespace tt::tt_metal::streaming_profiler
