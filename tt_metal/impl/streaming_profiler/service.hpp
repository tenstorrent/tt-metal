// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include <tt/insight/host/clock_sync_self_test.hpp>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt {
class Cluster;
namespace llrt {
class RunTimeOptions;
}
}  // namespace tt

namespace tt::tt_metal::streaming_profiler {

// What serves the public API (tt-metalium/experimental/streaming_profiler.hpp) beside tt-insight: Metal's core of each
// tile, for Record::core(), and the sinks rtoptions select (the Tracy sink and the CSV writers).
class Service {
public:
    // A capture of the chips is about to start: their cores are known to the records from here on, and with the first
    // capture the sinks' callbacks are registered.
    void attach_capture(
        const std::vector<ChipId>& chip_ids, const tt::Cluster& cluster, const tt::llrt::RunTimeOptions& rtoptions);
    // A capture of `devices` devices has stopped. `sync_check` is what its sync check measured, if it ran one, which
    // is logged. With the last capture the sinks' callbacks end, each with every record published before, and the file
    // sinks are written.
    void detach_capture(size_t devices, const std::optional<tt::insight::ClockSyncErrorDistribution>& sync_check);
    bool is_active() const;

private:
    struct Sink {
        std::function<experimental::streaming_profiler::Callback()> subscribe;
        std::function<void()> write_file;
        experimental::streaming_profiler::Callback callback;
    };
    mutable std::mutex mu_;
    int captures_ = 0;
    bool sinks_made_ = false;
    std::vector<Sink> sinks_;
};

// The process's Service, never destroyed.
Service& service();

}  // namespace tt::tt_metal::streaming_profiler
