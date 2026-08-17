// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal::test_utils {

// Thread-safe collector for deterministic realtime-profiler tests. Device
// completion is established independently by the command queue; this helper
// waits only for the asynchronous host callback to expose the expected device
// records. The timeout is test orchestration and is never used as a duration.
class RealtimeProfilerRecordCollector {
public:
    struct WaitResult {
        bool complete = false;
        uint64_t host_dropped = 0;
    };

    void consume(const experimental::ProgramRealtimeRecordBatch& batch) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            host_dropped_ += batch.dropped;
            records_.insert(records_.end(), batch.records.begin(), batch.records.end());
        }
        cv_.notify_all();
    }

    WaitResult wait_for_runtime_ids(const std::set<uint32_t>& expected, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool complete = cv_.wait_for(lock, timeout, [&] {
            std::set<uint32_t> observed;
            for (const auto& record : records_) {
                if (expected.contains(record.runtime_id)) {
                    observed.insert(record.runtime_id);
                }
            }
            return observed == expected;
        });
        return {.complete = complete, .host_dropped = host_dropped_};
    }

    WaitResult wait_for_record_count(
        uint32_t runtime_id, std::size_t expected_count, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool complete = cv_.wait_for(lock, timeout, [&] {
            std::size_t observed = 0;
            for (const auto& record : records_) {
                observed += record.runtime_id == runtime_id;
            }
            return observed >= expected_count;
        });
        return {.complete = complete, .host_dropped = host_dropped_};
    }

    // ProgramRealtimeRecord::kernel_sources contains non-owning spans. A copied
    // record is valid only while the owning MetalContext is alive.
    std::vector<experimental::ProgramRealtimeRecord> records() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<experimental::ProgramRealtimeRecord> records_;
    uint64_t host_dropped_ = 0;
};

}  // namespace tt::tt_metal::test_utils
