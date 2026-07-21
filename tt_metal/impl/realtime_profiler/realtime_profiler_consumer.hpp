// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal {

// Internal sink for real-time profiler output. A plain record consumer is constructed directly with a record callback
// (this is what user-registered callbacks become); built-in sinks (e.g. Tracy) subclass and override on_records. Each
// record carries its own host<->device clock mapping (frequency + device_cycle_offset), so consumers need no separate
// clock-sync stream. on_records runs on that consumer's single delivery thread.
class RealtimeProfilerConsumer {
public:
    RealtimeProfilerConsumer() = default;
    explicit RealtimeProfilerConsumer(tt::tt_metal::experimental::ProgramRealtimeProfilerCallback record_callback) :
        record_callback_(std::move(record_callback)) {}
    virtual ~RealtimeProfilerConsumer() = default;

    virtual void on_records(const tt::tt_metal::experimental::ProgramRealtimeRecordBatch& batch) {
        if (record_callback_) {
            record_callback_(batch);
        }
    }

private:
    tt::tt_metal::experimental::ProgramRealtimeProfilerCallback record_callback_;
};

}  // namespace tt::tt_metal
