// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "data_collection.hpp"

namespace tt::tt_metal {

// Real-time profiler "quick mode" perf report writer.
//
// Registers a single ProgramRealtimeProfilerCallback and streams incoming records into
// rt_device_perf_report.csv, whose rows are a column-subset of the full device profiler's
// cpp_device_perf_report.csv so the existing Python post-processing can consume it. Only the
// RT-derivable columns are populated (DEVICE KERNEL DURATION [ns], START/END CYCLE, OP TO OP
// LATENCY [ns]); the rest are left empty.
//
// Owned by MetalContext when TT_METAL_PROFILER_RT_QUICK is set.
//
// The profiler invokes the callback from one dedicated consumer thread per open mesh
// device, so the single-mesh case is single-threaded; the mutex only guards the shared buffer and
// file when multiple mesh devices are active at once.
class RealtimeProfilerCsvReportHandler {
public:
    RealtimeProfilerCsvReportHandler();
    ~RealtimeProfilerCsvReportHandler();

    RealtimeProfilerCsvReportHandler(const RealtimeProfilerCsvReportHandler&) = delete;
    RealtimeProfilerCsvReportHandler& operator=(const RealtimeProfilerCsvReportHandler&) = delete;

private:
    void HandleBatch(const tt::ProgramRealtimeRecordBatch& batch);

    std::mutex mutex_;
    std::ofstream csv_file_;
    std::string buffer_;
    std::vector<std::optional<uint64_t>> prev_end_cycle_by_chip_;
    uint64_t dropped_records_ = 0;
    std::optional<tt::ProgramRealtimeProfilerCallbackHandle> callback_handle_;
};

}  // namespace tt::tt_metal
