// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_csv_report_handler.hpp"

#include <tt-logger/tt-logger.hpp>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string_view>
#include <system_error>

#include <tt_metal.hpp>

#include "profiler/profiler_paths.hpp"

namespace tt::tt_metal {

namespace {

constexpr std::string_view kRtDevicePerfReportHeader =
    "GLOBAL CALL COUNT,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,DEVICE ID,DEVICE ARCH,OP NAME,CORE "
    "COUNT,AVAILABLE WORKER CORE COUNT,DEVICE KERNEL DURATION [ns],DEVICE KERNEL START CYCLE,DEVICE KERNEL END "
    "CYCLE,OP TO OP LATENCY [ns],OP TO OP LATENCY BR/NRISC START [ns]\n";

}  // namespace

RealtimeProfilerCsvReportHandler::RealtimeProfilerCsvReportHandler() {
    const std::filesystem::path logs_dir = get_profiler_logs_dir();
    std::error_code ec;
    std::filesystem::create_directories(logs_dir, ec);
    if (ec) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Quick mode could not create profiler logs directory '{}': {}",
            logs_dir.string(),
            ec.message());
        return;
    }

    const std::filesystem::path report_path = logs_dir / std::string(RT_DEVICE_PERF_REPORT_NAME);
    csv_file_.open(report_path, std::ios_base::trunc);
    if (!csv_file_.is_open()) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Quick mode could not open perf report '{}' for writing",
            report_path.string());
        return;
    }
    csv_file_ << kRtDevicePerfReportHeader;

    callback_handle_ = tt::RegisterProgramRealtimeProfilerCallback(
        [this](const tt::ProgramRealtimeRecordBatch& batch) { HandleBatch(batch); });
}

RealtimeProfilerCsvReportHandler::~RealtimeProfilerCsvReportHandler() {
    if (callback_handle_.has_value()) {
        tt::UnregisterProgramRealtimeProfilerCallback(*callback_handle_);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (csv_file_.is_open()) {
        csv_file_.flush();
        csv_file_.close();
    }

    if (dropped_records_ != 0) {
        log_warning(
            tt::LogMetal,
            "[Real-time profiler] Quick mode dropped {} record(s); the perf report is missing those ops. "
            "The callback could not keep up with incoming records.",
            dropped_records_);
    }
}

void RealtimeProfilerCsvReportHandler::HandleBatch(const tt::ProgramRealtimeRecordBatch& batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    dropped_records_ += batch.dropped;
    if (batch.dropped > 0) {
        // A drop broke per-chip continuity; forget each chip's predecessor so its next op-to-op is 0
        // rather than an inflated gap spanning the lost records.
        for (auto& prev_end : prev_end_cycle_by_chip_) {
            prev_end.reset();
        }
    }

    auto append_u64 = [this](uint64_t v) {
        char tmp[20];  // UINT64_MAX is 20 digits; std::to_chars writes no null terminator
        auto [p, ec] = std::to_chars(tmp, tmp + sizeof(tmp), v);
        buffer_.append(tmp, static_cast<size_t>(p - tmp));
    };

    buffer_.clear();
    for (const auto& record : batch.records) {
        if (record.end_timestamp < record.start_timestamp) {
            continue;
        }

        const uint64_t duration_ns = static_cast<uint64_t>(
            std::llround(static_cast<double>(record.end_timestamp - record.start_timestamp) / record.frequency));

        if (record.chip_id >= prev_end_cycle_by_chip_.size()) {
            prev_end_cycle_by_chip_.resize(record.chip_id + 1);
        }
        const std::optional<uint64_t> prev_end = prev_end_cycle_by_chip_[record.chip_id];
        prev_end_cycle_by_chip_[record.chip_id] = record.end_timestamp;

        const uint32_t global_call_count = detail::EncodePerDeviceProgramID(record.runtime_id, record.chip_id);

        // Positional against kRtDevicePerfReportHeader; the empty fields are columns RT can't fill.
        append_u64(global_call_count);
        buffer_.append(",,,", 3);
        append_u64(record.chip_id);
        buffer_.append(",,,,,", 5);
        append_u64(duration_ns);
        buffer_.push_back(',');
        append_u64(record.start_timestamp);
        buffer_.push_back(',');
        append_u64(record.end_timestamp);
        buffer_.push_back(',');
        // first op-to-op on a chip is 0, a negative gap leaves the field blank.
        if (!prev_end.has_value()) {
            buffer_.push_back('0');
        } else if (record.start_timestamp >= *prev_end) {
            append_u64(static_cast<uint64_t>(
                std::llround(static_cast<double>(record.start_timestamp - *prev_end) / record.frequency)));
        }
        buffer_.append(",\n", 2);
    }

    if (!buffer_.empty() && csv_file_.is_open()) {
        csv_file_.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    }
}

}  // namespace tt::tt_metal
