// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

// Writes one CSV row of device kernel durations per op execution, built from the "-KERNEL" zones. Trace replays run an
// op again under the same runtime id, so each execution is numbered by its order on each RISC.
class OpsCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::Zone>;
    explicit OpsCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Writes the CSV file. It must not run at the same time as operator(), which adds the rows.
    void write_csv();

private:
    using Time = std::chrono::steady_clock::time_point;
    struct Span {
        Time start = Time::max(), end = Time::min();
    };
    // Kernel start and end, in device cycles for the CYCLE columns and on the host timeline for every [ns] column. Only
    // the Tensix zones' cycles are used, because an eth tile's counter can be seconds away from the Tensix tiles'.
    struct OpAgg {
        uint64_t start_cycles = UINT64_MAX, end_cycles = 0;
        Time last_kernel_start = Time::min();
        std::array<Span, kProcessorCount> processors{};
        std::map<uint32_t, Span> cores;  // by physical (y << 16) | x
    };

    std::unique_ptr<FILE, FileClose> file_;
    bool header_written_ = false;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, OpAgg> ops_;  // (chip, runtime host-id, execution)
    // How many executions of each op each RISC has completed, keyed by (chip, core, processor, runtime host-id). It is
    // kept across captures, because every capture's rows go to the same file.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> executions_seen_;
};

}  // namespace tt::tt_metal::streaming_profiler
