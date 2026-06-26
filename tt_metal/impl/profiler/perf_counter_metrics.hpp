// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/experimental/profiler.hpp>

namespace tracy {
class TTDeviceMarker;
}

namespace tt::tt_metal::profiler_perf_counters {

// Per-op hardware-perf-counter columns, computed on the C++ fast post-process path.
//
// This is the C++ port of tools/tracy/perf_counter_analysis.py::compute_perf_counter_metrics.
// It exists so a counter-capture run no longer has to fall back to the legacy Python path
// that does a full pandas read of profile_log_device.csv (which OOMs at mesh scale). The
// per-(op,core,counter) values are read straight from the in-memory device markers, so no
// CSV is parsed and memory stays bounded (~ops × cores × counter-types of just the 9090
// markers, not every marker on every core).
struct PerfCounterColumns {
    // Column names that have at least one finite value across all ops, in the canonical
    // PERF_COUNTER_CSV_HEADERS order (so the emitted CSV matches the legacy column layout).
    std::vector<std::string> active_headers;
    // op UID -> (column name -> formatted value). Only finite values are present; a missing
    // entry is written as an empty CSV cell (== pandas NaN / absent on the consumer side).
    std::map<experimental::ProgramExecutionUID, std::map<std::string, std::string>> values_per_uid;

    bool empty() const { return active_headers.empty(); }
};

// Aggregate the perf-counter markers (marker_id == PERF_COUNTER_PROFILER_ID) into per-op
// utilization/efficiency columns.
//   total_compute_cores      : chip logical grid core count (DeviceProfiler::max_compute_cores),
//                              the denominator for the grid-normalized "Avg X util on full grid"
//                              columns — matches the legacy path's deviceInfo.max_compute_cores.
//   kernel_cycles_by_uid     : per-op kernel duration in cycles (end - start), used to turn the
//                              grid-summed counts into the "Avg X util on full grid (%)" columns.
PerfCounterColumns computePerfCounterColumns(
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    uint32_t total_compute_cores,
    const std::map<experimental::ProgramExecutionUID, double>& kernel_cycles_by_uid);

}  // namespace tt::tt_metal::profiler_perf_counters
