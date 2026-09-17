// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>

// The host's clocks as the streaming profiler reads them: the TSC, which every record's host time is in (tenths of a
// nanosecond of it, the API's host_clock), its rate, and its relation to std::chrono::steady_clock as the host probe
// measures it.
namespace tt::tt_metal::streaming_profiler {

// The host TSC, fenced, in ticks.
int64_t tsc_now() noexcept;
// TSC ticks per nanosecond, measured once per process against CLOCK_MONOTONIC_RAW.
double tsc_ticks_per_ns();
// host_clock units (tenths of a nanosecond) per TSC tick.
double units_per_tsc();
int64_t clock_ns(clockid_t id);

// The host TSC on CLOCK_MONOTONIC, one line between two NTP slews: mono_ns = mono0 + (tsc - tsc0) * ns_per_tick.
struct SteadySegment {
    int64_t tsc0 = 0, mono0 = 0;
    double ns_per_tick = 0.0;
    bool ok = false;
    int64_t mono_of(int64_t tsc) const { return mono0 + std::llrint(static_cast<double>(tsc - tsc0) * ns_per_tick); }
};

// The host TSC on steady_clock as the host probe measures it: one segment for the process, readable from any thread
// and cached per thread. No capture is involved, so the API's steady_time() reads it with no device open.
class SteadyView {
public:
    static void set(const SteadySegment& segment) noexcept;
    // A TSC/CLOCK_MONOTONIC pair taken here stands in until a probe publishes a segment.
    static int64_t mono_ns(int64_t tsc) noexcept;
};

}  // namespace tt::tt_metal::streaming_profiler
