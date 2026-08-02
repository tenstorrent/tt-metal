// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {

class IDevice;

struct ClockProbe {
    std::chrono::steady_clock::time_point host_time;  // taken immediately before the counter read
    std::chrono::nanoseconds bracket{};
    uint64_t device_ticks = 0;
};

// host_ns = (device_ticks - device_cycle_offset) / frequency. frequency is fit once at bring-up and held fixed;
// device_cycle_offset is re-anchored whenever a probe finds the mapping has moved.
class RealtimeProfilerClockModel {
public:
    struct FitResidual {
        double rms_ns = 0.0;  // in device time
        double max_ns = 0.0;
        size_t num_probes_fitted = 0;   // regressed after discarding wide brackets
        size_t num_probes_offered = 0;  // handed to fit()
    };

    void seed_frequency(double frequency);

    // Empty when there were fewer than two probes to regress; the seeded frequency then stands, though a lone probe
    // is still anchored on.
    std::optional<FitResidual> fit(
        std::span<const ClockProbe> probes, std::chrono::steady_clock::time_point host_start);

    // Taken when it would leave the mapping better placed than the miss it just measured, or than the standing
    // anchor is placed -- so a slow read rejects itself, its own miss being under the resolution it would bring.
    bool try_reanchor(const ClockProbe& probe);

    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping() const;

    [[nodiscard]] double frequency() const { return frequency_; }

    // Zero before the first anchor.
    [[nodiscard]] std::chrono::nanoseconds anchor_bracket() const { return bracket_; }

    [[nodiscard]] std::chrono::nanoseconds last_drift() const { return last_drift_; }

    // What the standing mapping is out by at `probe`, without moving the anchor.
    [[nodiscard]] std::chrono::nanoseconds drift_at(const ClockProbe& probe) const;

    [[nodiscard]] bool is_anchored() const { return last_reanchor_at_.has_value(); }

private:
    void set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks);

    double frequency_ = 0.0;
    int64_t device_cycle_offset_ = 0;
    std::chrono::nanoseconds bracket_{};
    std::chrono::nanoseconds last_drift_{};
    // Where the anchor was placed if the last probe was taken, or the miss it found if it was not.
    std::chrono::nanoseconds residual_{};
    std::optional<std::chrono::steady_clock::time_point> last_reanchor_at_;
};

// Reads the profiler core's cycle counter and drives a RealtimeProfilerClockModel from the probes. Nothing runs on
// device for this: the NOC serves the counter directly, so a read cannot be delayed by the profiler core's push loop.
//
// bring_up() runs before the receiver thread starts and every later call belongs to that thread, so nothing here is
// shared and the mapping needs no publication protocol.
class RealtimeProfilerClockSync {
public:
    // How often the owner should call resync(). A DVFS step goes uncorrected for at most this long, and the error it
    // leaves is ~5200ppm times that. Measured worst case: 10ms -> 14.2us, 1ms -> 5.7us, 250us -> 2.8us.
    static constexpr auto kSyncInterval = std::chrono::milliseconds(1);

    // Steps arrive in bursts; running this fast all the time would spend its probes on the quiet vast majority.
    static constexpr auto kBurstSyncInterval = std::chrono::microseconds(250);
    static constexpr auto kBurstWindow = std::chrono::milliseconds(5);

    // A miss this many times larger than the probe measuring it is a step, not noise: the ratio is 1 at the
    // acceptance boundary and above 10 for one throttler tick of dip.
    static constexpr int kExcursionDriftRatio = 4;

    // `profiler_core` is the reserved tensix running the profiler kernels on `device`.
    RealtimeProfilerClockSync(ContextId context_id, IDevice* device, CoreCoord profiler_core);
    ~RealtimeProfilerClockSync();

    // The commanded AICLK stands if every fit attempt fails, so the mapping is usable either way.
    void bring_up();

    // One probe, offered to the model. False only when the device did not answer.
    bool resync();

    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping() const { return model_.mapping(); }

    [[nodiscard]] double frequency() const { return model_.frequency(); }

    [[nodiscard]] bool saw_excursion() const {
        return model_.last_drift() >= kExcursionDriftRatio * model_.anchor_bracket() / 2;
    }

private:
    // False when there is no usable cache entry, or the anchor probe failed and a full fit is needed.
    bool try_cached_calibration();
    // False when the fit is not worth keeping and another attempt is likely to beat it.
    bool calibrate();
    void configure_clock_read_path();
    std::optional<ClockProbe> probe();
    // Ranked, not thresholded: under record load the whole bracket distribution shifts.
    std::optional<ClockProbe> best_of(int probes);

    ContextId context_id_;
    uint32_t chip_id_ = 0;
    // Resolved once so the resolve does not sit inside the bracket.
    CoreCoord profiler_core_virtual_;
    uint32_t wall_clock_addr_lo_ = 0;
    uint32_t wall_clock_addr_hi_ = 0;
    // A UC window makes a probe a load rather than a UMD call, whose TLB reconfiguration would sit inside the
    // bracket and widen it by ~450ns.
    std::unique_ptr<tt::umd::TlbWindow> clock_tlb_;
    volatile uint32_t* mapped_clock_lo_ = nullptr;
    volatile uint32_t* mapped_clock_hi_ = nullptr;
    // The high word only advances when the low word wraps, every 3.2s at ~1.35GHz.
    uint32_t cached_clock_hi_ = 0;
    uint32_t last_clock_lo_ = 0;
    std::chrono::steady_clock::time_point last_probe_at_;

    RealtimeProfilerClockModel model_;
};

}  // namespace tt::tt_metal
