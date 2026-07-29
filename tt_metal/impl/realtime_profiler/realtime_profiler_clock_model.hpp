// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <tt-metalium/experimental/realtime_profiler.hpp>

namespace tt::tt_metal {

// One completed host<->device clock handshake, as measured by RealtimeProfilerClockSync.
struct ClockSyncSample {
    std::chrono::steady_clock::time_point host_time;  // read immediately before the handshake was issued
    std::chrono::nanoseconds rtt{};                   // measured round trip of the handshake
    uint64_t device_ticks = 0;                        // device WALL_CLOCK captured inside that round trip
};

// The host's belief about one chip's clock: the affine device->host mapping, how much to trust it, and the policy for
// keeping it current. Knows nothing about how a handshake is performed; callers hand it ClockSyncSamples.
//
// The mapping is host_ns = (device_ticks - device_cycle_offset) / frequency. frequency is fit once at bring-up and then
// held fixed; device_cycle_offset is re-anchored continuously to absorb the chip's drift away from that fixed slope.
class ClockModel {
public:
    // How far the fitted line sat from the samples it was fit to, in nanoseconds of device time, and how many of the
    // offered samples those were. Reported so bring-up can log it; the model itself does not act on it.
    struct FitResidual {
        double rms_ns = 0.0;
        double max_ns = 0.0;
        size_t num_samples_fitted = 0;   // regressed after discarding slow round trips
        size_t num_samples_offered = 0;  // handed to fit()
    };

    // Establishes the commanded clock frequency (AICLK) as the starting mapping, before any handshake has happened.
    // Every later step only refines it, so frequency() is positive for the model's whole life.
    void seed_frequency(double frequency);

    // Fits frequency and an initial anchor by least squares over a batch of bring-up samples. Empty when there were
    // fewer than two to regress: the seeded frequency then stands, though a lone sample is still anchored on.
    std::optional<FitResidual> fit(
        std::span<const ClockSyncSample> samples, std::chrono::steady_clock::time_point host_start);

    // Offers a handshake as the new anchor, returning whether it was taken. A slow round trip places the anchor worse
    // than the drift it would correct, so a recent anchor is better kept; once that anchor has itself gone stale, a
    // loose anchor beats unbounded drift and anything is accepted. The anchor goes at the round trip's midpoint --
    // minimax placement, error <= rtt/2, assuming nothing about the two legs being equal.
    bool try_reanchor(const ClockSyncSample& sample);

    // The mapping, in the form published to profiler consumers.
    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping() const;

    [[nodiscard]] double frequency() const { return frequency_; }

    // Round trip the standing anchor was placed with. A fresh probe that matches it is as good as the mapping has
    // been getting, so callers gathering probes can stop looking once they have one.
    [[nodiscard]] std::chrono::nanoseconds anchor_rtt() const { return rtt_; }

    // False until a fit or a resync has placed an anchor; the mapping is meaningless before then.
    [[nodiscard]] bool is_anchored() const { return last_reanchor_at_.has_value(); }

private:
    // device_cycle_offset is expressed against the steady_clock epoch, which is the domain the public
    // ProgramRealtimeClockSync mapping is documented in.
    void set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks);

    double frequency_ = 0.0;           // device cycles per host ns; positive from seed_frequency() onwards
    int64_t device_cycle_offset_ = 0;  // device_ticks = frequency * host_ns + device_cycle_offset
    std::chrono::nanoseconds rtt_{};   // round trip of the last accepted handshake; half of it bounds anchor placement
    std::optional<std::chrono::steady_clock::time_point> last_reanchor_at_;
};

}  // namespace tt::tt_metal
