// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
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

// The best round trip a handshake path is currently capable of, which is the target a probe burst works towards.
// Drops to any new minimum at once and rises only slowly, so one lucky fast round trip cannot set a bar later bursts
// can never clear, and a path that has genuinely slowed is followed within seconds.
class RttFloor {
public:
    void observe(std::chrono::nanoseconds rtt);
    // Zero until the first observation, which callers take as "no target yet" and probe without one.
    [[nodiscard]] std::chrono::nanoseconds value() const;
    // Whether a round trip is close enough to the floor that further probing would not measurably improve on it.
    [[nodiscard]] bool is_near(std::chrono::nanoseconds rtt) const;

private:
    // Held as a double because the per-observation rise is a fraction of a nanosecond at realistic round trips
    // (~0.3ns at a 1.2us floor). In integer nanoseconds it would truncate to zero every time, leaving the floor a
    // monotone minimum that ratchets down to the all-time best and never lets a burst exit early again.
    double floor_ns_ = 0.0;
};

// How well an initial fit matched its samples. Reported so bring-up can log it; the model does not act on it.
struct ClockFitQuality {
    bool ok = false;  // false when there were too few samples to regress and the seeded frequency was kept
    uint32_t num_samples = 0;
    double residual_rms_ns = 0.0;
    double residual_max_ns = 0.0;
};

// The host's belief about one chip's clock: the affine device->host mapping, how much to trust it, and the policy for
// keeping it current. Knows nothing about how a handshake is performed; callers hand it ClockSyncSamples.
//
// The mapping is host_ns = (device_ticks - device_cycle_offset) / frequency. frequency is fit once at bring-up and then
// held fixed; device_cycle_offset is re-anchored continuously to absorb the chip's drift away from that fixed slope.
class ClockModel {
public:
    // Establishes the commanded clock frequency (AICLK) as the starting mapping, before any handshake has happened.
    // Every later step only refines it, so frequency() is positive for the model's whole life.
    void seed_frequency(double frequency);

    // Fits frequency and an initial anchor by least squares over a batch of bring-up samples. Falls back to the seeded
    // frequency (reporting ok == false) when there are too few samples to regress.
    ClockFitQuality fit(std::span<const ClockSyncSample> samples, std::chrono::steady_clock::time_point host_start);

    // Whether a handshake with this round-trip time is worth re-anchoring on. A slow round trip places the anchor worse
    // than the drift it would correct, so a recent anchor is better kept; once that anchor has itself gone stale, a
    // loose anchor beats unbounded drift and anything is accepted.
    [[nodiscard]] bool accept_reanchor(std::chrono::nanoseconds rtt, std::chrono::steady_clock::time_point now) const;

    // Re-anchors the mapping on a handshake. The host cannot see where inside the round trip the device captured its
    // clock, so the anchor is placed at the midpoint: minimax placement, error <= rtt/2 with no assumption of symmetric
    // latency.
    void reanchor(std::chrono::steady_clock::time_point now, const ClockSyncSample& sample);

    // The mapping as of `now`, in the form published to profiler consumers.
    [[nodiscard]] experimental::ProgramRealtimeClockSync mapping(std::chrono::steady_clock::time_point now) const;

    [[nodiscard]] double frequency() const { return frequency_; }

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
