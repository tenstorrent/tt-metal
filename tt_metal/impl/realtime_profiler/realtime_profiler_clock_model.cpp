// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_clock_model.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

// Past this half-RTT a re-anchor's placement error exceeds the drift it would correct.
constexpr auto kMaxAcceptablePlacementError = std::chrono::microseconds(20);

// How long the current anchor stays worth preferring over a loose new one.
constexpr auto kMaxAnchorAge = std::chrono::seconds(2);

// Half the round trip: the anchor could have landed anywhere inside it.
constexpr std::chrono::nanoseconds placement_error(std::chrono::nanoseconds rtt) { return rtt / 2; }

// How close to the floor a round trip has to land to count as near it. Measured on Blackhole: with the floor at
// ~1.2us, ~80% of single probes already land inside this, which is what makes stopping a burst early worthwhile.
constexpr double kRttFloorSlack = 1.05;

// How much the floor climbs per observation, so it can recover when the path's real best round trip gets worse.
// Multiplicative because its use is: is_near compares against floor * kRttFloorSlack, so recovery time depends on the
// ratio alone and not on the absolute round trip, which varies ~4x across chips and architectures. A fraction of the
// floor, never of a sample's excess over it -- rising in proportion to that excess lets the bulk of the distribution
// drag the floor up to its own mean, which is the one thing the floor must not become. 1/4096 is ~200 observations to
// climb one kRttFloorSlack; erring low only costs probes, since the burst then simply runs its full depth.
constexpr double kRttFloorRise = 1.0 / 4096.0;

}  // namespace

void RttFloor::observe(std::chrono::nanoseconds rtt) {
    const double rtt_ns = static_cast<double>(rtt.count());
    if (floor_ns_ == 0.0 || rtt_ns < floor_ns_) {
        floor_ns_ = rtt_ns;
        return;
    }
    floor_ns_ += floor_ns_ * kRttFloorRise;
}

std::chrono::nanoseconds RttFloor::value() const { return std::chrono::nanoseconds(static_cast<int64_t>(floor_ns_)); }

bool RttFloor::is_near(std::chrono::nanoseconds rtt) const {
    return floor_ns_ != 0.0 && static_cast<double>(rtt.count()) <= floor_ns_ * kRttFloorSlack;
}

void ClockModel::seed_frequency(double frequency) {
    TT_FATAL(frequency > 0.0, "Real-time profiler clock model needs a positive seed frequency, got {}", frequency);
    frequency_ = frequency;
}

void ClockModel::set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks) {
    const double host_ns = static_cast<double>(host_time.time_since_epoch().count());
    device_cycle_offset_ = std::llround(static_cast<double>(device_ticks) - frequency_ * host_ns);
}

ClockFitQuality ClockModel::fit(
    std::span<const ClockSyncSample> samples, std::chrono::steady_clock::time_point host_start) {
    ClockFitQuality quality;
    quality.num_samples = static_cast<uint32_t>(samples.size());
    if (samples.size() < 2) {
        // Keep the seeded frequency and anchor it at the start of the attempt.
        set_anchor(host_start, 0);
        return quality;
    }

    // Host times are mean-centered on host_start: regressing at absolute-timestamp magnitudes loses most of the
    // significant digits of the sums to catastrophic cancellation. Differencing two time_points yields a signed
    // duration, which is what the centering needs.
    const double n = static_cast<double>(samples.size());
    const auto centered_host = [host_start](const ClockSyncSample& s) {
        return static_cast<double>((s.host_time - host_start).count());
    };

    double host_mean = 0.0;
    double device_mean = 0.0;
    for (const auto& s : samples) {
        host_mean += centered_host(s);
        device_mean += static_cast<double>(s.device_ticks);
    }
    host_mean /= n;
    device_mean /= n;

    double num = 0.0;
    double den = 0.0;
    for (const auto& s : samples) {
        const double dx = centered_host(s) - host_mean;
        const double dy = static_cast<double>(s.device_ticks) - device_mean;
        num += dx * dy;
        den += dx * dx;
    }
    if (std::abs(den) > 1e-10) {
        const double fitted = num / den;
        // Consumers divide by this, so an unusable slope has to not reach them; the seeded AICLK is still a working
        // mapping, which is the whole reason configure() seeds one before any handshake happens.
        if (fitted > 0.0) {
            frequency_ = fitted;
        } else {
            log_warning(
                tt::LogMetal,
                "[Real-time profiler] Clock fit produced a non-positive frequency ({}); keeping the commanded AICLK",
                fitted);
        }
    }

    // Intercept via means: the device tick count at centered host time 0, i.e. at host_start.
    set_anchor(host_start, static_cast<uint64_t>(device_mean - frequency_ * host_mean));

    double residual_sumsq_ns = 0.0;
    for (const auto& s : samples) {
        const double predicted = device_mean + frequency_ * (centered_host(s) - host_mean);
        const double residual_ns = (static_cast<double>(s.device_ticks) - predicted) / frequency_;
        residual_sumsq_ns += residual_ns * residual_ns;
        quality.residual_max_ns = std::max(quality.residual_max_ns, std::abs(residual_ns));
    }
    quality.residual_rms_ns = std::sqrt(residual_sumsq_ns / n);
    quality.ok = true;

    // Without an anchor time and a round trip, mapping() would report zero error until the first resync. The line is
    // fit from every sample, so the median of their round trips represents its uncertainty.
    std::vector<std::chrono::nanoseconds> rtts;
    rtts.reserve(samples.size());
    for (const auto& s : samples) {
        rtts.push_back(s.rtt);
    }
    const auto median = rtts.begin() + static_cast<ptrdiff_t>(rtts.size() / 2);
    std::nth_element(rtts.begin(), median, rtts.end());
    rtt_ = *median;
    last_reanchor_at_ = samples.back().host_time;
    return quality;
}

bool ClockModel::accept_reanchor(std::chrono::nanoseconds rtt, std::chrono::steady_clock::time_point now) const {
    if (!last_reanchor_at_.has_value()) {
        return true;
    }
    if (now - *last_reanchor_at_ >= kMaxAnchorAge) {
        return true;
    }
    return placement_error(rtt) <= kMaxAcceptablePlacementError;
}

void ClockModel::reanchor(std::chrono::steady_clock::time_point now, const ClockSyncSample& sample) {
    rtt_ = sample.rtt;
    set_anchor(sample.host_time + placement_error(sample.rtt), sample.device_ticks);
    last_reanchor_at_ = now;
}

experimental::ProgramRealtimeClockSync ClockModel::mapping(std::chrono::steady_clock::time_point) const {
    return experimental::ProgramRealtimeClockSync{
        .device_cycle_offset = device_cycle_offset_,
        // Where the anchor could have landed inside its round trip. Measured drift between re-anchors is ~6 ppm, so
        // over the servo's 50 ms interval it stays far inside this bound.
        .sync_error_ns = static_cast<uint64_t>(placement_error(rtt_).count()),
    };
}

}  // namespace tt::tt_metal
