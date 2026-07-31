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

// Rate the device clock walks away from the frequency fitted at bring-up, used to judge how far the standing anchor
// has degraded since it was placed. Dominated by the chip warming after a fit taken cold, so it is architecture- and
// load-dependent: measured under trace replay at <=0.2ppm on Wormhole and 0.1-14.5ppm on Blackhole, always in the
// same direction (the clock slows). Sized above the worst of those, because underestimating is the unsafe direction
// -- it holds a stale anchor while the real error grows faster than modelled, where overestimating only re-anchors
// sooner. RealtimeProfilerStress.ClockDriftStaysWithinModelBudget measures the rate and asserts this still bounds it.
constexpr double kClockDriftPpm = 25.0;

// Half the round trip: the anchor could have landed anywhere inside it.
constexpr std::chrono::nanoseconds placement_error(std::chrono::nanoseconds rtt) { return rtt / 2; }

// How far above the median round trip a bring-up sample may sit and still be regressed. A handshake's round trip
// bounds how far its device timestamp could be from the host time it is paired with, so a slow one is a badly placed
// point rather than merely a late one. Chosen loose enough that ordinary spread survives and only the handshakes that
// were serviced late are cut.
constexpr int kFitRttOutlierFactor = 2;

}  // namespace

void ClockModel::seed_frequency(double frequency) {
    TT_FATAL(frequency > 0.0, "Real-time profiler clock model needs a positive seed frequency, got {}", frequency);
    frequency_ = frequency;
}

void ClockModel::set_anchor(std::chrono::steady_clock::time_point host_time, uint64_t device_ticks) {
    const double host_ns = static_cast<double>(host_time.time_since_epoch().count());
    device_cycle_offset_ = std::llround(static_cast<double>(device_ticks) - frequency_ * host_ns);
}

std::optional<ClockModel::FitResidual> ClockModel::fit(
    std::span<const ClockSyncSample> samples, std::chrono::steady_clock::time_point host_start) {
    if (samples.size() < 2) {
        // Too few to regress a slope, so the seeded frequency stands. One sample is still exactly the pair an anchor
        // needs, though, and anchoring on it beats leaving the offset unset. With none there is nothing to place: the
        // device's WALL_CLOCK free-runs from power-on, so no configured value says what it currently reads.
        if (samples.size() == 1) {
            try_reanchor(samples.front());
        }
        return std::nullopt;
    }

    // Round trips first, because they decide which samples are worth regressing. Every sample is paired with the host
    // time taken before its handshake, so a sample serviced late sits that much further right than the pair implies.
    // Regressed with equal weight, a handful of those tilts the slope by tens of ppm -- which the servo then spends
    // the whole session correcting as though the chip were drifting.
    std::vector<std::chrono::nanoseconds> rtts;
    rtts.reserve(samples.size());
    for (const auto& s : samples) {
        rtts.push_back(s.rtt);
    }
    const auto median = rtts.begin() + static_cast<ptrdiff_t>(rtts.size() / 2);
    std::nth_element(rtts.begin(), median, rtts.end());
    const std::chrono::nanoseconds median_rtt = *median;

    std::vector<ClockSyncSample> tight;
    tight.reserve(samples.size());
    for (const auto& s : samples) {
        if (s.rtt <= median_rtt * kFitRttOutlierFactor) {
            tight.push_back(s);
        }
    }
    // If the round trips were scattered enough that filtering leaves too few to regress, the scatter is what the link
    // is doing; fit everything and let the residual report it rather than inventing a slope from two points.
    const std::span<const ClockSyncSample> fitted_samples =
        tight.size() >= 2 ? std::span<const ClockSyncSample>(tight) : samples;

    // Host times are mean-centered on host_start: regressing at absolute-timestamp magnitudes loses most of the
    // significant digits of the sums to catastrophic cancellation. Differencing two time_points yields a signed
    // duration, which is what the centering needs.
    const double n = static_cast<double>(fitted_samples.size());
    const auto centered_host = [host_start](const ClockSyncSample& s) {
        return static_cast<double>((s.host_time - host_start).count());
    };

    double host_mean = 0.0;
    double device_mean = 0.0;
    for (const auto& s : fitted_samples) {
        host_mean += centered_host(s);
        device_mean += static_cast<double>(s.device_ticks);
    }
    host_mean /= n;
    device_mean /= n;

    double num = 0.0;
    double den = 0.0;
    for (const auto& s : fitted_samples) {
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

    // Reported over the samples actually regressed, so it says how well the line describes what it was fit to rather
    // than how far the rejected handshakes were.
    FitResidual residual;
    double residual_sumsq_ns = 0.0;
    for (const auto& s : fitted_samples) {
        const double predicted = device_mean + frequency_ * (centered_host(s) - host_mean);
        const double residual_ns = (static_cast<double>(s.device_ticks) - predicted) / frequency_;
        residual_sumsq_ns += residual_ns * residual_ns;
        residual.max_ns = std::max(residual.max_ns, std::abs(residual_ns));
    }
    residual.rms_ns = std::sqrt(residual_sumsq_ns / n);
    residual.num_samples_fitted = fitted_samples.size();
    residual.num_samples_offered = samples.size();

    // Without an anchor time and a round trip, mapping() would report zero error until the first resync. The median
    // over every offered sample, not just the regressed ones, keeps this the conservative of the two.
    rtt_ = median_rtt;
    last_reanchor_at_ = samples.back().host_time;
    return residual;
}

bool ClockModel::try_reanchor(const ClockSyncSample& sample) {
    if (last_reanchor_at_.has_value()) {
        // Only worth taking if it lands the anchor better than the standing one now sits. The standing anchor's error
        // is where it was placed plus what the clock has drifted since, so a slow round trip -- the device servicing
        // the handshake late while it is busy pushing records, say -- loses to a slightly older anchor that is still
        // better placed. The rule retires itself: the longer nothing is accepted, the larger the standing error, until
        // anything beats it.
        const double elapsed_ns = static_cast<double>((sample.host_time - *last_reanchor_at_).count());
        const auto drifted = std::chrono::nanoseconds(static_cast<int64_t>(elapsed_ns * kClockDriftPpm * 1e-6));
        if (placement_error(sample.rtt) > placement_error(rtt_) + drifted) {
            return false;
        }
    }
    rtt_ = sample.rtt;
    set_anchor(sample.host_time + placement_error(sample.rtt), sample.device_ticks);
    last_reanchor_at_ = sample.host_time;
    return true;
}

experimental::ProgramRealtimeClockSync ClockModel::mapping() const {
    return experimental::ProgramRealtimeClockSync{
        .device_cycle_offset = device_cycle_offset_,
        // Where the anchor could have landed inside its round trip. Drift between re-anchors measures ~6 ppm, which
        // over one resync interval stays far inside this bound, so it carries no term of its own.
        .sync_error = placement_error(rtt_),
    };
}

}  // namespace tt::tt_metal
