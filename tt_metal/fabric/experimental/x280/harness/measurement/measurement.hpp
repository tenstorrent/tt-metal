// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_HARNESS_MEASUREMENT_HPP
#define X280_HARNESS_MEASUREMENT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::harness::measurement {

constexpr uint64_t kX280MtimeHz = 50000000ull;
constexpr uint64_t kAiclkSampleGapNs = 50000000ull;
constexpr uint64_t kAiclkRateTolerancePpm = 10000ull;
constexpr uint64_t kAiclkMaximumEndpointBoundWidthPpm = 2500ull;
constexpr uint64_t kAiclkMaximumElapsedBoundWidthPpm = 5000ull;
// Device work (warm-up plus timed) a calibrated run spans. ~40 us of host read
// brackets, 60 us worst observed, stay under 2500 ppm of this.
constexpr uint64_t kAiclkTargetDeviceWorkNs = 25000000ull;

/**
 * Converts a millisecond phase budget into whole ticks of an `hz` timer,
 * rounding up.
 *
 * The single definition of that conversion: admission proves it holds and the
 * launch path reuses it for the value it transmits.
 *
 * @return true and writes `*out`; false leaving `*out` untouched when `out` is
 *         null, either input is zero, the product overflows 64 bits, or the
 *         result is zero or outside the timer half-range. The last two hold by
 *         construction rather than by branch -- rounding a non-zero product up
 *         cannot yield zero, and dividing a 64-bit product by 1000 cannot reach
 *         2^63 -- so they state the postcondition callers rely on and no input
 *         reaches them.
 */
inline bool phase_timeout_ticks(uint64_t ms, uint64_t hz, uint64_t* out) {
    uint64_t ticks = 0;
    if (out == nullptr || ms == 0u || hz == 0u) {
        return false;
    }
    if (__builtin_mul_overflow(ms, hz, &ticks) || __builtin_add_overflow(ticks, 999u, &ticks)) {
        return false;
    }
    ticks /= 1000u;
    if (ticks == 0u || ticks >= (1ULL << 63)) {
        return false;
    }
    *out = ticks;
    return true;
}

/**
 * The shortest host wait, in ms, for a run whose device side can spend
 * `phases` budgets of `budget_ms` in sequence before every kernel has exited,
 * plus `slack_ms` for what the budgets do not bound (launch, a healthy run's
 * traffic): phases x budget_ms + slack_ms.
 *
 * @return true and writes `*out`; false leaving `*out` untouched when that
 *         overflows 64 bits.
 */
inline bool host_wait_floor_ms(uint64_t phases, uint64_t budget_ms, uint64_t slack_ms, uint64_t* out) {
    uint64_t v = 0;
    if (__builtin_mul_overflow(phases, budget_ms, &v) || __builtin_add_overflow(v, slack_ms, &v)) {
        return false;
    }
    *out = v;
    return true;
}

// One device-counter read bracketed by process-local steady-clock nanoseconds.
struct HostClockSample {
    uint64_t host_before_ns = 0;
    uint64_t counter_ticks = 0;
    uint64_t host_after_ns = 0;
};

// One timestamped CMFW snapshot. Frequencies and the update period are MHz and
// milliseconds; arb_max_raw preserves the firmware's packed index/MHz word.
struct CmfwClockSample {
    uint64_t host_before_ns = 0;
    uint64_t host_after_ns = 0;
    uint32_t limit_max_mhz = 0;
    uint32_t current_mhz = 0;
    uint32_t arb_max_raw = 0;
    uint32_t thermal_trip_count = 0;
    uint32_t update_period_ms = 0;
};

// Inclusive lower and upper counter-rate bounds in Hz.
struct RateBounds {
    long double lower_hz = 0;
    long double upper_hz = 0;
};

// A counter whose raw ticks or cycles a record carries, one entry per physical
// counter. TensixLiveTimer is the Tensix debug timestamper and EriscWallClock
// the ERISC wall clock eth_read_wall_clock() reads; an AICLK endpoint's samples
// come from one of these two. RiscDebugWallClock is what a DRISC's
// get_timestamp() reads: the Tensix debug-register address on a DRAM tile,
// which no specification documents as the same counter. The others are X280
// mtime and rdcycle and a Metal RISC's cycle CSR.
enum class Counter : uint32_t {
    TensixLiveTimer,
    EriscWallClock,
    X280Mtime,
    X280Rdcycle,
    Rv32Cycle,
    RiscDebugWallClock,
};

// Returns the stable lower-snake-case JSON spelling; unknown values throw.
inline const char* counter_name(Counter counter) {
    switch (counter) {
        case Counter::TensixLiveTimer: return "tensix_live_timer";
        case Counter::EriscWallClock: return "erisc_wall_clock";
        case Counter::X280Mtime: return "x280_mtime";
        case Counter::X280Rdcycle: return "x280_rdcycle";
        case Counter::Rv32Cycle: return "rv32_cycle";
        case Counter::RiscDebugWallClock: return "risc_debug_wall_clock";
    }
    throw std::invalid_argument("unknown Counter value " + std::to_string(static_cast<uint32_t>(counter)));
}

// Stable machine-readable reasons that an AICLK conversion rate was withheld.
enum class ClockError : uint32_t {
    ReferenceReadFailed,
    TelemetryUnavailable,
    InvalidTelemetryPeriod,
    LimitChanged,
    CurrentFrequencyMismatch,
    ArbiterClampOutsideTolerance,
    ThermalTripCountChanged,
    NonmonotonicHostTime,
    NonmonotonicCounter,
    ExcessiveUncertainty,
    StartRateOutsideExpected,
    EndRateOutsideExpected,
    EndpointRatesDisagree,
    ElapsedRateOutsideExpected,
};

// Two local-rate samples and one CMFW snapshot taken on one side of a workload.
// The sampler sets `reference_core`, the translated (x, y) of the core it read,
// and `reference`, that core's live timer; both are unset only in an endpoint
// no sampler filled.
struct AiclkEndpoint {
    std::optional<Counter> reference;
    std::optional<std::array<uint32_t, 2>> reference_core;
    HostClockSample first;
    HostClockSample second;
    CmfwClockSample telemetry;
    std::vector<ClockError> errors;
};

// Complete per-run evidence. conversion_hz is zero exactly when errors is nonempty.
// `reference` and `reference_core` are the start endpoint's; the end endpoint
// samples the same core.
struct AiclkRunEvidence {
    std::optional<Counter> reference;
    std::optional<std::array<uint32_t, 2>> reference_core;
    uint64_t expected_hz = 0;
    uint64_t conversion_hz = 0;
    AiclkEndpoint start;
    AiclkEndpoint end;
    RateBounds start_rate;
    RateBounds end_rate;
    RateBounds elapsed_rate;
    // Steady-clock ns from the start bracket's end to the end bracket's start;
    // empty when those brackets are misordered.
    std::optional<uint64_t> elapsed_window_ns;
    std::vector<ClockError> errors;
};

// Bounds one constant-rate interval from two nonoverlapping host-bracketed
// samples. Timestamps are steady-clock nanoseconds; malformed ordering or a
// non-increasing counter throws std::invalid_argument.
inline RateBounds rate_bounds(const HostClockSample& first, const HostClockSample& second) {
    if (first.host_before_ns > first.host_after_ns || second.host_before_ns > second.host_after_ns ||
        second.host_before_ns <= first.host_after_ns) {
        throw std::invalid_argument("clock samples have overlapping or reversed host brackets");
    }
    if (second.counter_ticks <= first.counter_ticks) {
        throw std::invalid_argument("clock counter did not increase between samples");
    }
    const long double ticks = second.counter_ticks - first.counter_ticks;
    const auto widest_ns = second.host_after_ns - first.host_before_ns;
    const auto narrowest_ns = second.host_before_ns - first.host_after_ns;
    const RateBounds bounds{
        ticks * 1.0e9L / widest_ns,
        ticks * 1.0e9L / narrowest_ns,
    };
    if (!(bounds.lower_hz > 0 && bounds.lower_hz <= bounds.upper_hz) || !std::isfinite(bounds.lower_hz) ||
        !std::isfinite(bounds.upper_hz)) {
        throw std::invalid_argument("clock rate bounds are not finite and ordered");
    }
    return bounds;
}

// Returns the stable lower-snake-case JSON spelling; unknown enum values throw.
inline const char* clock_error_name(ClockError error) {
    switch (error) {
        case ClockError::ReferenceReadFailed: return "reference_read_failed";
        case ClockError::TelemetryUnavailable: return "telemetry_unavailable";
        case ClockError::InvalidTelemetryPeriod: return "invalid_telemetry_period";
        case ClockError::LimitChanged: return "limit_changed";
        case ClockError::CurrentFrequencyMismatch: return "current_frequency_mismatch";
        case ClockError::ArbiterClampOutsideTolerance: return "arbiter_clamp_outside_tolerance";
        case ClockError::ThermalTripCountChanged: return "thermal_trip_count_changed";
        case ClockError::NonmonotonicHostTime: return "nonmonotonic_host_time";
        case ClockError::NonmonotonicCounter: return "nonmonotonic_counter";
        case ClockError::ExcessiveUncertainty: return "excessive_uncertainty";
        case ClockError::StartRateOutsideExpected: return "start_rate_outside_expected";
        case ClockError::EndRateOutsideExpected: return "end_rate_outside_expected";
        case ClockError::EndpointRatesDisagree: return "endpoint_rates_disagree";
        case ClockError::ElapsedRateOutsideExpected: return "elapsed_rate_outside_expected";
    }
    throw std::invalid_argument("unknown ClockError value");
}

// Reduces endpoint observations into a conversion decision. Hardware failures
// are returned in errors for serialization; an invalid expected rate, or
// endpoints naming different reference cores, throws std::invalid_argument.
inline AiclkRunEvidence assess_aiclk_run(uint64_t expected_hz, const AiclkEndpoint& start, const AiclkEndpoint& end) {
    if (expected_hz == 0 || expected_hz % 1000000ull != 0 || expected_hz / 1000000ull > UINT32_MAX) {
        throw std::invalid_argument("expected AICLK must be a nonzero whole-MHz rate");
    }
    if (start.reference_core != end.reference_core) {
        const auto core = [](const std::optional<std::array<uint32_t, 2>>& c) {
            return c ? "(" + std::to_string((*c)[0]) + ", " + std::to_string((*c)[1]) + ")" : std::string("null");
        };
        throw std::invalid_argument(
            "AICLK start endpoint sampled core " + core(start.reference_core) + " but the end endpoint core " +
            core(end.reference_core));
    }
    AiclkRunEvidence result{
        .reference = start.reference,
        .reference_core = start.reference_core,
        .expected_hz = expected_hz,
        .start = start,
        .end = end};
    result.errors.insert(result.errors.end(), start.errors.begin(), start.errors.end());
    result.errors.insert(result.errors.end(), end.errors.begin(), end.errors.end());

    const auto has_error = [](const AiclkEndpoint& endpoint, ClockError error) {
        return std::find(endpoint.errors.begin(), endpoint.errors.end(), error) != endpoint.errors.end();
    };
    const bool telemetry_available =
        !has_error(start, ClockError::TelemetryUnavailable) && !has_error(end, ClockError::TelemetryUnavailable);
    const uint32_t expected_mhz = static_cast<uint32_t>(expected_hz / 1000000ull);
    const auto telemetry_within_tolerance = [expected_mhz](uint32_t observed_mhz) {
        const uint64_t difference =
            observed_mhz > expected_mhz ? observed_mhz - expected_mhz : expected_mhz - observed_mhz;
        return difference * 1000000ull <= expected_mhz * kAiclkRateTolerancePpm;
    };
    if (telemetry_available) {
        if (start.telemetry.update_period_ms == 0 || start.telemetry.update_period_ms > 1000 ||
            end.telemetry.update_period_ms == 0 || end.telemetry.update_period_ms > 1000 ||
            start.telemetry.update_period_ms != end.telemetry.update_period_ms) {
            result.errors.push_back(ClockError::InvalidTelemetryPeriod);
        }
        if (start.telemetry.limit_max_mhz != expected_mhz || end.telemetry.limit_max_mhz != expected_mhz) {
            result.errors.push_back(ClockError::LimitChanged);
        }
        if (!telemetry_within_tolerance(start.telemetry.current_mhz) ||
            !telemetry_within_tolerance(end.telemetry.current_mhz)) {
            result.errors.push_back(ClockError::CurrentFrequencyMismatch);
        }
        if (!telemetry_within_tolerance(start.telemetry.arb_max_raw & 0xffffu) ||
            !telemetry_within_tolerance(end.telemetry.arb_max_raw & 0xffffu)) {
            result.errors.push_back(ClockError::ArbiterClampOutsideTolerance);
        }
        if (start.telemetry.thermal_trip_count != end.telemetry.thermal_trip_count) {
            result.errors.push_back(ClockError::ThermalTripCountChanged);
        }
    }

    auto calculate = [&](const HostClockSample& first, const HostClockSample& second, RateBounds& bounds) {
        if (first.host_before_ns > first.host_after_ns || second.host_before_ns > second.host_after_ns ||
            second.host_before_ns <= first.host_after_ns) {
            result.errors.push_back(ClockError::NonmonotonicHostTime);
            return false;
        }
        if (second.counter_ticks <= first.counter_ticks) {
            result.errors.push_back(ClockError::NonmonotonicCounter);
            return false;
        }
        bounds = rate_bounds(first, second);
        return true;
    };

    const bool start_ok =
        !has_error(start, ClockError::ReferenceReadFailed) && calculate(start.first, start.second, result.start_rate);
    const bool end_ok =
        !has_error(end, ClockError::ReferenceReadFailed) && calculate(end.first, end.second, result.end_rate);
    const bool elapsed_ok = start_ok && end_ok && calculate(start.second, end.first, result.elapsed_rate);
    if (elapsed_ok) {
        result.elapsed_window_ns = end.first.host_before_ns - start.second.host_after_ns;
    }
    const long double accepted_delta = expected_hz * kAiclkRateTolerancePpm / 1.0e6L;
    const long double accepted_lower = expected_hz - accepted_delta;
    const long double accepted_upper = expected_hz + accepted_delta;
    const long double maximum_endpoint_width = expected_hz * kAiclkMaximumEndpointBoundWidthPpm / 1.0e6L;
    const long double maximum_elapsed_width = expected_hz * kAiclkMaximumElapsedBoundWidthPpm / 1.0e6L;
    auto inside = [&](const RateBounds& bounds) {
        return bounds.lower_hz >= accepted_lower && bounds.upper_hz <= accepted_upper;
    };
    if ((start_ok && result.start_rate.upper_hz - result.start_rate.lower_hz > maximum_endpoint_width) ||
        (end_ok && result.end_rate.upper_hz - result.end_rate.lower_hz > maximum_endpoint_width) ||
        (elapsed_ok && result.elapsed_rate.upper_hz - result.elapsed_rate.lower_hz > maximum_elapsed_width)) {
        result.errors.push_back(ClockError::ExcessiveUncertainty);
    }
    if (start_ok && !inside(result.start_rate)) {
        result.errors.push_back(ClockError::StartRateOutsideExpected);
    }
    if (end_ok && !inside(result.end_rate)) {
        result.errors.push_back(ClockError::EndRateOutsideExpected);
    }
    if (start_ok && end_ok) {
        const long double endpoint_gap = std::max(result.start_rate.lower_hz, result.end_rate.lower_hz) -
                                         std::min(result.start_rate.upper_hz, result.end_rate.upper_hz);
        if (endpoint_gap > accepted_delta) {
            result.errors.push_back(ClockError::EndpointRatesDisagree);
        }
    }
    if (elapsed_ok && !inside(result.elapsed_rate)) {
        result.errors.push_back(ClockError::ElapsedRateOutsideExpected);
    }
    if (result.errors.empty()) {
        result.conversion_hz = expected_hz;
    }
    return result;
}

// Why calibrate_warmup() withheld a warm-up count.
enum class WarmupRefusal : uint32_t {
    InvalidPilot,
    ExceedsWarmupLimit,
    ExceedsDeviceTimeLimit,
};

// A pilot observation and the suite's limits for one warm-up calibration.
// Device time is nanoseconds converted from the pilot's own device counters. A
// warm-up iteration is taken to cost what a pilot timed iteration cost.
struct WarmupCalibrationRequest {
    uint64_t pilot_device_ns = 0;  // device time of pilot_iterations iterations
    uint64_t pilot_iterations = 0;
    uint64_t default_warmup = 0;    // nonzero; the result is never below it
    uint64_t timed_iterations = 0;  // N; its device time counts toward the target
    uint64_t target_device_ns = 0;  // warm-up plus timed device time to reach
    uint64_t max_warmup = 0;        // largest W the suite's own bounds admit
    uint64_t max_device_ns = 0;     // largest (W+N) device time its host waits admit;
                                    // UINT64_MAX when no host wait bounds the run
};

// How a run's unrecorded warm-up count was chosen. calibration holds the
// request behind a calibrated count and is empty for an explicit count, used
// as given, or a refusal; warmup is zero exactly when refusal is set.
struct WarmupSelection {
    uint64_t warmup = 0;
    std::optional<WarmupCalibrationRequest> calibration;
    std::optional<WarmupRefusal> refusal;
};

// Returns a selection carrying `request` whose warmup is the smallest
// W >= default_warmup for which (W+N) iterations reach target_device_ns at the
// pilot's per-iteration cost, or one whose refusal names the violated bound.
// A pilot without device time is a hardware outcome and is refused; a zero
// default_warmup is a caller error and throws std::invalid_argument.
inline WarmupSelection calibrate_warmup(const WarmupCalibrationRequest& request) {
    if (request.default_warmup == 0) {
        throw std::invalid_argument("warm-up calibration requires a nonzero default");
    }
    if (request.pilot_device_ns == 0 || request.pilot_iterations == 0) {
        return {.refusal = WarmupRefusal::InvalidPilot};
    }
    using wide = unsigned __int128;
    // The refusal above keeps both divisors nonzero; reaching zero here is a
    // broken guard, not a pilot outcome.
    const auto ceil_div = [](wide numerator, wide divisor) {
        if (divisor == 0) {
            throw std::logic_error("warm-up calibration divided by a zero pilot figure");
        }
        return (numerator + divisor - 1) / divisor;
    };
    const wide total_iterations =
        ceil_div(wide(request.target_device_ns) * request.pilot_iterations, request.pilot_device_ns);
    const wide needed = total_iterations > request.timed_iterations ? total_iterations - request.timed_iterations : 0;
    const wide warmup = std::max<wide>(needed, request.default_warmup);
    if (warmup > request.max_warmup) {
        return {.refusal = WarmupRefusal::ExceedsWarmupLimit};
    }
    const wide device_ns =
        ceil_div((warmup + request.timed_iterations) * request.pilot_device_ns, request.pilot_iterations);
    if (device_ns > request.max_device_ns) {
        return {.refusal = WarmupRefusal::ExceedsDeviceTimeLimit};
    }
    return {.warmup = static_cast<uint64_t>(warmup), .calibration = request};
}

// Builds the calibration request for a batch member whose warm-up is a u32
// count and whose host wait is unbounded, from a pilot batch of `items`
// iterations spanning `wall_ticks` at `wall_hz`. A zero wall_hz throws
// std::invalid_argument; zero ticks or items reach calibrate_warmup() as an
// invalid pilot.
inline WarmupCalibrationRequest batch_warmup_calibration_request(
    uint64_t wall_ticks, uint64_t items, uint64_t wall_hz, uint64_t default_warmup, uint64_t timed_iterations) {
    if (wall_hz == 0) {
        throw std::invalid_argument("batch warm-up calibration requires a wall rate");
    }
    return {
        .pilot_device_ns = static_cast<uint64_t>(std::llround(wall_ticks * 1.0e9L / wall_hz)),
        .pilot_iterations = items,
        .default_warmup = default_warmup,
        .timed_iterations = timed_iterations,
        .target_device_ns = kAiclkTargetDeviceWorkNs,
        .max_warmup = UINT32_MAX,
        .max_device_ns = UINT64_MAX,
    };
}

// Returns the stable lower-snake-case spelling; unknown enum values throw.
inline const char* warmup_refusal_name(WarmupRefusal refusal) {
    switch (refusal) {
        case WarmupRefusal::InvalidPilot: return "invalid_pilot";
        case WarmupRefusal::ExceedsWarmupLimit: return "exceeds_warmup_limit";
        case WarmupRefusal::ExceedsDeviceTimeLimit: return "exceeds_device_time_limit";
    }
    throw std::invalid_argument("unknown WarmupRefusal value");
}

}  // namespace x280::harness::measurement
#endif
