// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "rt/measurement.h"

#include <gtest/gtest.h>

namespace measurement = x280::harness::measurement;

namespace {

measurement::CmfwClockSample telemetry(uint32_t mhz = 1000, uint32_t trips = 5) {
    return {
        .host_before_ns = 1,
        .host_after_ns = 2,
        .limit_max_mhz = mhz,
        .current_mhz = mhz,
        .arb_max_raw = mhz,
        .thermal_trip_count = trips,
        .update_period_ms = 100,
    };
}

// An endpoint as the sampler leaves it on a Tensix core.
measurement::AiclkEndpoint endpoint(uint64_t host_base, uint64_t tick_base) {
    return {
        .reference = measurement::Counter::TensixLiveTimer,
        .reference_core = std::array<uint32_t, 2>{1, 2},
        .first = {host_base, tick_base, host_base + 10},
        .second = {host_base + 10000000, tick_base + 10000000, host_base + 10000010},
        .telemetry = telemetry(),
    };
}

}  // namespace

TEST(MeasurementInterval, RetainsZeroAndSingleWrapButRejectsHalfRange) {
    uint64_t delta = 99;
    EXPECT_TRUE(measurement_delta(12, 12, &delta));
    EXPECT_EQ(delta, 0u);
    EXPECT_TRUE(measurement_delta(UINT64_MAX - 4, 3, &delta));
    EXPECT_EQ(delta, 8u);
    EXPECT_FALSE(measurement_delta(0, 1ULL << 63, &delta));
    EXPECT_EQ(delta, 8u);
}

TEST(MeasurementAiclk, BoundsOneHostBracketedInterval) {
    const auto bounds = measurement::rate_bounds({100, 1000, 110}, {10000100, 10001000, 10000110});
    EXPECT_LT(bounds.lower_hz, 1000000000.L);
    EXPECT_GT(bounds.upper_hz, 1000000000.L);
    EXPECT_THROW(measurement::rate_bounds({100, 1000, 110}, {105, 2000, 115}), std::invalid_argument);
    EXPECT_THROW(measurement::rate_bounds({100, 1000, 110}, {10000100, 999, 10000110}), std::invalid_argument);
}

TEST(MeasurementAiclk, AcceptsStableExpectedRate) {
    const auto start = endpoint(100, 1000);
    const auto end = endpoint(20000100, 20001000);
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_TRUE(evidence.errors.empty());
    EXPECT_EQ(evidence.conversion_hz, 1000000000u);
    EXPECT_LT(evidence.elapsed_rate.lower_hz, 1000000000.L);
    EXPECT_GT(evidence.elapsed_rate.upper_hz, 1000000000.L);
    const auto json = x280::harness::output::clock_json(evidence);
    EXPECT_NE(json.find("\"conversion_hz\":1000000000"), std::string::npos);
    EXPECT_NE(json.find("\"reference_counter\":\"tensix_live_timer\",\"reference_core\":[1,2],"), std::string::npos);
    EXPECT_NE(json.find("\"errors\":[]"), std::string::npos);
}

// Spec (AiclkRunEvidence): both endpoints sample one core, which the evidence
// names; endpoints naming different cores are a caller error.
TEST(MeasurementAiclk, EndpointsOnDifferentCoresAreRefused) {
    auto end = endpoint(20000100, 20001000);
    end.reference_core = std::array<uint32_t, 2>{1, 3};
    try {
        measurement::assess_aiclk_run(1000000000, endpoint(100, 1000), end);
        ADD_FAILURE() << "endpoints on different cores were assessed";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("(1, 2) but the end endpoint core (1, 3)"), std::string::npos)
            << error.what();
    }
    end.reference_core.reset();
    EXPECT_THROW(measurement::assess_aiclk_run(1000000000, endpoint(100, 1000), end), std::invalid_argument);
}

namespace {

size_t count_occurrences(const std::string& haystack, const std::string& needle) {
    size_t count = 0;
    for (auto at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

}  // namespace

// Spec: accepted and rejected evidence alike carry the endpoints' reference and
// serialise it once as reference_counter, "tensix_live_timer" or
// "erisc_wall_clock", and the reference changes no other byte. Accepted
// evidence converts at either reference; rejected evidence's null
// conversion_hz and non-empty errors array exercise the other serialisation
// paths.
TEST(MeasurementAiclk, EvidenceCarriesAndSerialisesEachReference) {
    using measurement::Counter;
    for (const bool rejected : {false, true}) {
        SCOPED_TRACE(rejected ? "rejected evidence" : "accepted evidence");
        std::string tensix_json;
        for (const auto& [reference, spelling] :
             {std::pair{Counter::TensixLiveTimer, "tensix_live_timer"},
              std::pair{Counter::EriscWallClock, "erisc_wall_clock"}}) {
            auto start = endpoint(100, 1000);
            auto end = endpoint(20000100, 20001000);
            if (rejected) {
                end.telemetry.current_mhz = 900;
                end.telemetry.thermal_trip_count = 6;
            }
            start.reference = reference;
            end.reference = reference;
            const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
            EXPECT_EQ(evidence.reference, reference) << spelling;
            EXPECT_EQ(evidence.errors.empty(), !rejected) << spelling;
            EXPECT_EQ(evidence.conversion_hz, rejected ? 0u : 1000000000u) << spelling;

            const auto json = x280::harness::output::clock_json(evidence);
            const std::string field = std::string("\"reference_counter\":\"") + spelling + "\"";
            EXPECT_EQ(count_occurrences(json, "\"reference_counter\""), 1u) << json;
            ASSERT_EQ(count_occurrences(json, field), 1u) << json;
            if (reference == Counter::TensixLiveTimer) {
                tensix_json = json;
            } else {
                auto restored = json;
                restored.replace(restored.find(field), field.size(), "\"reference_counter\":\"tensix_live_timer\"");
                EXPECT_EQ(restored, tensix_json);
            }
        }
    }
}

TEST(MeasurementAiclk, RejectsTelemetryAndRateChanges) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    end.telemetry.current_mhz = 900;
    end.telemetry.thermal_trip_count = 6;
    end.second.counter_ticks -= 1000000;
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_EQ(evidence.conversion_hz, 0u);
    EXPECT_NE(
        std::find(evidence.errors.begin(), evidence.errors.end(), measurement::ClockError::CurrentFrequencyMismatch),
        evidence.errors.end());
    EXPECT_NE(
        std::find(evidence.errors.begin(), evidence.errors.end(), measurement::ClockError::ThermalTripCountChanged),
        evidence.errors.end());
    EXPECT_NE(
        std::find(evidence.errors.begin(), evidence.errors.end(), measurement::ClockError::EndRateOutsideExpected),
        evidence.errors.end());
}

TEST(MeasurementAiclk, RejectsBadExpectationAndAcquisitionEvidence) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    start.errors.push_back(measurement::ClockError::TelemetryUnavailable);
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_EQ(evidence.conversion_hz, 0u);
    EXPECT_EQ(evidence.errors.front(), measurement::ClockError::TelemetryUnavailable);
    EXPECT_THROW(measurement::assess_aiclk_run(0, start, end), std::invalid_argument);
    EXPECT_THROW(measurement::assess_aiclk_run(1000000001, start, end), std::invalid_argument);
}

TEST(MeasurementAiclk, RejectsFirmwareLimitArbiterAndCadenceChanges) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    end.telemetry.limit_max_mhz = 900;
    end.telemetry.arb_max_raw = 900;
    end.telemetry.update_period_ms = 101;
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_EQ(evidence.conversion_hz, 0u);
    for (const auto error :
         {measurement::ClockError::LimitChanged,
          measurement::ClockError::ArbiterClampOutsideTolerance,
          measurement::ClockError::InvalidTelemetryPeriod}) {
        EXPECT_NE(std::find(evidence.errors.begin(), evidence.errors.end(), error), evidence.errors.end());
    }
}

TEST(MeasurementAiclk, RejectsEndpointRateDisagreement) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    end.second.counter_ticks += 200000;
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_EQ(evidence.conversion_hz, 0u);
    EXPECT_NE(
        std::find(evidence.errors.begin(), evidence.errors.end(), measurement::ClockError::EndpointRatesDisagree),
        evidence.errors.end());
}

TEST(MeasurementAiclk, AcceptsTelemetryWithinOnePercent) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    start.telemetry.current_mhz = 991;
    start.telemetry.arb_max_raw = 991;
    end.telemetry.current_mhz = 1009;
    end.telemetry.arb_max_raw = 1009;
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_TRUE(evidence.errors.empty());
    EXPECT_EQ(evidence.conversion_hz, 1000000000u);
}

TEST(MeasurementAiclk, AllowsShortWorkloadToUseTheElapsedUncertaintyLimit) {
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    start.second.host_after_ns += 20000;
    end.first.host_after_ns += 20000;
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_GT(
        evidence.elapsed_rate.upper_hz - evidence.elapsed_rate.lower_hz,
        1000000000.L * measurement::kAiclkMaximumEndpointBoundWidthPpm / 1.0e6L);
    EXPECT_TRUE(evidence.errors.empty());
    EXPECT_EQ(evidence.conversion_hz, 1000000000u);
}

namespace {

// 64 pilot iterations at 100 ns each; the target needs 250000 iterations.
measurement::WarmupCalibrationRequest warmup_request() {
    return {
        .pilot_device_ns = 6400,
        .pilot_iterations = 64,
        .default_warmup = 8,
        .timed_iterations = 64,
        .target_device_ns = 25000000,
        .max_warmup = UINT32_MAX,
        .max_device_ns = UINT64_MAX,
    };
}

}  // namespace

TEST(MeasurementWarmup, RaisesWarmupUntilWarmupAndTimedReachTarget) {
    const auto calibration = measurement::calibrate_warmup(warmup_request());
    EXPECT_FALSE(calibration.refusal.has_value());
    EXPECT_EQ(calibration.warmup, 250000u - 64u);
}

TEST(MeasurementWarmup, RoundsAPartialIterationUp) {
    auto request = warmup_request();
    request.pilot_device_ns = 3;
    request.pilot_iterations = 1;
    request.default_warmup = 1;
    request.timed_iterations = 0;
    request.target_device_ns = 10;
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 4u);
}

TEST(MeasurementWarmup, KeepsDefaultWhenAlreadyLongEnough) {
    auto request = warmup_request();
    request.timed_iterations = 300000;
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 8u);
    request.timed_iterations = 250000 - 4;
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 8u);
}

TEST(MeasurementWarmup, RefusesPilotWithoutDeviceTime) {
    auto request = warmup_request();
    request.pilot_device_ns = 0;
    auto calibration = measurement::calibrate_warmup(request);
    EXPECT_EQ(calibration.refusal, measurement::WarmupRefusal::InvalidPilot);
    EXPECT_EQ(calibration.warmup, 0u);
    request = warmup_request();
    request.pilot_iterations = 0;
    EXPECT_EQ(measurement::calibrate_warmup(request).refusal, measurement::WarmupRefusal::InvalidPilot);
}

TEST(MeasurementWarmup, RefusesWarmupAboveSuiteLimit) {
    auto request = warmup_request();
    request.max_warmup = 250000 - 64;
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 250000u - 64u);
    request.max_warmup -= 1;
    const auto calibration = measurement::calibrate_warmup(request);
    EXPECT_EQ(calibration.refusal, measurement::WarmupRefusal::ExceedsWarmupLimit);
    EXPECT_EQ(calibration.warmup, 0u);
}

TEST(MeasurementWarmup, RefusesDeviceTimeAboveHostWait) {
    auto request = warmup_request();
    request.max_device_ns = 25000000;
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 250000u - 64u);
    request.max_device_ns -= 1;
    const auto calibration = measurement::calibrate_warmup(request);
    EXPECT_EQ(calibration.refusal, measurement::WarmupRefusal::ExceedsDeviceTimeLimit);
    EXPECT_EQ(calibration.warmup, 0u);
}

TEST(MeasurementWarmup, RejectsZeroDefault) {
    auto request = warmup_request();
    request.default_warmup = 0;
    EXPECT_THROW(measurement::calibrate_warmup(request), std::invalid_argument);
}

TEST(MeasurementWarmup, SerializesSourceAndElapsedWindow) {
    const auto request = warmup_request();
    const auto warmup = [](const measurement::WarmupSelection& selection) {
        return x280::harness::output::record({.suite = "s", .member = "m", .warmup = selection});
    };
    EXPECT_NE(warmup({.warmup = 8}).find("\"warmup_source\":\"explicit\",\"tt_metal"), std::string::npos);
    EXPECT_NE(
        warmup(measurement::calibrate_warmup(request))
            .find("\"warmup_source\":\"calibrated\",\"warmup_target_device_ns\":25000000"
                  ",\"warmup_pilot_device_ns\":6400,\"warmup_pilot_iterations\":64,"),
        std::string::npos);
    const auto evidence = measurement::assess_aiclk_run(1000000000, endpoint(100, 1000), endpoint(20000100, 20001000));
    EXPECT_NE(x280::harness::output::clock_json(evidence).find("\"elapsed_window_ns\":9999990,"), std::string::npos);
}

TEST(MeasurementWarmup, BatchRequestConvertsPilotTicksAtTheWallRate) {
    const auto request = measurement::batch_warmup_calibration_request(1350000, 1000, 1350000000, 128, 4096);
    EXPECT_EQ(request.pilot_device_ns, 1000000u);
    EXPECT_EQ(request.pilot_iterations, 1000u);
    EXPECT_EQ(request.max_warmup, UINT32_MAX);
    EXPECT_EQ(measurement::calibrate_warmup(request).warmup, 25000u - 4096u);
    EXPECT_THROW(measurement::batch_warmup_calibration_request(1, 1, 0, 128, 1), std::invalid_argument);
}

// Spec (Counter): each physical counter has one stable record spelling (owner
// decision 41); an out-of-range value throws std::invalid_argument.
TEST(MeasurementAiclk, CounterNameSpellsEachCounterAndRejectsOutOfRangeValue) {
    using measurement::Counter;
    for (const auto& [counter, spelling] :
         {std::pair{Counter::TensixLiveTimer, "tensix_live_timer"},
          std::pair{Counter::EriscWallClock, "erisc_wall_clock"},
          std::pair{Counter::X280Mtime, "x280_mtime"},
          std::pair{Counter::X280Rdcycle, "x280_rdcycle"},
          std::pair{Counter::Rv32Cycle, "rv32_cycle"},
          std::pair{Counter::RiscDebugWallClock, "risc_debug_wall_clock"}}) {
        EXPECT_STREQ(measurement::counter_name(counter), spelling);
    }
    EXPECT_THROW(measurement::counter_name(static_cast<Counter>(6)), std::invalid_argument);
    EXPECT_THROW(measurement::counter_name(static_cast<Counter>(UINT32_MAX)), std::invalid_argument);
}

// Spec (AiclkEndpoint): reference is unset until the sampler names the core's
// counter, and evidence whose start is unset serialises reference_counter null.
TEST(MeasurementAiclk, UnsetReferenceSerialisesAsNull) {
    EXPECT_FALSE(measurement::AiclkEndpoint{}.reference.has_value());
    auto start = endpoint(100, 1000);
    auto end = endpoint(20000100, 20001000);
    start.reference.reset();
    const auto evidence = measurement::assess_aiclk_run(1000000000, start, end);
    EXPECT_FALSE(evidence.reference.has_value());
    const auto json = x280::harness::output::clock_json(evidence);
    EXPECT_EQ(count_occurrences(json, "\"reference_counter\":null"), 1u) << json;
    EXPECT_EQ(json.find("tensix_live_timer"), std::string::npos) << json;
}
