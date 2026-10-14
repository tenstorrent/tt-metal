// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_PIPELINE_STAGE_SUITE_COMMON_HPP
#define X280_SUITES_PIPELINE_STAGE_SUITE_COMMON_HPP

#include "harness/artifacts/artifacts.hpp"

#include "pipeline_stage_common.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x280::suites::pipeline_stage::common {

// Returns the stable JSON label for a known instrumentation mode; unknown
// values map to "off" and are rejected by validate_measurement.
inline const char* instrumentation_name(uint32_t mode) {
    return mode == PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE ? "per_sample" : "off";
}

// Returns the stable diagnostic label for a terminal-state wire value, or
// "?" when the value is not part of the current schema.
inline const char* state_name(uint32_t state) {
    switch (state) {
        case PIPELINE_STAGE_STATE_INIT: return "INIT";
        case PIPELINE_STAGE_STATE_VERIFY_STAGE: return "VERIFY_STAGE";
        case PIPELINE_STAGE_STATE_WARMUP: return "WARMUP";
        case PIPELINE_STAGE_STATE_TIMED_PLAIN: return "TIMED_PLAIN";
        case PIPELINE_STAGE_STATE_TIMED_INSTRUMENTED: return "TIMED_INSTRUMENTED";
        case PIPELINE_STAGE_STATE_PUBLISH: return "PUBLISH";
        case PIPELINE_STAGE_STATE_DONE: return "DONE";
        case PIPELINE_STAGE_STATE_VERIFY_MISMATCH: return "VERIFY_MISMATCH";
        case PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH: return "ARG_LAYOUT_MISMATCH";
        case PIPELINE_STAGE_STATE_VERIFY_AFTER: return "VERIFY_AFTER";
        case PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED: return "WINDOW_CLEANUP_FAILED";
        case PIPELINE_STAGE_STATE_COUNTER_ERROR: return "COUNTER_ERROR";
    }
    return "?";
}

/**
 * The timed item count for one launch: PIPELINE_STAGE_N when set, otherwise
 * `default_count`, capped at PIPELINE_STAGE_MAX_SAMPLES under per-sample
 * instrumentation. Throws std::invalid_argument naming the value when a set N
 * exceeds that capacity under per-sample instrumentation, and env_u32's
 * std::runtime_error when N is not a decimal.
 */
inline uint32_t timed_count_from_env(uint32_t instrumentation, uint32_t default_count) {
    const bool per_sample = instrumentation == PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE;
    const auto requested = x280::harness::artifacts::env_u32("PIPELINE_STAGE_N");
    if (!requested) {
        return per_sample ? std::min(default_count, PIPELINE_STAGE_MAX_SAMPLES) : default_count;
    }
    if (per_sample && *requested > PIPELINE_STAGE_MAX_SAMPLES) {
        throw std::invalid_argument(
            "PIPELINE_STAGE_N=" + std::to_string(*requested) + " exceeds the per-sample capacity of " +
            std::to_string(PIPELINE_STAGE_MAX_SAMPLES) + " samples");
    }
    return *requested;
}

// Validates a completed raw-tick measurement and returns its optional sample
// vector, empty unless instrumented. `timed` is an item count; every accepted
// delta is strictly positive.
// Throws std::invalid_argument for invalid modes, counts, or sample metadata,
// and std::runtime_error for a zero batch or sample delta.
inline std::vector<uint64_t> validate_measurement(
    const pipeline_stage_result_t& result,
    const pipeline_stage_samples_t* samples,
    uint32_t timed,
    uint32_t instrumentation) {
    if (timed == 0u) {
        throw std::invalid_argument("timed count must be non-zero");
    }
    if (instrumentation != PIPELINE_STAGE_INSTRUMENTATION_OFF &&
        instrumentation != PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE) {
        throw std::invalid_argument("unknown instrumentation mode (" + std::to_string(instrumentation) + ")");
    }
    if (result.raw_batch_delta == 0u) {
        throw std::runtime_error("batch timed delta must be positive, got 0 ticks");
    }

    if (instrumentation != PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE) {
        return {};
    }
    if (samples == nullptr) {
        throw std::invalid_argument("samples pointer must be non-null for instrumented run");
    }
    if (timed > PIPELINE_STAGE_MAX_SAMPLES) {
        throw std::invalid_argument(
            "timed count (" + std::to_string(timed) + ") exceeds PIPELINE_STAGE_MAX_SAMPLES (" +
            std::to_string(PIPELINE_STAGE_MAX_SAMPLES) + ")");
    }
    if (samples->samples_completed != timed) {
        throw std::invalid_argument(
            "samples_completed (" + std::to_string(samples->samples_completed) + ") does not match timed count (" +
            std::to_string(timed) + ")");
    }

    std::vector<uint64_t> sample_deltas;
    sample_deltas.reserve(timed);
    for (uint32_t i = 0u; i < timed; ++i) {
        const uint64_t sample_delta = samples->raw_deltas[i];
        if (sample_delta == 0u) {
            throw std::runtime_error("sample delta at index " + std::to_string(i) + " must be positive, got 0 ticks");
        }
        sample_deltas.push_back(sample_delta);
    }
    return sample_deltas;
}

}  // namespace x280::suites::pipeline_stage::common

#endif  // X280_SUITES_PIPELINE_STAGE_SUITE_COMMON_HPP
